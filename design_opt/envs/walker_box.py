import numpy as np
from gym import utils
from design_opt.utils.rand import *
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
from khrylib.robot.xml_robot import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges
from copy import deepcopy
import mujoco_py
import time
import os


class WalkerPushEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, cfg, agent):
        self.cur_t = 0
        self.cfg = cfg
        self.task_specs = cfg.task_specs
        self.agent = agent
        if self.cfg.xml_name == "default":
            self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", "walkerbox.xml")
        else:
            self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", f"{self.cfg.xml_name}.xml")
        # robot xml
        self.robot = Robot(cfg.robot_cfg, xml=self.model_xml_file)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # design options
        self.clip_qvel = cfg.obs_specs.get('clip_qvel', False)
        self.use_projected_params = cfg.obs_specs.get('use_projected_params', True)
        self.abs_design = cfg.obs_specs.get('abs_design', False)
        self.use_body_ind = cfg.obs_specs.get('use_body_ind', False)
        self.use_body_depth_height = cfg.obs_specs.get('use_body_depth_height', False)
        self.use_shortest_distance = cfg.obs_specs.get('use_shortest_distance', False)
        self.use_position_encoding = cfg.obs_specs.get('use_position_encoding', False)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self.attr_design_dim = self.design_ref_params.shape[-1]
        add_body_condition = self.cfg.add_body_condition
        self.index_base = add_body_condition.get('index_base', 5)
        self.stage = 'skeleton_transform'    # transform or execute
        self.control_nsteps = 0
        self.sim_specs = set(cfg.obs_specs.get('sim', []))
        self.attr_specs = set(cfg.obs_specs.get('attr', []))
        # task attr
        if self.task_specs.get('mov_goal', True):
            self.goal_pos = np.array([0.0, 0.0])
            self.box_goal_dist = np.array([0.0, 0.0])
        self.box_pos = np.array(self.task_specs.get('box_pos'))
        self.rob_box_dist = np.array([0.0, 0.0, 0.0])
        MujocoEnv.__init__(self, self.model_xml_file, 4)
        utils.EzPickle.__init__(self)
        self._cache_box_addrs()
        self.box_id = self.model.body_name2id("box")
        self.control_action_dim = 1
        self.skel_num_action = 3 if cfg.enable_remove else 2
        self.sim_obs_dim = self.get_sim_obs().shape[-1]
        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]
        self.ground_geoms = np.where(self.model.geom_bodyid == 0)[0]

    def allow_add_body(self, body):
        add_body_condition = self.cfg.add_body_condition
        max_nchild = add_body_condition.get('max_nchild', 3)
        min_nchild = add_body_condition.get('min_nchild', 0)
        return body.depth >= self.cfg.min_body_depth and body.depth < self.cfg.max_body_depth - 1 and len(body.child) < max_nchild and len(body.child) >= min_nchild
    
    def allow_remove_body(self, body):
        if body.depth >= self.cfg.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            else:
                return True
        return False

    def apply_skel_action(self, skel_action):
        bodies = list(self.robot.bodies)
        for body, a in zip(bodies, skel_action):
            if a == 1 and self.allow_add_body(body):
                self.robot.add_child_to_body(body)
            if a == 2 and self.allow_remove_body(body):
                self.robot.remove_body(body)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
            self._cache_box_addrs()
        except:
            print(f'reload in apply_skel_action failed: ')
            print(self.cur_xml_str)
            return False      
        self.design_cur_params = self.get_attr_design()
        return True

    def set_design_params(self, in_design_params):
        design_params = in_design_params
        for params, body in zip(design_params, self.robot.bodies):
            body.set_params(params, pad_zeros=True, map_params=True)
            # new_params = body.get_params([], pad_zeros=True, demap_params=True)
            body.sync_node()

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
            self._cache_box_addrs()
        except:
            print(f'reload in set_design_params failed: ')
            print(self.cur_xml_str)
            return False
        if self.use_projected_params:
            self.design_cur_params = self.get_attr_design()
        else:
            self.design_cur_params = in_design_params.copy()
        return True

    def action_to_control(self, a):
        ctrl = np.zeros_like(self.data.ctrl)

        assert a.shape[0] == len(self.robot.bodies), f"action dim {a.shape} != num bodies {len(self.robot.bodies)}"
        # print(f'a: {[x for xs in a for x in xs]}')

        for body, body_a in zip(self.robot.bodies[1:], a[1:]):
            aname = body.get_actuator_name()
            aind = self.model.actuator_names.index(aname)
            ctrl[aind] = body_a
        return ctrl        

    def step(self, a):
        if not self.is_inited:
            return self._get_obs(), 0, False, False, {'use_transform_action': False, 'stage': 'execution'}

        self.cur_t += 1
        # skeleton transform stage
        if self.stage == 'skeleton_transform':
            skel_a = a[:, -1]
            succ = self.apply_skel_action(skel_a)
            if not succ:
                return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'skeleton_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps:
                self.transit_attribute_transform()

            ob = self._get_obs()
            reward = 0.0
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'skeleton_transform'}
        # attribute transform stage
        elif self.stage == 'attribute_transform':
            design_a = a[:, self.control_action_dim:-1] 
            if self.abs_design:
                design_params = design_a * self.cfg.robot_param_scale
            else:
                design_params = self.design_cur_params + design_a * self.cfg.robot_param_scale
            succ = self.set_design_params(design_params)
            if not succ:
                return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps + 1:
                succ = self.transit_execution()
                if not succ:
                    return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform'}

            ob = self._get_obs()
            reward = 0.0
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'attribute_transform'}
        # execution stage
        else:
            self.control_nsteps += 1
            assert np.all(a[:, self.control_action_dim:] == 0)
            control_a = a[:, :self.control_action_dim]
            ctrl = self.action_to_control(control_a)

            rob_pos_bef = self.get_body_com("0")[0:3].copy()
            box_pos_bef = self.get_body_com("box")[0:3].copy()

            if self.task_specs.get('mov_goal', False):
                box_goal_dist_bef = np.linalg.norm(box_pos_bef[:2] - self.goal_pos)
            else:
                rob_box_dist_bef = np.linalg.norm(rob_pos_bef - box_pos_bef)

            # print(f'ctrl: {ctrl}')
            try:
                self.do_simulation(ctrl, self.frame_skip)
            except:
                print(f'do_simulation in step failed: ')
                print(self.cur_xml_str)
                return self._get_obs(), 0, True, False, {'use_transform_action': False, 'stage': 'execution'}

            rob_pos_aft = self.get_body_com("0")[0:3].copy()
            box_pos_aft = self.get_body_com("box")[0:3].copy()
            rob_box_dist_aft = np.linalg.norm(box_pos_aft - rob_pos_aft)
            self.rob_box_dist = box_pos_aft - rob_pos_aft
            
            if self.task_specs.get('mov_goal', False):
                box_goal_dist_aft = np.linalg.norm(box_pos_aft - self.goal_pos)
                reward = (box_goal_dist_bef - box_goal_dist_aft) /self.dt
            else:
                reward = (box_pos_aft[0] - box_pos_bef[0]) /self.dt

            reward += (rob_box_dist_bef - rob_box_dist_aft) / self.dt

            reward += self.cfg.reward_specs.get('alive_bonus', 0.0)
            scale = self.cfg.reward_specs.get('exec_reward_scale', 1.0)
            reward *= scale

            height, ang = self.sim.data.qpos[1:3]
            s = self.state_vector()
            # misc
            done_condition = self.cfg.done_condition
            min_height = done_condition.get('min_height', 0.7)
            max_height = done_condition.get('max_height', 2.0)
            max_ang = done_condition.get('max_ang', 3600)
            max_nsteps = done_condition.get('max_nsteps', 1000)
            termination = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))
            truncation = not (self.control_nsteps < max_nsteps)
            # if termination:
            #     print(f'termination cause:')
            #     if not (np.isfinite(s).all()):
            #         print('s is not finite: {s}')
            #     elif not (height > min_height):
            #         print(f'height {height} < min_height {min_height}')
            #     elif not (height < max_height):
            #         print(f'height {height} > max_height {max_height}')
            #     elif not (abs(ang) < np.deg2rad(max_ang)):
            #         print(f'ang {abs(ang)} > max_ang {np.deg2rad(max_ang)}')
            if self.task_specs.get('mov_goal', False) and box_goal_dist_bef < 1.0:
                self.reset_state(True) # TODO make it more beautiful (set flag if done -> reset in the beginning)
            ob = self._get_obs()
            return ob, reward, termination, truncation, {'use_transform_action': False, 'stage': 'execution'}
    
    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
        try:
            self.reset_state(True)
        except:
            print(f'reset_state in transit_execution failed: ')
            print(self.cur_xml_str)
            return False
        # try:
        #     mujoco_py.cymj._mj_forward(self.model, self.data)
        # except Exception as e:
        #     print("mj_forward failed:", e)
        # self.log_model_layout(header="AFTER SKELETON TRANSFORM")
        # self.model.geom_rgba[self.box_id][3] = 1.0
        return True
        

    def if_use_transform_action(self):
        return ['skeleton_transform', 'attribute_transform', 'execution'].index(self.stage)

    def get_sim_obs(self):
        rob_obs = []
        env_obs = []
        if 'root_offset' in self.sim_specs:
            root_pos = self.data.body_xpos[self.model._body_name2id[self.robot.bodies[0].name]]
        # if self.stage == 'execution': print(f'bodies in robot.bodies: {self.robot.bodies}')
        for i, body in enumerate(self.robot.bodies):
            qvel = self.data.qvel.copy()
            if self.clip_qvel:
                qvel = np.clip(qvel, -10, 10)
            if i == 0:
                obs_i = [np.flip(self.data.qpos[1:3]), np.flip(qvel[:3])]
            else:
                qs, qe = get_single_body_qposaddr(self.model, body.name)
                assert qe - qs == 1
                obs_i = [self.data.qpos[qs:qe], np.zeros(1), qvel[qs:qe], np.zeros(2)]
                # print(qs)
            if 'root_offset' in self.sim_specs:
                offset = self.data.body_xpos[self.model._body_name2id[body.name]][[0, 2]] - root_pos[[0, 2]]
                obs_i.append(offset)
            obs_i = np.concatenate(obs_i)
            rob_obs.append(obs_i)
        rob_obs = np.stack(rob_obs)

        for i in range(rob_obs.shape[0]):
            if i == 0:
                obs_i = [self.rob_box_dist, self.data.qpos[-1:], qvel[-3:]]
            elif i == 1 and self.task_specs.get('mov_goal', True):
                obs_i = [self.box_goal_dist, np.zeros(5)]
            else:
                obs_i = [np.zeros(7)]
            obs_i = np.concatenate(obs_i)
            env_obs.append(obs_i)
        env_obs = np.stack(env_obs)
        obs = np.concatenate([rob_obs, env_obs], axis=-1)
        if self.control_nsteps == 1:
            assert np.count_nonzero(obs[:, :5]) == self.model.nq + self.model.nv - 7 # 7 statt 1

        return obs

    def get_attr_fixed(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = []
            if 'depth' in self.attr_specs:
                obs_depth = np.zeros(self.cfg.max_body_depth)
                obs_depth[body.depth] = 1.0
                obs_i.append(obs_depth)
            if 'jrange' in self.attr_specs:
                obs_jrange = body.get_joint_range()
                obs_i.append(obs_jrange)
            if 'skel' in self.attr_specs:
                obs_add = self.allow_add_body(body)
                obs_rm = self.allow_remove_body(body)
                obs_i.append(np.array([float(obs_add), float(obs_rm)]))
            if len(obs_i) > 0:
                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)
        
        if len(obs) == 0:
            return None
        obs = np.stack(obs)
        return obs

    def get_attr_design(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs

    def get_body_index(self):
        index = []
        for i, body in enumerate(self.robot.bodies):
            ind = int(body.name, base=self.index_base)
            index.append(ind)
        index = np.array(index)
        return index
    
    def get_body_height(self):
        heights = []
        for i, body in enumerate(self.robot.bodies):
            h = body.height
            heights.append(h)
        heights = np.array(heights)
        return heights
        
    def get_body_depth(self):
        depths = []
        for i, body in enumerate(self.robot.bodies):
            d = body.depth
            depths.append(d)
        depths = np.array(depths)
        return depths

    def _get_obs(self):
        obs = []
        attr_fixed_obs = self.get_attr_fixed()
        sim_obs = self.get_sim_obs()
        design_obs = self.design_cur_params
        obs = np.concatenate(list(filter(lambda x: x is not None, [attr_fixed_obs, sim_obs, design_obs])), axis=-1)
        if self.cfg.obs_specs.get('fc_graph', False):
            edges = get_graph_fc_edges(len(self.robot.bodies))
        else:
            edges = self.robot.get_gnn_edges()
        use_transform_action = np.array([self.if_use_transform_action()])
        num_nodes = np.array([sim_obs.shape[0]])
        all_obs = [obs, edges, use_transform_action, num_nodes]
        if self.use_body_ind:
            body_index = self.get_body_index()
            all_obs.append(body_index)
        if self.use_body_depth_height:
            body_depths = self.get_body_depth()
            all_obs.append(body_depths)
            body_heights = self.get_body_height()
            all_obs.append(body_heights)
        if self.use_shortest_distance:
            distances = self.robot.get_shortest_distances()
            all_obs.append(distances)
        if self.use_position_encoding:
            lapPE = self.robot.get_laplacian_position_encoding()
            all_obs.append(lapPE)
        return all_obs

    def reset_state(self, add_noise):
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel

        if self.task_specs.get('mov_goal', False):
            self.box_pos[0] = rand_val()
            self.goal_pos[0] = rand_val()
            qpos[self._box_qadr[0]] = self.box_pos[0]
            qpos[self._box_qadr[1]] = self.box_pos[1]
            qpos[self._box_qadr[2]] = self.box_pos[2]
        else:
            qpos[self._box_qadr[0]] = self.box_pos[0]
            qpos[self._box_qadr[1]] = self.box_pos[1]
            qpos[self._box_qadr[2]] = self.box_pos[2]

        if self.stage == 'execution' and self.cfg.env_init_height:
            qpos[1] = 0.0
            while True:
                self.set_state(qpos, qvel)
                has_contact = False
                for contact in self.data.contact[:self.data.ncon]:
                    g1, g2 = contact.geom1, contact.geom2
                    # print(f'g1: {g1} g2: {g2}')
                    if g1 in self.ground_geoms or g2 in self.ground_geoms:
                        has_contact = True
                        break
                if has_contact:
                    qpos[1] += 0.05
                else:
                    break
        else:
            self.set_state(qpos, qvel)
            
        self.rob_box_dist = self.get_body_com("box")[0:3].copy() #TODO hübscher machen

    def reset_robot(self):
        del self.robot
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        self.reload_sim_model(self.cur_xml_str)
        self._cache_box_addrs()
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()

    def reset_model(self):
        self.reset_robot()
        self.control_nsteps = 0
        self.stage = 'skeleton_transform'
        self.cur_t = 0
        self.reset_state(False)
        # print(f'============================== MODEL RESET ==============================')
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 15
        # self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.lookat[0] = self.data.qpos[0] 
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 110

    def _cache_box_addrs(self):
        names = ("boxx", "boxz", "box_yaw")
        self._box_jids, self._box_qadr, self._box_vadr = [], [], []
        for n in names:
            jid = self.model.joint_name2id(n)
            self._box_jids.append(jid)
            self._box_qadr.append(int(self.model.jnt_qposadr[jid]))
            self._box_vadr.append(int(self.model.jnt_dofadr[jid]))

    def log_model_layout(self, header="MODEL LAYOUT"):
        m, d = self.model, self.data
        print(f"\n===== {header} =====")
        print(f"nq (qpos size): {m.nq}")
        print(f"nv (qvel size): {m.nv}")
        print(f"nu (ctrl  size): {m.nu}")

        # erste Werte aus qpos/qvel zur NaN/Inf-Prüfung
        qpos_head = np.array2string(d.qpos[:min(10, m.nq)], precision=3)
        qvel_head = np.array2string(d.qvel[:min(10, m.nv)], precision=3)
        print(f"qpos[:10]: {qpos_head}")
        print(f"qvel[:10]: {qvel_head}")

        # Aktuatoren (nur zur Kontrolle; deine Box hat keine)
        try:
            actuators = [m.actuator_id2name(i) for i in range(m.nu)]
        except Exception:
            # Fallback für ältere mujoco_py, falls id2name fehlt
            actuators = list(getattr(m, "actuator_names", []))
        print(f"Actuators ({m.nu}): {actuators}")

        # Optional: Box-Joint-Adressen (falls vorhanden)
        for jn in ("boxx", "boxz", "box_yaw"):
            try:
                jid = m.joint_name2id(jn)
                qadr = int(m.jnt_qposadr[jid])
                vadr = int(m.jnt_dofadr[jid])
                print(f"Joint '{jn}': jid={jid}, qpos_adr={qadr}, qvel_adr={vadr}")
            except Exception:
                print(f"Joint '{jn}' not found.")

        # Control-Range (zum späteren Clippen hilfreich)
        if m.nu > 0 and hasattr(m, "actuator_ctrlrange"):
            lo = m.actuator_ctrlrange[:, 0]
            hi = m.actuator_ctrlrange[:, 1]
            print(f"ctrl range lo[:5]: {lo[:min(5, m.nu)]}")
            print(f"ctrl range hi[:5]: {hi[:min(5, m.nu)]}")

        # NaN/Inf Check
        print("finite(qpos):", np.isfinite(d.qpos).all(), "  finite(qvel):", np.isfinite(d.qvel).all())
        print("===== END =====\n")
