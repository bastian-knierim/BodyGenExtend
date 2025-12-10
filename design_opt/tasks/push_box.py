# design_opt/tasks/push_box.py
import numpy as np
from .base_task import Task

class PushBox(Task):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.mov_goal = self.task_specs.get('mov_goal', False)
        if self.mov_goal == True:
            self.goal_pos = None

    def reset(self, env):
        rng = env.np_random
        if self.mov_goal:
            box_xy = rng.uniform(low=-2.0, high=2.0, size=2)
            goal_xy = rng.uniform(low=-2.0, high=2.0, size=2)
            self.goal_pos = np.array([goal_xy[0], 0.5, goal_xy[1]])
        else:
            box_xy = np.array(self.task_specs.get('box_pos'))

        env.data.qpos[env.box_qpos_adr: env.box_qpos_adr+3] = [box_xy[0], box_xy[1], 0.5]

        # if hasattr(env, "arrow_id"):
        #     env.model.body_pos[env.arrow_id] = self.goal_pos
        env.sim_forward()

    def pre_step(self, env):
        # cache Distanzen "vorher"
        self.rob_bef = env.get_body_com("0")[:3].copy()
        self.box_bef = env.get_body_com("box")[:3].copy()
        self.rob_box_dist_bef = np.linalg.norm(self.box_bef - self.rob_bef)
        if self.mov_goal:
            self.box_goal_dist_bef = np.linalg.norm(self.box_bef - self.goal_pos)

    def post_step(self, env, ctrl, info):
        # „nachher“-Werte
        rob_aft = env.get_body_com("0")[:3].copy()
        box_aft = env.get_body_com("box")[:3].copy()
        rob_box_dist_aft = np.linalg.norm(box_aft - rob_aft)
        dt = env.dt
        r_robo_box = (self.rob_box_dist_bef - rob_box_dist_aft) / dt

        if self.mov_goal:
            box_goal_dist_aft = np.linalg.norm(box_aft - self.goal_pos)
            r_task = (self.box_goal_dist_bef - box_goal_dist_aft) / dt
        else:
            r_task = (box_aft[0] - self.box_bef[0]) / dt

        ctrl_coeff = self.cfg.reward_specs.get('ctrl_cost_coeff', 1e-4)
        r_ctrl = -ctrl_coeff * np.square(ctrl).mean()
        alive = self.cfg.reward_specs.get('alive_bonus', 0.0)

        reward = r_task + r_robo_box + r_ctrl + alive
        reward *= self.cfg.reward_specs.get('exec_reward_scale', 1.0)
        return reward

    def done_condition(self):
        if self.mov_goal and self.box_goal_dist_bef < 1.0:
            return True
        return False
