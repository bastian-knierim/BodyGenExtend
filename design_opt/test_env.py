import os
import sys

sys.path.append(os.getcwd())

from khrylib.robot.xml_robot import Robot
from design_opt.utils.config import Config
from omegaconf import OmegaConf
import hydra

project_path = os.getcwd()

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    cfg = Config(cfg, project_path)

    # Nur der Teil, den du brauchst
    robot_cfg = cfg.robot_cfg
    xml_path = os.path.join(cfg.project_path, "assets", "mujoco_envs", "pusher.xml")
    robot = Robot(robot_cfg, xml=xml_path)

    print("Körper:")
    for i, body in enumerate(robot.bodies):
        print(f"[{i}] name={body.name}, depth={body.depth}")


if __name__ == "__main__":
    main()
