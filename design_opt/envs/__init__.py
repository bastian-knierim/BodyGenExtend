from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv
from .walker import WalkerEnv
from .ant_box import AntPushEnv
from .walker_box import WalkerPushEnv
from .swimmer_box import SwimmerPushEnv
from .ant_box_flip import AntFlipEnv
from .ant_box_lift import AntLiftEnv
from .ant_ import AntEnv_


env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv,
    'walker': WalkerEnv,
    'ant_box': AntPushEnv,
    'walker_box': WalkerPushEnv,
    'swimmer_box': SwimmerPushEnv,
    'ant_box_flip': AntFlipEnv,
    'ant_box_lift': AntLiftEnv,
    'ant_': AntEnv_
}