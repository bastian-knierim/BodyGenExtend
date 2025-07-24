from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv
from .walker import WalkerEnv
from .pusher import PusherEnv


env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv,
    'walker': WalkerEnv,
    'pusher': PusherEnv
}