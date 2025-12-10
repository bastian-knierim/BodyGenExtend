from abc import ABC, abstractmethod
import numpy as np

class Task(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, env):
        """Darf env.data.qpos/qvel anpassen (über Joint-Offsets),
        Ziele platzieren, interne Variablen setzen. Muss danach mj_forward auslösen (env.set_state ruft das)."""
        pass

    @abstractmethod
    def pre_step(self, env):
        """Optional: Dinge vor der Simulation messen/cachen (z.B. Abstände vorher)."""
        ...

    @abstractmethod
    def post_step(self, env, ctrl, info: dict) -> tuple[float, bool, bool, dict]:
        """
        Muss Reward/Termination/Truncation und evtl. Info zurückgeben.
        env hat nach do_simulation() den neuen Zustand.
        """
        ...
