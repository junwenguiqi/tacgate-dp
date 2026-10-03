"""Independent tactile gate policy release package."""

from .config import PolicyConfig

__version__ = "0.1.0"
__all__ = ["PolicyConfig", "TactileGatePolicy"]


def __getattr__(name):
    if name == "TactileGatePolicy":
        from .policy import TactileGatePolicy

        return TactileGatePolicy
    raise AttributeError(name)
