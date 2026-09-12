"""
Game-specific afterstate lookahead simulator registry.
"""
from typing import Any, Dict, Optional, Type
from rl.simulators.base import BaseAfterstateSimulator
from rl.simulators.tetris import TetrisSimulator

_SIMULATORS: Dict[str, Type[BaseAfterstateSimulator]] = {
    "tetris": TetrisSimulator,
    "tetris_lookahead": TetrisSimulator,
    "TetrisTime-Nes-v0": TetrisSimulator,
}


def register_simulator(name: str, cls: Type[BaseAfterstateSimulator]):
    """Register a custom simulator class for a game."""
    _SIMULATORS[name] = cls


def get_simulator(name: Optional[str] = None, **kwargs: Any) -> Optional[BaseAfterstateSimulator]:
    """
    Retrieve an instantiated simulator by name (as configured in games.json).
    Returns None if no simulator is registered for that name.
    """
    if not name:
        return None
    cls = _SIMULATORS.get(name)
    if cls is None:
        return None
    return cls(**kwargs)
