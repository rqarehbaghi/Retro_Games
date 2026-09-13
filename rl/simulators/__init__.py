"""
Afterstate lookahead simulator registry.

Every registered name maps to the GENERIC GridPlacementSimulator; a game's
pieces, board size and reward parameters come from its `afterstate` block in
games.json, not from code. To register a genuinely different mechanism (one the
generic grid-drop simulator cannot express), add its class here.
"""
from typing import Any, Dict, Optional, Type

from rl.simulators.base import BaseAfterstateSimulator
from rl.simulators.grid_placement import GridPlacementSimulator

_SIMULATORS: Dict[str, Type[BaseAfterstateSimulator]] = {
    "grid_placement": GridPlacementSimulator,
    # Backwards-compatible aliases for configs that still name "tetris".
    "tetris": GridPlacementSimulator,
    "tetris_lookahead": GridPlacementSimulator,
    "TetrisTime-Nes-v0": GridPlacementSimulator,
}


def register_simulator(name: str, cls: Type[BaseAfterstateSimulator]):
    """Register a custom simulator class for a game."""
    _SIMULATORS[name] = cls


def get_simulator(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                  **kwargs: Any) -> Optional[BaseAfterstateSimulator]:
    """Instantiate a simulator by name, handing it the game's afterstate config
    (from games.json) so it knows the board and the pieces. None if unregistered."""
    if not name:
        return None
    cls = _SIMULATORS.get(name)
    if cls is None:
        return None
    return cls(config=config, **kwargs)
