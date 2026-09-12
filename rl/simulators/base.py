"""
Abstract base class for game-specific afterstate lookahead simulators.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseAfterstateSimulator(ABC):
    """
    Interface for game-specific afterstate generators.
    Any game that supports 1-step lookahead or afterstate evaluation implements this.
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """The dimensionality of the afterstate feature vector."""
        pass

    @abstractmethod
    def get_candidates(
        self,
        obs: np.ndarray,
        ram: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        vars: Optional[Any] = None,
        spec: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate legal candidate actions and their resulting afterstate representations.

        Returns:
            List of candidate dictionaries:
            [
              {
                "action": int,               # Discrete action index to execute
                "afterstate": np.ndarray,    # 1D float32 feature vector of resulting state
                "immediate_reward": float,   # Optional immediate transition reward
              },
              ...
            ]
        """
        pass
