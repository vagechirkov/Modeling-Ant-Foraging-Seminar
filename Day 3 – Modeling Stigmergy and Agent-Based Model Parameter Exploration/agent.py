from typing import Tuple

import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent, ContinuousSpace


class Ant(ContinuousSpaceAgent):
    """Continuous‑space ant with random search and pheromone perception."""

    # Angular constants (degrees) to mimic NetLogo *uphill* checks
    _RIGHT: int = 45
    _LEFT: int = -45

    def __init__(
        self,
        model,
        space: ContinuousSpace,
        position: Tuple[float, float],
        speed: float = 1.0,
        kappa: float = 5.0,
    ) -> None:
        super().__init__(space, model)
        self.position = np.asarray(position, dtype=float)
        self.heading: float = self.random.uniform(-np.pi, np.pi)  # radians
        self.speed: float = float(speed)

        # Random‑Walker parameters
        self.kappa = float(kappa)

        # Foraging state
        self.carrying: bool = False

    @property
    def _ixiy(self) -> Tuple[int, int]:
        """Current grid cell index (wrap torus)."""
        x, y = self.position
        return int(x) % self.model.width, int(y) % self.model.height

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    def _move(self, distance: float | None = None) -> None:
        """Translate by *distance* along current heading (default = self.speed)."""
        if distance is None:
            distance = self.speed
        dx = distance * np.cos(self.heading)
        dy = distance * np.sin(self.heading)
        self.position += (dx, dy)
        self.position = self.space.torus_correct(tuple(self.position))

    def _random_search(self) -> None:
        self._move()
        self.heading += self.random.vonmisesvariate(0, kappa=5.0)

    def _pickup_food(self) -> bool:
        ix, iy = self._ixiy
        if self.model.food[ix, iy] > 0:
            self.model.food[ix, iy] -= 1
            self.carrying = True
            return True
        return False

    def _return_to_nest(self) -> None:
        """Ballistic homing: aim directly at a nest center and move forward."""
        nest_x, nest_y = self.model._nest_center
        vec_x = nest_x - self.position[0]
        vec_y = nest_y - self.position[1]

        # Short‑circuit if we are already at nest cell
        if self.model.nest[
            int(self.position[0]) % self.model.width,
            int(self.position[1]) % self.model.height,
        ]:
            self.carrying = False
            self.heading += np.pi  # head back out next tick
            return

        # Orient towards nest
        self.heading = np.atan2(vec_y, vec_x)

        # Deposit pheromone before moving (so the trail covers the current patch too)
        self._deposit_pheromone()
        self._move()

    def _deposit_pheromone(self) -> None:
        ix, iy = self._ixiy
        self.model.pheromone[ix, iy] += 60.0

    def _sample_deg(self, field: np.ndarray, deg: int) -> float:
        """Sample field one unit ahead turned by deg degrees."""
        theta = self.heading + np.radians(deg)
        x = self.position[0] + np.cos(theta)
        y = self.position[1] + np.sin(theta)
        return float(field[int(x) % self.model.width, int(y) % self.model.height])

    def _uphill(self, field: np.ndarray) -> None:
        """Turn ±45° toward the highest neighbouring cell in *field*."""
        ahead = self._sample_deg(field, 0)
        right = self._sample_deg(field, self._RIGHT)
        left = self._sample_deg(field, self._LEFT)
        if (right > ahead) or (left > ahead):
            self.heading += np.radians(self._RIGHT if right > left else self._LEFT)

    def _check_pheromones(self) -> bool:
        ix, iy = self._ixiy
        pheromone_here = self.model.pheromone[ix, iy]
        if pheromone_here > 0.05:
            self._uphill(self.model.pheromone)
            self._move()
            return True
        return False

    def step(self) -> None:
        if self.carrying:
            self._return_to_nest()
            return

        if self._pickup_food():
            # finish step if food is found
            return

        if self._check_pheromones():
            # finish step if pheromone tracking is performed
            return

        # Perform an uninformed search as a fallback if neither food nor pheromone cues are found
        self._random_search()
