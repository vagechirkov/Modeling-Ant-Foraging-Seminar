from typing import Tuple

import numpy as np
from mesa import Model
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.datacollection import DataCollector

from agent import Ant
from utils import diffuse


class StigmergyModel(Model):
    def __init__(
        self,
        width: int = 101,
        height: int = 101,
        population: int = 125,
        speed: float = 1.0,
        kappa: float = 5.0,
        diffusion_rate: float = 25.0,
        evaporation_rate: float = 5.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width, self.height = width, height

        # Continuous space
        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=True,
            random=self.random,
            n_agents=population,
        )

        # Grid fields
        self.pheromone = np.zeros((width, height), dtype=float)
        self.food = np.zeros((width, height), dtype=int)
        self.nest = np.zeros((width, height), dtype=bool)

        # Environment parameters
        self.diff_coeff = diffusion_rate / 100.0
        self.evap_factor = (100.0 - evaporation_rate) / 100.0

        # Build world & agents
        self._setup_patches()
        self._create_ants(population, kappa, speed)

        self.datacollector = DataCollector(
            model_reporters={
                "total_pheromone": lambda m: m.pheromone.sum(),
                "remaining_food": lambda m: m.food.sum(),
            }
        )

    def _setup_patches(self) -> None:
        cx, cy = self.width / 2, self.height / 2
        self._nest_center = (cx, cy)
        for x in range(self.width):
            for y in range(self.height):
                dist = np.hypot(x - cx, y - cy)
                if dist < 5:
                    self.nest[x, y] = True

        rng, radius = self.random, 5

        def _fill_circle(center: Tuple[float, float]):
            cx, cy = center
            for x in range(self.width):
                for y in range(self.height):
                    if np.hypot(x - cx, y - cy) < radius:
                        self.food[x, y] = rng.choice([1, 2])

        _fill_circle((0.8 * self.width, self.height / 2))  # right
        _fill_circle((0.2 * self.width, 0.2 * self.height))  # lower‑left
        _fill_circle((0.1 * self.width, 0.9 * self.height))  # upper‑left

    def _create_ants(self, n: int, kappa: float, speed: float):
        nest_pos = np.array([self.width / 2, self.height / 2])
        Ant.create_agents(
            self,
            n,
            self.space,
            position=np.tile(nest_pos, (n, 1)),
            speed=speed,
            kappa=kappa,
        )

    def _update_pheromone(self):
        self.pheromone = diffuse(self.pheromone, rate=self.diff_coeff)
        self.pheromone *= self.evap_factor

    def step(self) -> None:
        self.agents.shuffle_do("step")  # perform agent steps in a randomized order
        self._update_pheromone()
        self.datacollector.collect(self)
