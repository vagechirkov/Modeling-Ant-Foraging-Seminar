import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent

class Ant(ContinuousSpaceAgent):
    def __init__(
            self,
            model,
            space,
            position=(0, 0),
            speed=1,
            direction=(1, 1)
    ):
        super().__init__(space, model)
        self.position = position
        self.speed = speed
        self.direction = direction

    def step(self):
        # correlated random walk
        # self.direction = None

        # Normalize direction vector
        self.direction /= np.linalg.norm(self.direction)

        self.position += self.direction * self.speed