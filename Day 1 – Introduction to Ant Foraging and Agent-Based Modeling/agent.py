import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent


class BoltzmannWalkerAnt(ContinuousSpaceAgent):
    """Boltzmann‑Walker ant after Khuong et al.(2013)."""

    def __init__(self, model, space, position=(0, 0), mean_free_path=5.0, g=0.3):
        super().__init__(space, model)
        self.position = np.asarray(position, dtype=float)
        self.x, self.y = self.position
        # initial heading uniformly in (–π, π]
        self.heading = self.random.uniform(-np.pi, np.pi)
        # Input parameters of Alg. 2
        self.mean_free_path = float(mean_free_path)
        self.g = float(g)

    def _relliptic(self):
        """
        Ref: Alg.2 Khuong et al.(2013)
        """
        if not (-1 < self.g < 1):
            raise ValueError("g must lie in (–1,1)")
        # 1. gratio ← (1–g)/(1+g)
        gratio = (1 - self.g) / (1 + self.g)
        # 2. tmp ← tan(U·π/2) · gratio,  U∼U(0,1)
        tmp = np.tan(self.random.random() * np.pi / 2) * gratio
        # 3. sgn ← sign(V),  V∼U(–1,1)
        sgn = 1 if self.random.uniform(-1, 1) >= 0 else -1
        # 4. Δθ ← 2·sgn·atan(tmp)
        return 2.0 * sgn * np.arctan(tmp)  # uniform in (–π, π)

    def step(self):
        """
        Ref: Alg.4 Khuong et al.(2013)
        """
        # Draw free‑path length ℓ from an exponential law
        l = self.random.expovariate(1 / self.mean_free_path)

        # # Update position along the current heading
        self.position += np.array([l * np.cos(self.heading), l * np.sin(self.heading)])
        self.position = self.space.torus_correct(tuple(self.position))
        self.x, self.y = self.position

        # Turn by Δθ drawn from the elliptical angular sampling
        self.heading += self._relliptic()
