import numpy as np
from mesa import Model
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.datacollection import DataCollector

from agent import BoltzmannWalkerAnt

class AntModel(Model):
    def __init__(self, width=100, height=100, N=50, mean_free_path=0.01, g=0.3, seed=None):
        super().__init__(seed=seed)
        self.space = ContinuousSpace(
            [[0, width], [0, height]], torus=True, random=self.random, n_agents=N
        )
        # Create agents
        BoltzmannWalkerAnt.create_agents(
            self,
            N,
            self.space,
            position=np.array((width / 2, height / 2) * N),
            mean_free_path=mean_free_path,
            g=g
        )

        # Data collector
        self.datacollector = DataCollector(
            agent_reporters={"x": "x", "y": "y", "heading": "heading"},
        )


    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)