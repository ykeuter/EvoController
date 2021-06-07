from abc import ABC, abstractmethod


class BasePopulation(ABC):
    @abstractmethod
    def add_agent(self, id, parent1_id=0, parent2_id=0):
        pass

    @abstractmethod
    def terminate(self, terminal_steps):
        pass

    @abstractmethod
    def activate(self, decision_steps):
        pass
