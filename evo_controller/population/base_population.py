from abc import ABC, abstractmethod


class BasePopulation(ABC):
    @abstractmethod
    def add_agent(id, parent1_id=0, parent2_id=0):
        pass

    @abstractmethod
    def remove_agent(id):
        pass

    @abstractmethod
    def activate(obs):
        pass
