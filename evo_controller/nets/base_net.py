from abc import ABC, abstractmethod


class BaseNet(ABC):
    @staticmethod
    def activate_net(net, states):
        outputs = net.activate(states).cpu().numpy()
        return outputs

    @staticmethod
    @abstractmethod
    def make_net(genome, config, bs):
        pass
