from .base_net import BaseNet
from pytorch_neat.recurrent_net import RecurrentNet


class NeatRecurrentNet(BaseNet):
    @staticmethod
    def make_net(genome, config, bs):
        return RecurrentNet.create(genome, config, bs)
