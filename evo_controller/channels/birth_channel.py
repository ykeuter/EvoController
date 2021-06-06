from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.side_channel import IncomingMessage


class BirthChannel(SideChannel):
    def __init__(self, channel_id, population):
        super().__init__(channel_id)
        self.population = population

    def on_message_received(self, msg: IncomingMessage) -> None:
        child_id = msg.read_int32()
        parent1_id = msg.read_int32()
        parent2_id = msg.read_int32()
        self.population.add_agent(child_id, parent1_id, parent2_id)
