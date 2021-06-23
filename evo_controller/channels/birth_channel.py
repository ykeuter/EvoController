from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.side_channel import IncomingMessage
import uuid


class BirthChannel(SideChannel):
    def __init__(self, population):
        super().__init__(uuid.UUID("51d41610-6239-4ef1-98c4-cad4b8d70a17"))
        self.population = population

    def on_message_received(self, msg: IncomingMessage) -> None:
        # print("birth msg received")
        child_id = msg.read_int32()
        parent1_id = msg.read_int32(-1)
        parent2_id = msg.read_int32(-1)
        self.population.add_agent(child_id, parent1_id, parent2_id)
