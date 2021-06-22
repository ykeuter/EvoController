from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.side_channel import IncomingMessage
import uuid
import logging


class AgeChannel(SideChannel):
    def __init__(self, log_stream=None):
        super().__init__(uuid.UUID("130f04e4-fb3d-41b6-bc6a-2b796678dd47"))
        self.logger = logging.getLogger(__name__)

    def on_message_received(self, msg: IncomingMessage) -> None:
        print("age msg received")
        age = msg.read_float32()
        self.logger.info(age)
