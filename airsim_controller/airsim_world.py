import airsim
import numpy as np


class AirsimWorld:
    SPEED = 1.
    DURATION = 1.
    UP = np.array([0, 1, 0])
    FORWARD = np.array([1, 0, 0])
    RIGHT = np.array([0, 0, 1])
    ACTIONS = [FORWARD, RIGHT, -FORWARD, -RIGHT]
    CAMS = ["front", "right", "rear", "left"]

    def __init__(self):
        self.client = None

    def connect(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def disconnect(self):
        self.client.enableApiControl(False)

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

    def step(self, action):
        v = self.ACTIONS[np.argmax(action)] * self.SPEED
        self.client.moveByVelocityAsync(v[0], v[1], v[2], self.DURATION).join()
        observations = self.perceive()
        reward = self.client.simGetHealth()
        done = False
        info = {}
        return observations, reward, done, info

    def perceive(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest(cam, airsim.ImageType.Scene, True)
            for cam in self.CAMS
        ])
        results = [np.mean(r.image_data_float) for r in responses]
        return results
