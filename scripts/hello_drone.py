import airsim
import math
import time

client = airsim.MultirotorClient()


def waitAndGetHealth(seconds):
    print("Health: {}".format(client.simGetHealth()), end="\r")
    for _ in range(math.ceil(seconds)):
        time.sleep(1)
        print("Health: {}".format(client.simGetHealth()), end="\r")


client.confirmConnection()
client.reset()

client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

client.moveByVelocityAsync(0, 1, 0, 13)
waitAndGetHealth(1.5 * 13 / 3)
client.moveByVelocityAsync(0, -1, 0, 30)
waitAndGetHealth(1.5 * 30 / 3)
client.moveByVelocityAsync(0, 1, 0, 17)
waitAndGetHealth(1.5 * 17 / 3)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
