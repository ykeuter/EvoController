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

client.enableApiControl(True, "Drone1")
client.enableApiControl(True, "Drone2")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Drone2")
client.takeoffAsync(vehicle_name="Drone1")
client.takeoffAsync(vehicle_name="Drone2").join()

client.moveByVelocityAsync(0, 1, 0, 13, vehicle_name="Drone1")
client.moveByVelocityAsync(0, 1, 0, 13, vehicle_name="Drone2")
waitAndGetHealth(1.5 * 13 / 3)
client.moveByVelocityAsync(0, -1, 0, 30, vehicle_name="Drone1")
client.moveByVelocityAsync(0, -1, 0, 30, vehicle_name="Drone2")
waitAndGetHealth(1.5 * 30 / 3)
client.moveByVelocityAsync(0, 1, 0, 17, vehicle_name="Drone1")
client.moveByVelocityAsync(0, 1, 0, 17, vehicle_name="Drone2")
waitAndGetHealth(1.5 * 17 / 3)

client.landAsync(vehicle_name="Drone1")
client.landAsync(vehicle_name="Drone2").join()
client.armDisarm(False, "Drone1")
client.armDisarm(False, "Drone2")
client.enableApiControl(False, "Drone1")
client.enableApiControl(False, "Drone2")
