import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

client.moveByVelocityAsync(0, 1, 0, 13).join()
client.moveByVelocityAsync(0, -1, 0, 32).join()
client.moveByVelocityAsync(0, 1, 0, 19).join()

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
