# import setup_path 
import airsim

import numpy as np

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
# 
client.takeoffAsync().join()
client.moveByVelocityAsync(1, 0, 0, 50).join()
# client.moveByVelocityAsync(0, 5, 0, 5).join()
# client.moveByVelocityAsync(0, 0, 5, 5).join()

responses = client.simGetImages([
    airsim.ImageRequest("front", airsim.ImageType.Scene, True)
])  #scene vision image in uncompressed RGBA array
response = responses[0]
print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
airsim.write_pfm('test.pfm', airsim.get_pfm_array(response))

client.enableApiControl(False)
