import logging
import logging.config
import json

with open("log_config.json") as fp:
    logging.config.dictConfig(json.load(fp))
logging.getLogger("evo_controller").info("dit is info")
logging.getLogger("evo_controller").error("dit is error")
logging.info("dit is root info")
logging.error("dit is root error")
