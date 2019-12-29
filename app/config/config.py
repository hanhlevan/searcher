configFile = "./app/config/service.conf"
from app.models.config_reader import ConfigReader
configer = ConfigReader(configFile)
configer.parseFile()