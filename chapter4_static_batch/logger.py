
from enum import Enum


class LogLevel(Enum):
    INFO = 1
    WARN = 2
    ERROR = 3


class ModelLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.logLevel = LogLevel.INFO
        self.EnableLog = False
        # 需要拆分成几个类型的日志
    
    def info(self, *args):
        if (self.EnableLog == False or self.logLevel.value > LogLevel.INFO.value):
            return
        
        print('[INFO]:', *args)


logger = ModelLogger()
