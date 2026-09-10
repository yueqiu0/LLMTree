from tqdm import tqdm
import sys


class Logger:
    def __init__(self, file) -> None:
        self.file = file
        self.prefix = ""

    def log(self, msg: str):
        tqdm.write(self.prefix + msg, self.file)


DEFAULT_LOGGERS = [Logger(sys.stderr)]


def log(msg: str):
    for logger in DEFAULT_LOGGERS:
        logger.log(msg)


def add_logger(logger: Logger):
    DEFAULT_LOGGERS.append(logger)
