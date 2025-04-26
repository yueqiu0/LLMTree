from tqdm import tqdm
import sys


class Logger:
    def __init__(self, file) -> None:
        self.file = file
        self.prefix = ""

    def log(self, msg: str):
        # 直接写入文件，确保超长内容完整输出
        self.file.write(self.prefix + msg + "\n")
        self.file.flush()


DEFAULT_LOGGERS = [Logger(sys.stderr)]


def log(msg: str):
    for logger in DEFAULT_LOGGERS:
        logger.log(msg)


def add_logger(logger: Logger):
    DEFAULT_LOGGERS.append(logger)
