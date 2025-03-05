from collections.abc import Generator
from pathlib import Path
import time

from .runner import Runner
from .parallel import ParallelRunner
from hugchat.login import Login
from hugchat import hugchat
from .. import logger


class HuggingChatRunner(Runner):
    def __init__(self, username: str, passwd: str, cookie_dir: str) -> None:
        self.username = username
        self.passwd = passwd
        self.cookie_dir = Path(cookie_dir)
        self.chatbot = None

    def run(self, messages: list[str]) -> Generator[list[str], None, None]:
        if self.chatbot is None:
            need_login = True
            if self.cookie_dir.exists():
                try:
                    sign = Login(self.username, None)
                    cookies = sign.loadCookiesFromDir(self.cookie_dir.as_posix())
                    need_login = False
                except:
                    pass

            if need_login:
                self.cookie_dir.mkdir(parents=True, exist_ok=True)
                sign = Login(self.username, self.passwd)
                cookies = sign.login()
                sign.saveCookiesToDir(self.cookie_dir.as_posix())

            self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

        for prompt in messages:
            retry_count = 0
            while True:
                try:
                    id = self.chatbot.new_conversation()
                    self.chatbot.change_conversation(id)
                    result = self.chatbot.chat(prompt, temperature=0.1)
                    self.chatbot.delete_conversation(id)
                    yield [result]
                    break
                except BaseException as e:
                    if f"{e}".find("too many messages") != -1:
                        retry_count += 1
                        # backoff
                        delay = 0.5 * 2**retry_count
                        logger.log(
                            f"Access too frequent, backoff for {delay} seconds..."
                        )
                        time.sleep(delay)
                        continue
                    raise e


class HuggingChatParallelRunner(ParallelRunner, HuggingChatRunner):
    def __init__(
        self,
        username: str,
        passwd: str,
        cookie_dir: str,
        interval: float,
        timeout: int,
        parallel_batch_size: int,
    ) -> None:
        HuggingChatRunner.__init__(self, username, passwd, cookie_dir)
        ParallelRunner.__init__(
            self, HuggingChatRunner.run, interval, timeout, parallel_batch_size
        )
