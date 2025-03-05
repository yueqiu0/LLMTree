import openai
import time
from collections.abc import Generator

from .runner import Runner
from .parallel import ParallelRunner

from .. import logger


class OpenAIAPIRunner(Runner):
    def __init__(self, api_base: str, model_name: str, api_key: str) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name

    def run(self, messages: list[str]) -> Generator[list[str], None, None]:
        if self.api_base:
            openai.api_base = self.api_base
        openai.api_key = self.api_key
        for prompt in messages:
            message = {"role": "user", "content": prompt}

            retry_count = 0
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[message],
                        max_tokens=2048,
                        temperature=0.0,  # for reproducibility
                    )
                # TODO: real error handling
                except BaseException as e:
                    err_msg = f"{e}"
                    if err_msg.find("Rate limit reached") != -1:
                        retry_count += 1
                        # backoff
                        delay = 0.5 * 2**retry_count
                        logger.log(
                            f"Access too frequent, backoff for {delay} seconds..."
                        )
                        time.sleep(delay)
                        continue
                    elif err_msg.find("maximum context length") != -1:
                        logger.log("Exceed context length, skipping...")
                        result = ""
                        break
                    raise e

                result = response["choices"][0]["message"]["content"]
                # usage = response["usage"]
                # print(usage)
                break

            yield [result]


class OpenAIAPIParallelRunner(ParallelRunner, OpenAIAPIRunner):
    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str,
        interval: float,
        timeout: int,
        parallel_batch_size: int,
    ) -> None:
        OpenAIAPIRunner.__init__(self, api_base, model_name, api_key)
        ParallelRunner.__init__(
            self, OpenAIAPIRunner.run, interval, timeout, parallel_batch_size
        )
