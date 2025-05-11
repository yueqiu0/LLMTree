import openai
import time
from collections.abc import Generator

from .runner import Runner
from .parallel import ParallelRunner

from .. import logger


class OpenAIAPIRunner(Runner):
    def __init__(self, api_base: str, model_name: str, api_key: str, 
                 temperature: float = 0.0,  # 设为0以获得最确定性的输出
                 top_p: float = 0.0,        # 设为0以只考虑最可能的token
                 top_k: int = 1,            # 只选择概率最高的token
                 seed: int = 42,            # 固定随机种子
                 presence_penalty: float = 0.0,  # 不惩罚重复出现的token
                 frequency_penalty: float = 0.0, # 不惩罚频繁出现的token
                 ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def run(self, messages: list[str]) -> Generator[tuple[list[str], dict], None, None]:
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
                        max_tokens=4096,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        seed=self.seed,
                        presence_penalty=self.presence_penalty,
                        frequency_penalty=self.frequency_penalty,
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
                        token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        break
                    raise e

                result = response["choices"][0]["message"]["content"]

                
                # 提取token信息
                token_info = {
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "total_tokens": response["usage"]["total_tokens"]
                }
                
                
                break

            # 返回结果和token信息
            yield [result], token_info
    


class OpenAIAPIParallelRunner(ParallelRunner, OpenAIAPIRunner):
    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str,
        interval: float,
        timeout: int,
        parallel_batch_size: int,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        seed: int = 42,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> None:
        OpenAIAPIRunner.__init__(
            self, api_base, model_name, api_key, 
            temperature, top_p, top_k, seed,
            presence_penalty, frequency_penalty
        )
        ParallelRunner.__init__(
            self, OpenAIAPIRunner.run, interval, timeout, parallel_batch_size
        )
