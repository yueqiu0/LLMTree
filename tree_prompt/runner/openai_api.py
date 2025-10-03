import openai
from openai import OpenAI
import time
from collections.abc import Generator

from .runner import Runner
from .parallel import ParallelRunner

from .. import logger
from tree_prompt.model.strategy import clean_llm_response


class OpenAIAPIRunner(Runner):
    def __init__(self, api_base: str, model_name: str, api_key: str, 
                 temperature: float = 0.0,
                 top_p: float = 0.0,
                 top_k: int = 1,
                 seed: int = 42,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0,
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
        

        for prompt in messages:
            client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            stream = False
            max_tokens=2048

            retry_count = 0
            while True:
                try:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        presence_penalty=self.presence_penalty,
                        frequency_penalty=self.frequency_penalty,
                        seed=self.seed,
                        stream=False,
                        extra_body={}
             
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

                # Normalize response (object style)
                result = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                token_info = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                }

                result = clean_llm_response(result)
                
                
                break


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
