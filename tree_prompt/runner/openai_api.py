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
        logger.log(f"[OpenAI API] Starting with {len(messages)} messages")
        logger.log(f"[OpenAI API] Model: {self.model_name}")
        logger.log(f"[OpenAI API] Base URL: {self.api_base}")

        for i, prompt in enumerate(messages):
            logger.log(f"[OpenAI API] Processing message {i+1}/{len(messages)}")
            client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            stream = False
            max_tokens=2048

            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    logger.log(f"[OpenAI API] Attempt {retry_count + 1} for message {i+1}")
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        presence_penalty=self.presence_penalty,
                        frequency_penalty=self.frequency_penalty,
                        seed=self.seed,
                        stream=False,
                        max_tokens=max_tokens,
                        extra_body={}
                    )
                except BaseException as e:
                    err_msg = f"{e}"
                    logger.log(f"[OpenAI API] Error on attempt {retry_count + 1}: {err_msg}")
                    
                    if retry_count >= max_retries - 1:
                        logger.log(f"[OpenAI API] Max retries reached, giving up")
                        raise e
                    
                    if err_msg.find("Rate limit reached") != -1 or err_msg.find("rate limit") != -1:
                        retry_count += 1
                        delay = 2 ** retry_count  # Exponential backoff
                        logger.log(f"[OpenAI API] Rate limit hit, waiting {delay} seconds...")
                        time.sleep(delay)
                        continue
                    elif err_msg.find("maximum context length") != -1 or err_msg.find("context length") != -1:
                        logger.log("[OpenAI API] Context length exceeded, skipping...")
                        result = ""
                        token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        break
                    elif err_msg.find("timeout") != -1 or err_msg.find("Timeout") != -1:
                        retry_count += 1
                        delay = 2 ** retry_count
                        logger.log(f"[OpenAI API] Timeout, retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        retry_count += 1
                        delay = 2 ** retry_count
                        logger.log(f"[OpenAI API] Unknown error, retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue

                # Success - normalize response
                result = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                token_info = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                }
                logger.log(f"[OpenAI API] Success for message {i+1}, tokens: {token_info['total_tokens']}")

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
