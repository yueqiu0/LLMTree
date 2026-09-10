import together
from collections.abc import Generator

from .runner import Runner
from .parallel import ParallelRunner


class TogetherAPIRunner(Runner):
    def __init__(self, api_base: str, model_name: str, api_key: str) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name

    def run(self, messages: list[str]) -> Generator[list[str], None, None]:
        if self.api_base:
            together.api_base = self.api_base
        together.api_key = self.api_key

        for prompt in messages:
            while True:
                try:
                    response = together.Complete.create(
                        prompt=prompt,
                        model=self.model_name,
                        max_tokens=2048,
                        temperature=0.0,
                        top_k=50,
                        top_p=0.7,
                    )
                except BaseException as e:
                    raise e

                result = response["output"]["choices"][0]["text"]
                break

            yield [result]


class TogetherAPIParallelRunner(ParallelRunner, TogetherAPIRunner):
    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str,
        interval: float,
        timeout: int,
        parallel_batch_size: int,
    ) -> None:
        TogetherAPIRunner.__init__(self, api_base, model_name, api_key)
        ParallelRunner.__init__(
            self, TogetherAPIRunner.run, interval, timeout, parallel_batch_size
        )
