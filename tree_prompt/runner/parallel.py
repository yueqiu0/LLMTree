import multiprocessing
import time
from collections.abc import Generator

from .runner import Runner
from .. import logger


class ParallelRunner(Runner):
    def __init__(
        self, runner: callable, interval: float, timeout: float, num_parallel: int
    ) -> None:
        self.timeout = timeout
        self.interval = interval
        self.runner = runner
        self.num_parallel = num_parallel

    def _run_wrapper(
        self,
        idx: int,
        queue: multiprocessing.Queue,
        message: str,
    ):
        try:
            for results in self.runner(self, [message]):
                queue.put((idx, results))
        except BaseException as e:
            logger.log("Exception occured in runner {}: {}".format(idx, e))
            exit(1)

    def _run_batched(self, messages: list[str]) -> Generator[list[str], None, None]:
        queue = multiprocessing.Queue()
        test_idxs = list(range(len(messages)))
        processes = []
        all_results = []

        retry_count = -1

        while len(test_idxs) > 0:
            num_requests = len(test_idxs)
            retry_count += 1
            if retry_count > 0:
                logger.log(
                    "Retrying {} failed tests: {}".format(num_requests, test_idxs)
                )
            for idx in test_idxs:
                prompt = messages[idx]
                p = multiprocessing.Process(
                    target=self._run_wrapper,
                    args=(idx, queue, prompt),
                )
                p.start()
                processes.append((idx, p))
                if idx != test_idxs[-1]:
                    time.sleep(self.interval)

            for idx, process in processes:
                process.join(timeout=self.timeout)
                if process.exitcode == 0:
                    test_idxs.remove(idx)
                else:
                    process.kill()

            processes = []

            for _ in range(num_requests - len(test_idxs)):
                idx, result = queue.get()
                all_results.append({"idx": idx, "result": result})

        all_results.sort(key=lambda r: r["idx"])
        for result in [r["result"] for r in all_results]:
            yield result

    def run(
        self,
        messages: list[str],
    ) -> Generator[list[str], None, None]:
        for start in range(0, len(messages), self.num_parallel):
            end = min(start + self.num_parallel, len(messages))
            batch = messages[start:end]

            for result in self._run_batched(batch):
                yield result
