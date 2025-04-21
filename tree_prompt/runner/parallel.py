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
            logger.log(f"Process {idx} started with message length: {len(message)}")
            for results in self.runner(self, [message]):
                # 添加调试日志
                logger.log(f"Process {idx} received raw response: {type(results)}")
                if isinstance(results, (list, tuple)):
                    logger.log(f"Response length: {len(results)}")
                    if len(results) > 0:
                        logger.log(f"First item type: {type(results[0])}")
                
                # 确保结果正确传递
                if isinstance(results, str):
                    # 如果是字符串，确保不是被拆分的字符
                    if len(results) == 1:
                        logger.log(f"Warning: single character response detected: {results}")
                    queue.put((idx, [results]))  # 包装为列表
                else:
                    queue.put((idx, results))
        except BaseException as e:
            logger.log(f"Exception in process {idx}: {str(e)}")
            exit(1)

    def _run_batched(self, messages: list[str]) -> Generator[list[str], None, None]:
        queue = multiprocessing.Queue()
        test_idxs = list(range(len(messages)))
        processes = []
        all_results = []

        retry_count = -1
        total_start = time.time()



        while len(test_idxs) > 0:
            num_requests = len(test_idxs)
            retry_count += 1
            if retry_count > 0:
                logger.log(
                    "Retry #{retry_count}: retrying {num_requests} failed tests: {test_idxs}"
                )
            
            batch_start = time.time()
            # Reduced verbose logging for process management
            logger.log(f"Processing batch with {num_requests} requests")
            
            for idx in test_idxs:
                prompt = messages[idx]
                p = multiprocessing.Process(
                    target=self._run_wrapper,
                    args=(idx, queue, prompt),
                    name=f"RunnerProcess-{idx}",
                )
                p.start()
                processes.append((idx, p))
                if idx != test_idxs[-1]:
                    time.sleep(self.interval)

            # Wait for processes with minimal logging
            for idx, process in processes:
                logger.log(f"Waiting for process {idx} with timeout {self.timeout}s")
                process.join(timeout=self.timeout)
                if process.exitcode != 0:
                    logger.log(f"Process {idx} failed, exit code: {process.exitcode}, is_alive: {process.is_alive()}")
                    process.kill()
                elif process.is_alive():
                    logger.log(f"Process {idx} timed out, terminating")
                    process.terminate()
                test_idxs.remove(idx)

            processes = []
            processed_count = num_requests - len(test_idxs)
            
            for _ in range(processed_count):
                idx, result = queue.get()
                all_results.append({"idx": idx, "result": result})

        total_duration = time.time() - total_start
        logger.log(f"Batch processing completed in {total_duration:.2f}s")
        
        all_results.sort(key=lambda r: r["idx"])
        for result in [r["result"] for r in all_results]:
            logger.log(f"Yielding result of size {len(result)}")
            yield result

    def run(
        self,
        messages: list[str],
    ) -> Generator[list[str], None, None]:
        try:
            for start in range(0, len(messages), self.num_parallel):
                end = min(start + self.num_parallel, len(messages))
                batch = messages[start:end]
                logger.log(f"Starting batch {start}-{end} with {len(batch)} messages")

                for result in self._run_batched(batch):
                    logger.log(f"Yielding batch result of size {len(result)}")
                    yield result
        except KeyboardInterrupt:
            logger.log("Received KeyboardInterrupt, terminating all processes")
            raise
