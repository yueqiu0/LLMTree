from collections.abc import Generator


class Runner:
    def run(self, input: list[str]) -> Generator[list[str], None, None]:
        raise NotImplementedError
