class _Repr:
    def __repr__(self):
        return self.__dict__.__repr__()


class DatasetArgs(_Repr):
    def __init__(self) -> None:
        self.data_file: str = None
        self.meta_file: str = None
        self.format: str = None
        self.shuffle_column: bool = None


class OpenAIAPIArgs(_Repr):
    def __init__(self) -> None:
        self.api_base: str = ""
        self.model_name: str = None
        self.openai_api_key: str = ""
        self.request_interval: float = 0.2
        self.timeout: int = 30
        self.parallel_batch_size: int = 6


class HuggingChatArgs(_Repr):
    def __init__(self) -> None:
        self.hf_username: str = None
        self.hf_password: str = None
        self.hf_cookie_dir: str = ".huggingchat"
        self.request_interval: float = 1
        self.timeout: int = 120
        self.parallel_batch_size: int = 6


class TogetherAPIArgs(_Repr):
    def __init__(self) -> None:
        self.api_base: str = ""
        self.model_name: str = None
        self.together_api_key: str = ""
        self.request_interval: float = 0.2
        self.timeout: int = 30
        self.parallel_batch_size: int = 6
