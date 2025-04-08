from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence
import logging

from datasets import Dataset

from verifiers import RewardFunc
from ..imports import LLM, SamplingParams  # type: ignore


class Environment(ABC):
    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.tokenizer = None
        self.dataset = None
        self.eval_dataset = None
        self.pad_token_id = kwargs.get("pad_token_id", 151643)
        self.eos_token_id = kwargs.get("eos_token_id", 151645)
        self.new_line_token_id = kwargs.get("new_line_token_id", 198)
        self.reward_funcs = []
        self.reward_weights = []

    @abstractmethod
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        pass

    @abstractmethod
    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Dict[str, List[list[int]] | List[str] | List[List[Dict[str, Any]]]]:
        pass
