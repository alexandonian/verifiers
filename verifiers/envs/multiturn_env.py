import random
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List, TypedDict

from datasets import Dataset
from pydantic import BaseModel
from tqdm import tqdm
from vllm.outputs import RequestOutput

from verifiers.envs.environment import Environment
from verifiers.inference.vllm_client import VLLMClient
from verifiers.utils import format_dataset

from ..imports import LLM, SamplingParams  # type: ignore


class ChatOutput(BaseModel):
    token_ids: List[int]
    text: str


class ChatResponseItem(BaseModel):
    prompt_token_ids: List[int]
    outputs: List[ChatOutput]


class ChatResponse(BaseModel):
    responses: List[ChatResponseItem]


class State(TypedDict):
    messages: list[dict[str, str]]
    prompt_messages: int
    prompt_ids: list[int]
    completed: bool
    completion_ids: list[int]
    completion_mask: list[int]


def dict_to_chat_response(data: Dict[str, Any]) -> ChatResponse:
    """
    Recursively convert a dictionary to a ChatResponse object
    """
    # First, convert all outputs to ChatOutput objects
    if "responses" in data:
        for i, response_item in enumerate(data["responses"]):
            if "outputs" in response_item:
                data["responses"][i]["outputs"] = [
                    ChatOutput(**output) for output in response_item["outputs"]
                ]

        # Then convert all response items to ChatResponseItem objects
        data["responses"] = [ChatResponseItem(**item) for item in data["responses"]]

    # Finally, convert the entire dict to a ChatResponse object
    return ChatResponse(**data)


def request_output_to_chat_response(data: list[RequestOutput]) -> ChatResponse:
    """
    Convert a list of RequestOutput objects to a ChatResponse object.
    Each RequestOutput is converted to a ChatResponseItem.
    """
    responses = []
    for output in data:
        prompt_token_ids = output.prompt_token_ids or []
        outputs = [
            ChatOutput(token_ids=list(resp.token_ids), text=resp.text)
            for resp in output.outputs
        ]
        responses.append(
            ChatResponseItem(prompt_token_ids=prompt_token_ids, outputs=outputs)
        )

    return ChatResponse(responses=responses)


class MultiTurnEnv(Environment):
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str = "",
        few_shot: List[Dict[str, str]] | None = None,
        sampling_args: Dict[str, Any] | None = None,
        mask_env_response: bool = True,
        max_workers: int = 10,
        max_steps: int = 10,
        sleep_time: float = 0.1,
        **kwargs,
    ):
        if few_shot is None:
            few_shot = []
        if sampling_args is None:
            sampling_args = {}

        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        if dataset is not None:
            self.dataset = format_dataset(
                dataset=dataset,
                system_prompt=self.system_prompt,
                few_shot=self.few_shot,
            )
        else:
            self.dataset = None
        if eval_dataset is not None:
            self.eval_dataset = format_dataset(
                dataset=eval_dataset,
                system_prompt=self.system_prompt,
                few_shot=few_shot,
            )
        else:
            self.eval_dataset = None
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
        }
        self.sampling_args |= sampling_args
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.sleep_time = sleep_time
        self.max_steps = max_steps

    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.dataset is not None:
            return self.dataset.shuffle(seed=seed).select(range(n))  # type: ignore
        return self.dataset

    def get_eval_dataset(
        self, n: int = -1, seed: int = 0, **kwargs: Any
    ) -> Dataset | None:
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle(seed=seed).select(range(n))  # type: ignore
        return self.eval_dataset

    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, str]:
        pass

    def step(
        self,
        states: list[State],
        llm: LLM | VLLMClient,
        sampling_params: SamplingParams,
    ) -> List[State]:
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]

        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = 2048

        assert sampling_params.max_tokens > 0, (
            "max_tokens must be greater than 0 for sampling parameters."
        )

        if isinstance(llm, VLLMClient):
            llm_responses = llm.chat(
                messages_to_step,
                n=1,
                repetition_penalty=sampling_params.repetition_penalty,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                min_p=sampling_params.min_p,
                max_tokens=sampling_params.max_tokens,
                stop=sampling_params.stop,  # type: ignore
                include_stop_str_in_output=sampling_params.include_stop_str_in_output,
                skip_special_tokens=sampling_params.skip_special_tokens,
                spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
            )  # type: ignore
            llm_responses = dict_to_chat_response(llm_responses).responses
        else:
            llm_responses = llm.chat(
                messages_to_step,  # type: ignore
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            llm_responses = request_output_to_chat_response(llm_responses).responses

        # for i, j in enumerate(live_indices):
        def update_state(j, llm_response: ChatResponseItem) -> tuple[int, State]:
            try:
                # sleep for 0-1 seconds to avoid rate limiting
                time.sleep(self.sleep_time * random.random())

                state: State = deepcopy(states[j])
                if len(state["prompt_ids"]) == 0:
                    state["prompt_ids"] = llm_response.prompt_token_ids

                state["messages"].append(
                    {"role": "assistant", "content": llm_response.outputs[0].text}
                )

                # get token lengths of env response and new completion
                total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
                num_prev_prompt_ids = len(state["prompt_ids"])
                num_prompt_tokens = len(list(llm_response.prompt_token_ids))
                env_response_len = num_prompt_tokens - total_prev_len
                new_completion_len = len(llm_response.outputs[0].token_ids)

                # update completion masks
                state["completion_mask"].extend([self.env_mask] * env_response_len)
                state["completion_mask"].extend([1] * new_completion_len)

                # update completion ids
                state["completion_ids"] = list(llm_response.prompt_token_ids)
                state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
                state["completion_ids"] = state["completion_ids"][num_prev_prompt_ids:]

                if (
                    state["completion_ids"][-1] != self.new_line_token_id
                    and state["completion_ids"][-2] != self.eos_token_id
                ):
                    state["completion_ids"].append(self.eos_token_id)
                    state["completion_ids"].append(self.new_line_token_id)
                    state["completion_mask"].append(1)
                    state["completion_mask"].append(1)

                if len(state["completion_ids"]) > len(state["completion_mask"]):
                    n = len(state["completion_ids"]) - len(state["completion_mask"])
                    state["completion_mask"].extend([1] * n)

                if len(state["completion_mask"]) > len(state["completion_ids"]):
                    n_id = len(state["completion_ids"])
                    state["completion_mask"] = state["completion_mask"][n_id:]

                assert sampling_params.max_tokens is not None

                if (
                    self.is_completed(state["messages"])
                    or len(state["completion_ids"]) > sampling_params.max_tokens - 1
                ):
                    state["completed"] = True
                    state["completion_ids"] = state["completion_ids"][
                        : sampling_params.max_tokens
                    ]
                    state["completion_mask"] = state["completion_mask"][
                        : len(state["completion_ids"])
                    ]
                else:
                    state["messages"].append(self.env_response(state["messages"]))

                # enforce that the completion mask and completion ids are the same length
                # weird bug that happens rarely and only for certain models; something tokenizer related :(
                if len(state["completion_mask"]) != len(state["completion_ids"]):
                    print(state["messages"])
                    print(state["completion_mask"])
                    print(state["completion_ids"])
                    min_len = min(
                        len(state["completion_mask"]), len(state["completion_ids"])
                    )
                    state["completion_mask"] = state["completion_mask"][:min_len]
                    state["completion_ids"] = state["completion_ids"][:min_len]

                return j, state

            except Exception as e:
                print(f"Error in update_state for index {j}: {str(e)}")
                # Return the original state to avoid losing data
                return j, states[j]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(update_state, j, llm_responses[i]): j
                for i, j in enumerate(live_indices)
            }
            # Collect results with timeout handling
            results = []
            task_timeout = 60  # 60 second timeout per task

            for future in tqdm(
                as_completed(futures, timeout=None),  # No overall timeout
                total=len(futures),
                desc="Stepping through states",
                unit="state",
            ):
                j = futures[future]
                try:
                    result = future.result(timeout=task_timeout)
                    results.append(result)
                except TimeoutError:
                    print(f"Task for state {j} timed out after {task_timeout} seconds")
                    # Keep the original state and mark as completed to prevent further processing
                    states[j]["completed"] = True
                    results.append((j, states[j]))
                except Exception as e:
                    print(f"Task for state {j} raised an exception: {e}")
                    # Keep the original state and mark as completed
                    states[j]["completed"] = True
                    results.append((j, states[j]))

        for j, state in results:
            states[j] = state

        return states

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM | VLLMClient,
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Dict[str, List[list[int]] | List[str] | List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states: list[State] = [
            {
                "messages": m,
                "prompt_messages": len(m),
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
            }
            for m in prompts
        ]

        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"] :] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        return {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
        }

    def step_api(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        sampling_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.

        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API

        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        if sampling_args is None:
            sampling_args = {}
        messages_copy = deepcopy(messages)

        try:
            # Get assistant response
            response = client.chat.completions.create(
                model=model, messages=messages_copy, extra_body=sampling_args
            )

            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            messages_copy.append(assistant_msg)

            # Check if we're done
            if self.is_completed(messages_copy):
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, get and add environment response
                env_msg = self.env_response(messages_copy)
                messages_copy.append(env_msg)

            return messages_copy, rollout_is_completed

        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True

    def eval_api(
        self,
        client: Any,
        model: str,
        max_concurrent: int = 32,
        timeout: int = 60,
        sampling_args: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        eval_sampling_args = deepcopy(self.sampling_args)
        eval_sampling_args.update(sampling_args)
        """
        Evaluate model using OpenAI API with proper concurrency.

        Args:
            client: OpenAI client instance
            model: Model name as string
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling (separate from env sampling_args)
            **kwargs: Additional arguments for evaluation

        Returns:
            Tuple of (eval_dataset, rewards)
        """

        def run_evaluation():
            # Import libraries here to avoid requiring them for normal operation
            import asyncio
            from asyncio import Semaphore

            # Get the evaluation dataset
            if self.eval_dataset is None:
                self.eval_dataset = self.get_eval_dataset(**kwargs)

            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")

            eval_dataset = self.eval_dataset

            async def process_example(example, semaphore):
                async with semaphore:
                    # Initialize conversation with system prompt and few-shot examples
                    prompt = example["prompt"]
                    messages = deepcopy(example["prompt"])
                    answer = example["answer"]

                    # Save the length of initial messages to extract just the interaction part later
                    initial_length = len(messages)

                    # Run the conversation loop until completion or max steps
                    for _ in range(
                        self.max_steps
                    ):  # Safety limit on conversation turns
                        try:
                            # Run step_api to get model and environment response
                            # Note: step_api now returns a tuple (messages, is_completed)
                            step_result = (
                                await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: self.step_api(
                                        client=client,
                                        model=model,
                                        messages=messages,
                                        sampling_args=eval_sampling_args,
                                    ),
                                )
                            )

                            # Unpack the step_api result
                            messages, is_completed = step_result

                            # If the rollout is completed, break the loop
                            if is_completed:
                                break

                        except Exception as e:
                            print(
                                f"Error processing example {example.get('id', 'unknown')}: {str(e)}"
                            )
                            break

                    # Extract only the interaction part (not system/few-shot)
                    completions = messages[initial_length:]

                    return {
                        "prompt": prompt,
                        "completions": completions,
                        "task": example["task"],
                        "answer": answer,
                    }

            async def run_all_examples():
                # Create semaphore for concurrency control
                from tqdm.asyncio import tqdm_asyncio

                semaphore = Semaphore(max_concurrent)

                # Process all examples concurrently
                tasks = [
                    process_example(example, semaphore) for example in eval_dataset
                ]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples",
                )

                return results

            # Run the async evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()

            # Calculate rewards
            results_prompt = [result["prompt"] for result in results]
            results_answer = [result["answer"] for result in results]
            results_task = [result["task"] for result in results]
            results_completions = [result["completions"] for result in results]
            results = {
                "prompt": results_prompt,
                "answer": results_answer,
                "completions": results_completions,
                "task": results_task,
            }

            reward_funcs = self.get_reward_funcs()
            rewards = {}
            avg_rewards = {}

            for reward_func in reward_funcs:
                func_rewards = reward_func(**results)  # type: ignore
                func_rewards = [fr for fr in func_rewards if fr is not None]
                func_reward_avg = sum(func_rewards) / max(1, len(func_rewards))
                func_name = reward_func.__name__  # type: ignore
                print(f"{func_name}: {func_reward_avg}")
                avg_rewards[func_name] = func_reward_avg
                rewards[func_name] = func_rewards

            return {"avg_rewards": avg_rewards, "rewards": rewards, "results": results}

        # Run the evaluation function
        return run_evaluation()
