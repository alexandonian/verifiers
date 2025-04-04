import math
import json
from typing import Callable, Dict, List, Optional

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.rubrics.math_grader import grade


class CodeToolRubric(Rubric):
    def __init__(
        self,
        parser: XMLParser = XMLParser(fields=["reasoning", ("code", "tool", "answer")]),
        env_parser: XMLParser = XMLParser(fields=["code_result", "tool_result"]),
        tools: List[Callable] | None = None,
    ):
        if tools is None:
            tools = []
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.__name__: tool for tool in tools}
        self.reward_funcs = [
            self.mc_reward_func,
            self.math_reward_func,
            self.code_reward_func,
            self.code_execution_reward_func,
            self.correct_answer_reward_func,
            self.tool_execution_reward_func,
            self.code_performance_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]
        self.reward_weights = [
            0.0,
            0.0,
            0.0,
            0.5,
            1.0,
            0.5,
            0.25,
            0.25,
            0.25,
        ]
        for tool_name in self.tools.keys():
            self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
            self.reward_weights.append(0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        import io
        import signal
        import sys
        from contextlib import redirect_stdout

        try:
            test_cases = json.loads(answer)["test_cases"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return 0.0
        # strip ```python and ``` if present at the beginning and end of the code
        code_str = code_str.strip()
        if code_str.startswith("```python"):
            code_str = code_str[9:]
        elif code_str.startswith("```"):
            code_str = code_str[3:]
        if code_str.endswith("```"):
            code_str = code_str[:-3]
        code_str = code_str.strip()

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        def normalize_output(output):
            # Normalize line endings and whitespace
            return "\n".join(line.strip() for line in output.splitlines())

        total_cases = 0
        passed = 0

        for test in test_cases:
            output = io.StringIO()
            sys.stdin = io.StringIO(test["input"])
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                with redirect_stdout(output):
                    exec(code_str)
                signal.alarm(0)
                actual = normalize_output(output.getvalue())
                expected = normalize_output(test["output"])

                # Compare each line individually
                actual_lines = actual.splitlines()
                expected_lines = expected.splitlines()
                total_cases += len(expected_lines)
                for a, e in zip(actual_lines, expected_lines):
                    if a == e:
                        passed += 1

            except Exception as e:
                sys.stdin = sys.__stdin__
                return 0.0
            sys.stdin = sys.__stdin__

        return passed / total_cases if total_cases else 0.0

    def code_reward_func(
        self, completions, answer, task, **kwargs
    ) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "code":
                response = str(self.get_last_answer(completion))
                reward = self.evaluate_code(response, ans, **kwargs)
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def mc_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "mc":
                response = str(self.get_last_answer(completion))  # [0]
                if len(response.strip()) > 0 and isinstance(response, str):
                    response = response.strip()[0]
                reward = 1.0 if response == ans.strip() else 0.0
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def math_reward_func(
        self, completions, answer, task, **kwargs
    ) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "math":
                response = str(self.get_last_answer(completion))
                try:
                    reward = 1.0 if grade(response, ans) else 0.0
                except:
                    reward = 0.0
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def correct_answer_reward_func(
        self, completions, answer, task, **kwargs
    ) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            reward = None
            if t == "mc":
                try:
                    reward = self.mc_reward_func([completion], [ans], [t], **kwargs)[0]
                except Exception:
                    reward = None
            elif t == "math":
                try:
                    reward = self.math_reward_func([completion], [ans], [t], **kwargs)[
                        0
                    ]
                except Exception:
                    reward = None
            elif t == "code":
                try:
                    reward = self.code_reward_func([completion], [ans], [t], **kwargs)[
                        0
                    ]
                except Exception:
                    reward = None
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def tool_execution_reward_func(
        self, completions: List[List[Dict[str, str]]], **kwargs
    ) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """

        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0

            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg["role"] == "assistant":
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if (
                            i + 1 < len(trajectory)
                            and trajectory[i + 1]["role"] == "user"
                        ):
                            tool_attempts += 1
                            # Check response with env_parser
                            multiplier = 1.0
                            response = str(parsed.tool)
                            if (("sympy" in response) or ("numpy" in response)) and len(
                                response
                            ) > 100:
                                multiplier = 1.5
                            else:
                                multiplier = 0.5
                            parsed_response = self.env_parser.parse(
                                trajectory[i + 1]["content"]
                            )
                            if (
                                hasattr(parsed_response, "tool_result")
                                and parsed_response.tool_result is not None
                                and not parsed_response.tool_result.startswith("Error:")
                            ):
                                successful_executions += 1 * multiplier

            # Calculate reward
            return 0.0 if tool_attempts == 0 else successful_executions / tool_attempts

        return [check_execution(c) for c in completions]

    def get_named_tool_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """

        def tool_reward_func(
            completions: List[List[Dict[str, str]]], **kwargs
        ) -> List[float]:
            """
            Reward function that checks execution success for the {tool_name} tool.

            Uses XMLParser to identify proper tool calls for the specified tool.
            """
            import json

            def check_tool_execution(trajectory: List[Dict[str, str]]) -> float:
                tool_attempts = 0
                successful_executions = 0

                # Find assistant messages with the specific tool and their responses
                for i, msg in enumerate(trajectory):
                    if msg["role"] == "assistant":
                        # Use parser to check for tool tag
                        parsed = self.parser.parse(msg["content"])
                        if hasattr(parsed, "tool") and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if (
                                    isinstance(command, dict)
                                    and command.get("name") == tool_name
                                ):
                                    # Found a properly formatted tool message for the specific tool
                                    if (
                                        i + 1 < len(trajectory)
                                        and trajectory[i + 1]["role"] == "user"
                                    ):
                                        tool_attempts += 1
                                        # Check response with env_parser
                                        parsed_response = self.env_parser.parse(
                                            trajectory[i + 1]["content"]
                                        )
                                        if (
                                            hasattr(parsed_response, "tool_result")
                                            and parsed_response.tool_result is not None
                                            and not parsed_response.tool_result.startswith(
                                                "Error:"
                                            )
                                        ):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass

                # Calculate reward
                return (
                    0.0 if tool_attempts == 0 else successful_executions / tool_attempts
                )

            return [check_tool_execution(c) for c in completions]

        # Create a function with the dynamic name based on tool_name
        tool_reward_func.__name__ = f"{tool_name}_reward_func"
        return tool_reward_func

    def get_named_tool_count_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """

        def tool_count_reward_func(
            completions: List[List[Dict[str, str]]], **kwargs
        ) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                successful_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg["role"] == "assistant":
                        parsed = self.parser.parse(msg["content"])
                        if hasattr(parsed, "tool") and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if (
                                    isinstance(command, dict)
                                    and command.get("name") == tool_name
                                ):
                                    # Found a properly formatted tool message for the specific tool
                                    if (
                                        i + 1 < len(trajectory)
                                        and trajectory[i + 1]["role"] == "user"
                                    ):
                                        parsed_response = self.env_parser.parse(
                                            trajectory[i + 1]["content"]
                                        )
                                        if (
                                            hasattr(parsed_response, "tool_result")
                                            and parsed_response.tool_result is not None
                                            and not parsed_response.tool_result.startswith(
                                                "Error:"
                                            )
                                        ):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                return successful_executions

            return [count_tool_executions(c) for c in completions]

        tool_count_reward_func.__name__ = f"{tool_name}_count_reward_func"
        return tool_count_reward_func

    def get_named_tool_attempt_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """

        def tool_attempt_reward_func(
            completions: List[List[Dict[str, str]]], **kwargs
        ) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                attempted_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg["role"] == "assistant":
                        parsed = self.parser.parse(msg["content"])
                        if hasattr(parsed, "tool") and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if (
                                    isinstance(command, dict)
                                    and command.get("name") == tool_name
                                ):
                                    attempted_executions += 1
                            except json.JSONDecodeError:
                                pass
                return attempted_executions

            return [count_tool_executions(c) for c in completions]

        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func

    def code_execution_reward_func(
        self,
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        """Reward function that checks code execution success at each step."""

        def check_execution(trajectory: List[Dict[str, str]]) -> float:
            total_code_steps = 0
            successful_executions = 0

            for i, msg in enumerate(trajectory):
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "code") and parsed.code is not None:
                        total_code_steps += 1
                        # Look for the next user message (environment response)
                        if (
                            i + 1 < len(trajectory)
                            and trajectory[i + 1]["role"] == "user"
                        ):
                            env_response = trajectory[i + 1]["content"]
                            parsed_response = self.env_parser.parse(
                                env_response, strict=False
                            )
                            if (
                                hasattr(parsed_response, "code_result")
                                and parsed_response.code_result
                            ):
                                output = parsed_response.code_result
                                if len(output) > 0 and not output.startswith("Error:"):
                                    successful_executions += 1

            # Return proportional reward based on successful executions
            if total_code_steps == 0:
                return 0.0
            return 0.3 * (successful_executions / total_code_steps) + 0.05 * (
                successful_executions
            )

        return [check_execution(c) for c in completions]

    def code_performance_reward_func(
        self,
        completions: List[List[Dict[str, str]]],
        reference_metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Reward function that evaluates code execution performance at each step.

        Performance is measured by the execution time and memory usage, with adaptive
        scaling based on problem complexity and statistical normalization across solutions.
        """

        def check_performance(trajectory: List[Dict[str, str]]) -> float:
            # Track performance metrics for all executions in this trajectory
            performance_metrics = []

            for i, msg in enumerate(trajectory):
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "code") and parsed.code is not None:
                        # Look for the next user message (environment response)
                        if (
                            i + 1 < len(trajectory)
                            and trajectory[i + 1]["role"] == "user"
                        ):
                            env_response = trajectory[i + 1]["content"]
                            parsed_response = self.env_parser.parse(
                                env_response, strict=False
                            )

                            # Check if we have performance metrics
                            if (
                                hasattr(parsed_response, "code_result_attributes")
                                and parsed_response.code_result_attributes
                            ):
                                attr_dict = parsed_response.code_result_attributes

                                # Extract execution time and memory usage
                                try:
                                    exec_time = float(
                                        attr_dict.get("execution_time", 0)
                                    )
                                    # Convert memory from string (potentially with B suffix) to float
                                    memory_str = attr_dict.get("memory_used", "0")
                                    if isinstance(
                                        memory_str, str
                                    ) and memory_str.endswith("B"):
                                        memory_str = memory_str[:-1]
                                    memory_used = float(memory_str)

                                    # Skip if both are zero (likely an error)
                                    if exec_time <= 0 and memory_used <= 0:
                                        continue

                                    # Check if execution was successful by looking at code_result
                                    if (
                                        hasattr(parsed_response, "code_result")
                                        and parsed_response.code_result
                                        and not parsed_response.code_result.startswith(
                                            "Error:"
                                        )
                                    ):
                                        # Save metrics and code length for successful executions
                                        code_length = len(parsed.code)
                                        performance_metrics.append(
                                            {
                                                "execution_time": exec_time,
                                                "memory_used": memory_used,
                                                "code_length": code_length,
                                                "message_index": i,
                                            }
                                        )
                                except (ValueError, TypeError):
                                    # Skip metrics we can't parse
                                    continue

            # If no successful executions, return 0
            if not performance_metrics:
                return 0.0

            # Calculate performance score using adaptive metrics
            # Calculate performance score using adaptive metrics
            return calculate_performance_score(performance_metrics, reference_metrics)

        def calculate_performance_score(metrics_list, reference_metrics=None):
            """
            Calculate a performance score that adapts to available data.

            Handles single execution case by using either reference metrics or
            reasonable default heuristics.
            """
            if not metrics_list:
                return 0.0

            # For a single execution or multiple executions, calculate a score
            if len(metrics_list) == 1:
                return calculate_single_execution_score(
                    metrics_list[0], reference_metrics
                )
            else:
                return calculate_multiple_execution_score(metrics_list)

        def calculate_single_execution_score(metric, reference_metrics=None):
            """
            Calculate performance score for a single execution.

            Uses reference metrics if available, otherwise applies reasonable heuristics.
            """
            # Default weights
            time_weight = 0.4
            memory_weight = 0.4
            code_length_weight = 0.2

            # Extract metrics
            exec_time = metric["execution_time"]
            memory_used = metric["memory_used"]
            code_length = metric["code_length"]

            # 1. Calculate time efficiency score
            if (
                reference_metrics
                and "executuion_time" in reference_metrics
                and reference_metrics["execution_time"] > 0
            ):
                # Compare with reference time (if available)
                ref_time = reference_metrics["execution_time"]
                time_ratio = ref_time / exec_time if exec_time > 0 else 0
                # Sigmoid-like function to map ratio to [0,1] with reasonable thresholds
                time_score = min(1.0, 2 / (1 + math.exp(-2 * time_ratio)))
            else:
                # Without reference, use absolute thresholds
                # Faster code gets higher scores
                if exec_time <= 0.01:  # Very fast execution
                    time_score = 1.0
                elif exec_time <= 0.1:  # Fast execution
                    time_score = 0.9
                elif exec_time <= 0.5:  # Moderate execution
                    time_score = 0.7
                elif exec_time <= 1.0:  # Standard execution
                    time_score = 0.5
                elif exec_time <= 5.0:  # Slower execution
                    time_score = 0.3
                else:  # Very slow execution
                    time_score = 0.1

            # 2. Calculate memory efficiency score
            if (
                reference_metrics
                and "memory_used" in reference_metrics
                and reference_metrics["memory_used"] > 0
            ):
                # Compare with reference memory (if available)
                ref_memory = reference_metrics["memory_used"]
                memory_ratio = ref_memory / memory_used if memory_used > 0 else 0
                # Sigmoid-like function to map ratio to [0,1]
                memory_score = min(1.0, 2 / (1 + math.exp(-2 * memory_ratio)))
            else:
                # Without reference, use absolute thresholds based on typical Python memory usage
                # (these thresholds assume bytes as the unit)
                if memory_used <= 1000:  # Extremely memory efficient
                    memory_score = 1.0
                elif memory_used <= 10000:  # Very memory efficient
                    memory_score = 0.9
                elif memory_used <= 100000:  # Memory efficient
                    memory_score = 0.8
                elif memory_used <= 1000000:  # Standard memory usage
                    memory_score = 0.6
                elif memory_used <= 10000000:  # Higher memory usage
                    memory_score = 0.4
                elif memory_used <= 100000000:  # High memory usage
                    memory_score = 0.2
                else:  # Very high memory usage
                    memory_score = 0.1

            # 3. Calculate code length score
            # Very short code (less than 10 chars) is likely not meaningful
            if code_length < 10:
                length_score = 0.0
            # Beyond a minimum threshold, shorter code is generally better
            elif code_length <= 50:
                length_score = 1.0
            elif code_length <= 100:
                length_score = 0.9
            elif code_length <= 200:
                length_score = 0.8
            elif code_length <= 500:
                length_score = 0.6
            elif code_length <= 1000:
                length_score = 0.4
            else:
                length_score = 0.2

            # Combine the scores with their respective weights
            final_score = (
                time_weight * time_score
                + memory_weight * memory_score
                + code_length_weight * length_score
            )

            return final_score

        def calculate_multiple_execution_score(metrics_list):
            """
            Calculate performance score when multiple executions are available.
            This allows for normalization and trend analysis.
            """
            # This function contains the same logic as our original implementation
            # for multiple executions

            # Sort metrics by message index to maintain chronological order
            metrics_list.sort(key=lambda x: x["message_index"])

            # Get performance values
            time_values = [m["execution_time"] for m in metrics_list]
            memory_values = [m["memory_used"] for m in metrics_list]
            code_lengths = [m["code_length"] for m in metrics_list]

            # Find min/max values
            min_time = min(time_values)
            max_time = max(time_values)
            min_memory = min(memory_values)
            max_memory = max(memory_values)
            min_length = min(code_lengths)

            # Calculate complexity indicators
            time_mean = sum(time_values) / len(time_values)
            memory_mean = sum(memory_values) / len(memory_values)

            # Estimate problem complexity using coefficient of variation
            time_cv = (
                (sum((t - time_mean) ** 2 for t in time_values) ** 0.5) / time_mean
                if time_mean > 0
                else 0
            )
            memory_cv = (
                (sum((m - memory_mean) ** 2 for m in memory_values) ** 0.5)
                / memory_mean
                if memory_mean > 0
                else 0
            )

            # Adjust weights based on complexity
            complexity_indicator = (time_cv + memory_cv) / 2
            performance_weight = min(0.8, 0.5 + complexity_indicator)
            code_length_weight = 1 - performance_weight

            # Calculate normalized scores
            if max_time > min_time:
                time_scores = [
                    (max_time - m["execution_time"]) / (max_time - min_time)
                    for m in metrics_list
                ]
            else:
                time_scores = [1.0 for _ in metrics_list]

            if max_memory > min_memory:
                memory_scores = [
                    (max_memory - m["memory_used"]) / (max_memory - min_memory)
                    for m in metrics_list
                ]
            else:
                memory_scores = [1.0 for _ in metrics_list]

            # Calculate code length scores
            length_scores = []
            for length in code_lengths:
                if length < 10:  # Very short code is likely not a complete solution
                    length_score = 0.0
                else:
                    relative_length = length / min(max(min_length, 20), length)
                    length_score = 1.0 / relative_length
                length_scores.append(min(1.0, length_score))

            # Combine scores
            performance_scores = []
            for i in range(len(metrics_list)):
                perf_score = 0.5 * time_scores[i] + 0.5 * memory_scores[i]
                combined_score = (
                    performance_weight * perf_score
                    + code_length_weight * length_scores[i]
                )
                performance_scores.append(combined_score)

            # Check for improvement trend
            trend_bonus = 0.0
            if len(performance_scores) >= 2:
                if performance_scores[-1] >= max(performance_scores[:-1]):
                    trend_bonus = 0.1
                if len(performance_scores) >= 3 and all(
                    performance_scores[i] <= performance_scores[i + 1]
                    for i in range(len(performance_scores) - 1)
                ):
                    trend_bonus = 0.2

            # Return best score plus trend bonus
            return min(1.0, max(performance_scores) + trend_bonus)

        return [check_performance(c) for c in completions]
