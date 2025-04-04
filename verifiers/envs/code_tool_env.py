import contextlib
import inspect
import json
from typing import List, Dict, Any, Callable

from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts.system_prompts import DEFAULT_CODE_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import CodeToolRubric
from verifiers.tools.python import secure_execute_python


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()

    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip()

    return_type = str(
        sig.return_annotation.__name__
        if sig.return_annotation != inspect.Parameter.empty
        else "any"
    )

    print(f"return_description: {return_description} ({return_type})")
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name) + 1 :].strip()

        args[name] = {
            "type": str(
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "any"
            ),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default

    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": f"{return_description} ({return_type})",
        "examples": examples,
    }


def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}", "\nArguments:"]

        for arg_name, arg_info in schema["args"].items():
            default = (
                f" (default: {arg_info['default']})" if "default" in arg_info else ""
            )
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")

        if schema["examples"]:
            desc.append("\nExamples:")
            desc.extend(f"  {example}" for example in schema["examples"])
        if schema["returns"]:
            desc.append(f"\nReturns: {schema['returns']}")

        descriptions.append("\n".join(desc))

    return "\n\n".join(descriptions)


class CodeToolEnv(MultiTurnEnv):
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        tools: List[Callable] | None = None,
        system_prompt: str = DEFAULT_CODE_TOOL_PROMPT_TEMPLATE,
        few_shot: List[Dict[str, str]] | None = None,
        sampling_args=None,
        mask_env_response: bool = True,
        max_steps: int = 10,
        **kwargs,
    ):
        if tools is None:
            tools = []
        if few_shot is None:
            few_shot = []
        if sampling_args is None:
            sampling_args = {
                "stop": [
                    "</tool>\n",
                    "</answer>\n",
                    "</code>\n",
                ],
                "include_stop_str_in_output": True,
            }
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}

        # Format the system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs,
        )
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.rubric = CodeToolRubric(tools=tools)
        self.llm_parser = XMLParser(fields=["reasoning", ("code", "tool", "answer")])
        self.env_parser = XMLParser(fields=["tool_result", "code_result"])

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""

        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)

        return sum(
            message.get("role") == "assistant"
            for message in messages[conversation_start:]
        )

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_steps:
                return True

            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, "answer") and parsed.answer is not None
        except Exception:
            return False

    def call_tool(self, tool_json: str, **kwargs: Any) -> str:
        """Call a tool based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return 'Error: Tool command must be a JSON object, e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            tool_name = command.get("name")
            if not tool_name:
                return 'Error: Tool command must specify \'name\', e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            if tool_name not in self.tools:
                return (
                    f"Error: Unknown tool '{tool_name}. "
                    + 'Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
                )

            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                tool_schema = next(
                    (
                        schema["args"]
                        for schema in self.tool_schemas
                        if schema["name"] == tool_name
                    ),
                    None,
                )
                return f"Error: Arguments for {tool_name} must be a JSON object with schema {tool_schema}, not a string."

            # Call the tool function with arguments
            result = tool_func(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return 'Error: Invalid JSON format. Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
        except Exception as e:
            return (
                f"Error: {str(e)}. "
                + 'Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
            )

    def env_response(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, str]:
        parsed = self.llm_parser.parse(messages[-1]["content"])
        # Check if we got a valid tool field (not just None from failed parsing)
        outputs = []
        with contextlib.suppress(Exception):
            if hasattr(parsed, "tool") and parsed.tool is not None:
                tool_result = self.call_tool(parsed.tool)
                if len(tool_result.strip()) > 0:
                    tool_result = tool_result.strip()
                else:
                    tool_result = "Error: Tool execution returned empty output"
                outputs.append(
                    self.env_parser.format(strict=False, tool_result=tool_result)
                )

        with contextlib.suppress(Exception):
            if hasattr(parsed, "code") and parsed.code is not None:
                code_result = secure_execute_python(parsed.code.strip())
                code_output = code_result["output"]
                if len(code_output.strip()) == 0:
                    code_output = "Error: Code execution returned empty output."

                outputs.append(
                    self.env_parser.format(
                        strict=False,
                        code_result={
                            "content": code_output,
                            "attributes": {
                                "execution_time": code_result["execution_time"],
                                "memory_used": code_result["memory_used"],
                            },
                        },
                    )
                )

        if outputs:
            return {
                "role": "user",
                "content": "\n".join(outputs),
            }

        return {
            "role": "user",
            "content": "Error: Code or Tool command not found or invalid XML format. Please ensure correct formatting.",
        }
