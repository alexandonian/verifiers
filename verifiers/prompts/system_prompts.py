THINK = "think"
ANSWER = "answer"
CODE = "code"
TOOL = "tool"
TOOL_RESULT = "tool_result"
CODE_RESULT = "code_result"
BEGIN_THINK = f"<{THINK}>"
END_THINK = f"</{THINK}>"
BEGIN_ANSWER = f"<{ANSWER}>"
END_ANSWER = f"</{ANSWER}>"
BEGIN_CODE = f"<{CODE}>"
END_CODE = f"</{CODE}>"
BEGIN_TOOL = f"<{TOOL}>"
END_TOOL = f"</{TOOL}>"
BEGIN_TOOL_RESULT = f"<{TOOL_RESULT}>"
END_TOOL_RESULT = f"</{TOOL_RESULT}>"
BEGIN_CODE_RESULT = f"<{CODE_RESULT}>"
END_CODE_RESULT = f"</{CODE_RESULT}>"
TOOL_EXAMPLE = '{{"name": "calculator", "args": {{"expression": "715.019337 / 4.0"}}}}'
ANSWER_EXAMPLE = "\\frac{{1}}{{2}}"

SYSTEM_PROMPT = f"""
Respond in the following format:

{BEGIN_THINK}
...
{END_THINK}
{BEGIN_ANSWER}
...
{END_ANSWER}
"""

SIMPLE_PROMPT = f"""
Respond in the following format, using careful step-by-step reasoning.

{BEGIN_THINK}
...
{END_THINK}
{BEGIN_ANSWER}
...
{END_ANSWER}
"""

CODE_PROMPT = f"""\
Given a math problem, use step-by-step reasoning and code execution to solve the problem.

For each step:
1. Think through your reasoning inside {BEGIN_THINK} tags
2. Write Python scripts inside {BEGIN_CODE} tags to work out calculations
   - Functions and variables do not persist across {BEGIN_CODE} calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in {BEGIN_CODE_RESULT} tags
4. Continue until you can give the final answer inside {BEGIN_ANSWER} tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = f"""\
You have access to the following tools to help solve problems:

{{tool_descriptions}}

For each step:
1. Think through your reasoning inside {BEGIN_THINK} tags
2. If needed, use a tool by writing a JSON command inside {BEGIN_TOOL} tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside {BEGIN_TOOL_RESULT} tags
4. Continue until you can give the final answer inside {BEGIN_ANSWER} tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

DEFAULT_CODE_TOOL_PROMPT_TEMPLATE = str(f"""\
Use step-by-step reasoning, code execution and the following tools to help solve problems:

{{tool_descriptions}}

For each step:
1. Think through your reasoning inside {BEGIN_THINK} tags
2a. If needed, use a tool by writing a JSON command inside {BEGIN_TOOL} tags with:
   - "name": the tool to use
   - "args": the arguments for the tool

Example usage:
{BEGIN_TOOL}
{TOOL_EXAMPLE}
{END_TOOL}

2b. Write Python scripts inside {BEGIN_CODE} tags to work out calculations
   - Functions and variables do not persist across {BEGIN_CODE} calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
   - Within {BEGIN_CODE} tags, use triple backticks to denote code blocks

Example usage:
{BEGIN_CODE}
```python
import numpy as np
result = np.sqrt(16)
print(result)  # This will output 4.0
```
{END_CODE}

3a. You will see the tool's output inside {BEGIN_TOOL_RESULT} tags
3b. You will see the output from print() statements in your code in {BEGIN_CODE_RESULT} tags
4. Continue until you can give the final answer inside {BEGIN_ANSWER} tags

The {BEGIN_ANSWER}...{END_ANSWER} tags should contain only your final answer.

Example for multiple choice questions:
{BEGIN_ANSWER}
A
{END_ANSWER}

Example for math problems:
{BEGIN_ANSWER}
{ANSWER_EXAMPLE}
{END_ANSWER}

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
""")
