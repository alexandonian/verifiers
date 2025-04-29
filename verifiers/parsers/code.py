import regex


def extract_code_blocks(text: str) -> list[str]:
    """Extract all code blocks from the given text.

    Args:
        text (str): The text to extract the code blocks from.

    Returns:
        list[str]: A list of all code blocks found in the text.
    """
    pattern = r"```(?:\w+\n)?([\s\S]*?)```"
    matches = regex.findall(pattern, text)

    return [match.strip() for match in matches]


def extract_first_code_block(text: str) -> str:
    """Extract the first code block from the given text.

    Args:
        text (str): The text to extract the code block from.

    Returns:
        str: The first code block found in the text, or an empty string if no code blocks are found.
    """
    code_blocks = extract_code_blocks(text)
    return code_blocks[0] if code_blocks else ""


def extract_last_code_block(text: str) -> str:
    """Extract the last code block from the given text.

    Args:
        text (str): The text to extract the code block from.

    Returns:
        str: The last code block found in the text, or an empty string if no code blocks are found.
    """
    code_blocks = extract_code_blocks(text)
    return code_blocks[-1] if code_blocks else ""
