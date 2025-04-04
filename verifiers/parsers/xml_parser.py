import re
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from types import SimpleNamespace


class XMLParser:
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]]):
        """
        Initialize the parser with field definitions.

        Each field may be:
          - a string (e.g. "reasoning"): the XML tag is fixed.
          - a tuple of alternatives (e.g. ("code", "answer")): the first element is
            the canonical name used for formatting, and all elements are allowed tags
            when parsing.

        The schema is assumed to have no duplicate names.
        """
        self._fields: List[
            Tuple[str, List[str]]
        ] = []  # List of (canonical, [alternatives])
        seen = set()
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                if not all(isinstance(alt, str) for alt in field):
                    raise TypeError("All alternatives in a tuple must be strings.")
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))

    def get_xml_reward_func(self) -> Callable:
        """
        Return a reward function that checks for proper XML tag usage.

        The returned function evaluates if messages in trajectories properly use
        the expected XML tags defined in this parser's fields configuration.
        """

        def xml_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks for proper XML tag usage in completions."""

            def count_xml(trajectory) -> float:
                # Get all messages from the model
                model_messages = [
                    msg for msg in trajectory if msg["role"] == "assistant"
                ]
                if not model_messages:
                    return 0.0

                # Calculate XML tag usage scores for each message
                xml_scores = []
                for msg in model_messages:
                    content = msg["content"]
                    score = 0
                    total_checks = 0

                    # For each canonical field with its alternatives
                    for canonical, alternatives in self._fields:
                        # Track if at least one alternative was used for this field

                        # Check all alternatives for this field
                        for alt in alternatives:
                            # If this alternative is used, check it has proper tags
                            # Updated to handle tags with attributes
                            opening_tags = re.findall(rf"<{alt}(\s+[^>]*)?>\s*", content)
                            closing_tags = re.findall(rf"</{alt}>\s*", content)

                            if opening_tags or closing_tags:
                                score += 1 - abs(len(opening_tags) - 1)
                                score += 1 - abs(len(closing_tags) - 1)
                                total_checks += 2

                        # If no alternatives for this field were used, we don't add to total_checks
                        # because we're not requiring any specific field to be present

                    # Calculate normalized score for this message
                    if total_checks > 0:
                        xml_scores.append(score / total_checks)
                    else:
                        # If no tags used at all, give a zero score
                        xml_scores.append(0.0)

                # Return average XML score across all messages
                return sum(xml_scores) / len(xml_scores) if xml_scores else 0.0

            # Apply the XML check to each completion trajectory
            return [count_xml(c) for c in completions]

        return xml_reward_func

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks if messages follow the expected format.

        The function does not make assumptions about which fields should start/end the message
        or the specific order of fields. It checks that:
        - At least one field from the schema is present in each message
        - Fields have proper content and spacing
        """

        def format_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks if each step follows the expected format."""

            def check_format(trajectory):
                # Get assistant messages
                model_messages = [
                    msg for msg in trajectory if msg["role"] == "assistant"
                ]
                if not model_messages:
                    return 0.0

                # Calculate format adherence for each message
                format_scores = []
                for msg in model_messages:
                    content = msg["content"]
                    parsed = self.parse(content)
                    parsed_no_strip = self.parse(content, strip=False)

                    # Check if the message has at least one valid field
                    has_any_field = False
                    fields_with_content = 0
                    total_fields = 0

                    # Keep track of which expected fields are present
                    expected_field_count = len(
                        self._fields
                    )  # Total number of expected field sets
                    present_field_sets = (
                        set()
                    )  # Which field sets have at least one alternative present

                    # Check proper spacing for fields
                    has_correct_spacing = True

                    for i, (canonical, alternatives) in enumerate(self._fields):
                        field_set_present = False
                        for alt in alternatives:
                            if (
                                hasattr(parsed, alt)
                                and getattr(parsed, alt) is not None
                            ):
                                has_any_field = True
                                fields_with_content += 1
                                total_fields += 1
                                field_set_present = True

                                # Check if field exists in non-stripped version too (proper spacing)
                                if not (
                                    hasattr(parsed_no_strip, alt)
                                    and getattr(parsed_no_strip, alt) is not None
                                ):
                                    has_correct_spacing = False
                            elif (
                                re.search(rf"<{alt}(\s+[^>]*)?>\s*", content)
                                or re.search(rf"</{alt}>\s*", content)
                            ):
                                # Tag exists but content wasn't properly parsed
                                total_fields += 1
                                field_set_present = True

                        # If any alternative from this field set was present, count it
                        if field_set_present:
                            present_field_sets.add(i)

                    # Calculate format score components
                    format_score = 0.0

                    # Check if any field from the first field set starts the message
                    starts_with_any_field = False
                    first_field_set = self._fields[0][
                        1
                    ]  # Get alternatives for first field set
                    for alt in first_field_set:
                        if re.match(rf"^\s*<{alt}(\s+[^>]*)?>\s*", content.strip()):
                            starts_with_any_field = True
                            break

                    # Check if any field from the last field set ends the message
                    ends_with_any_field = False
                    last_field_set = self._fields[-1][
                        1
                    ]  # Get alternatives for last field set
                    for alt in last_field_set:
                        if content.strip().endswith(f"</{alt}>"):
                            ends_with_any_field = True
                            break

                    # Weight the score based on different criteria
                    if has_any_field:
                        # Calculate the proportion of expected field sets that are present
                        field_set_ratio = len(present_field_sets) / expected_field_count
                        format_score += 0.4 * field_set_ratio

                    if has_correct_spacing:
                        format_score += 0.2

                    if starts_with_any_field:
                        format_score += 0.2

                    if ends_with_any_field:
                        format_score += 0.2

                    format_scores.append(format_score)

                # Return average format adherence
                return sum(format_scores) / len(format_scores) if format_scores else 0.0

            # Apply the format check to each completion trajectory
            return [check_format(c) for c in completions]

        return format_reward_func

    def get_fields(self) -> List[str]:
        """Return a list of the canonical field names (in order)."""
        return [canonical for canonical, _ in self._fields]

    def format(self, strict=True, **kwargs) -> str:
        """
        Format the provided keyword arguments into an XML string.

        For fields with alternatives (tuple), the canonical name (the first element)
        is used as the XML tag. The method looks for a provided value using any of the
        allowed names (preferring the canonical if present).

        Args:
            strict (bool): If True, require all field sets defined in the schema.
                        If False, allow formatting with a subset of fields.
            **kwargs: Values for each field, which can be strings or dicts with
                    "content" and "attributes" keys.

        Example usage:
            parser = XMLParser(['reasoning', ('code', 'answer')])

            # With strict=True (default), all fields are required
            formatted_str = parser.format(reasoning="...", code="...")

            # With strict=False, only specified fields are included
            formatted_str = parser.format(strict=False, code="...")  # Only includes code

            # To include attributes:
            formatted_str = parser.format(
                code={"content": "...", "attributes": {"exec_time": "1.5"}}
            )

        Raises:
            ValueError: If strict=True and any required field is missing
        """
        parts = []
        for canonical, alternatives in self._fields:
            value = None
            attributes = {}

            # Look for a provided value using any of the acceptable keys,
            # preferring the canonical name if it exists.
            if canonical in kwargs:
                value_data = kwargs[canonical]
                # Check if the value is a dict with content and attributes
                if isinstance(value_data, dict) and "content" in value_data:
                    value = value_data["content"]
                    if "attributes" in value_data and isinstance(value_data["attributes"], dict):
                        attributes = value_data["attributes"]
                else:
                    value = value_data
            else:
                for alt in alternatives:
                    if alt in kwargs:
                        value_data = kwargs[alt]
                        # Check if the value is a dict with content and attributes
                        if isinstance(value_data, dict) and "content" in value_data:
                            value = value_data["content"]
                            if "attributes" in value_data and isinstance(value_data["attributes"], dict):
                                attributes = value_data["attributes"]
                        else:
                            value = value_data
                        break

            # Skip this field if value is None and we're not in strict mode
            if value is None:
                if strict:
                    raise ValueError(
                        f"Missing value for field '{canonical}' (allowed: {alternatives})"
                    )
                else:
                    continue  # Skip this field in non-strict mode

            # Format attributes if present
            attr_str = ""
            if attributes:
                attr_str = " " + " ".join(f'{k}="{v}"' for k, v in attributes.items())

            # Use the canonical name as the tag for formatting.
            parts.append(f"<{canonical}{attr_str}>\n{value}\n</{canonical}>")

        return "\n".join(parts)

    def parse(self, text: str, strip: bool = True, strict: bool = True) -> Any:
        """
        Parse the given XML string and return an object with attributes.

        Args:
            text (str): The text to parse
            strip (bool): Whether to strip whitespace from tag content
            strict (bool): If True, raise an error for missing required tags.
                        If False, return None for missing tags.

        Returns:
            SimpleNamespace: An object with attributes for each tag
        """
        results: Dict[str, Optional[Union[str, Dict[str, str]]]] = {}

        missing_fields = []

        for canonical, alternatives in self._fields:
            # For each allowed alternative tag, search independently.
            field_set_present = False
            for alt in alternatives:
                # Find the tag opening pattern which may include attributes
                tag_pattern = rf"<{alt}(\s+[^>]+?)?\s*>\s*(.*?)\s*</{alt}>"
                match = re.search(tag_pattern, text, re.DOTALL)

                if match:
                    field_set_present = True
                    # Extract the content
                    content = match[2]
                    results[alt] = content.strip() if strip else content

                    # Extract and parse attributes if any
                    attr_str = match[1] if match[1] else ""
                    if attr_str:
                        attr_dict = {}

                        # First, look for quoted attributes: attr="value"
                        quoted_attrs = re.finditer(r'([^=>\s]+)\s*=\s*"([^"]*)"', attr_str)
                        for attr_match in quoted_attrs:
                            attr_name = attr_match[1]
                            attr_value = attr_match[2]
                            attr_dict[attr_name] = attr_value

                        # Then, look for unquoted attributes: attr=value
                        unquoted_attrs = re.finditer(r'([^=>\s]+)\s*=\s*([^"\s][^\s>]*)', attr_str)
                        for attr_match in unquoted_attrs:
                            attr_name = attr_match[1]
                            attr_value = attr_match[2]
                            # Only add if not already added by quoted pattern
                            if attr_name not in attr_dict:
                                attr_dict[attr_name] = attr_value

                        # Add attributes dictionary
                        results[f"{alt}_attributes"] = attr_dict
                    else:
                        results[f"{alt}_attributes"] = {}
                else:
                    results[alt] = None
                    results[f"{alt}_attributes"] = {}

            # Track missing field sets for strict mode
            if not field_set_present and strict:
                missing_fields.append(canonical)

        # In strict mode, raise an error if required fields are missing
        if strict and missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        return SimpleNamespace(**results)
