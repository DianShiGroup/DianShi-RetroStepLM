import ast
import json


def safe_label(text):
    """Escape special characters for Mermaid labels."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace('"', "#quot;")
        .replace("(", "#40;")
        .replace(")", "#41;")
        .replace("[", "#91;")
        .replace("]", "#93;")
    )


def parse_json_or_raw(value):
    """Parse JSON string or return value as-is.

    Args:
        value: String to parse as JSON, or any other value

    Returns:
        Parsed JSON dict/list, or original value if not a JSON string
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Try parsing as Python literal (handles single-quoted dicts)
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value
