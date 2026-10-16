import functools
import inspect
from typing import Any

from docstring_parser import parse as parse_docstring
from pydantic import ValidationError, create_model


def drop_titles(obj):
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            current.pop("title", None)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def build_args_schema(tool, name):
    """Build a Pydantic model from function signature for validation."""
    base = tool.func if isinstance(tool, functools.partial) else tool
    signature = inspect.signature(base)
    params = list(signature.parameters.values())

    if isinstance(tool, functools.partial):
        bound_positional = len(tool.args or ())
        bound_keywords = set((tool.keywords or {}).keys())
        filtered = []
        for param in params:
            if param.name == "self":
                continue
            if bound_positional > 0 and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                bound_positional -= 1
                continue
            if param.name in bound_keywords:
                continue
            filtered.append(param)
        params = filtered

    fields = {}
    for param in params:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        anno = param.annotation if param.annotation is not inspect._empty else Any
        default = param.default if param.default is not inspect._empty else ...
        fields[param.name] = (anno, default)
    return create_model(f"{name}Args", **fields)


class Tool:
    """Tool wrapper for logging and schema generation."""

    def __init__(self, func, *, name=None, desc=None):
        self.func = func
        base = func.func if isinstance(func, functools.partial) else func
        self.base_func = base
        self.name = name or getattr(base, "__name__")
        self.desc = desc or inspect.getdoc(base) or ""
        self.args_schema = build_args_schema(func, self.name)

    def __call__(self, **kwargs):
        return self.func(**kwargs)

    def invoke(self, args):
        try:
            validated = self.args_schema(**args)
            result = self(**validated.model_dump())
            return result
        except ValidationError as exc:
            errors = "; ".join(
                f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()
            )
            return {"error": f"Invalid arguments: {errors}"}
        except Exception as exc:
            return {"error": str(exc)}

    def to_schema(self):
        """Return an OpenAI-compatible tool schema."""
        docstring = self.desc or inspect.getdoc(self.base_func) or ""
        args_schema = getattr(self, "args_schema", None) or build_args_schema(
            self.func, self.name
        )
        schema = args_schema.model_json_schema()
        properties = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        parsed_doc = parse_docstring(docstring)
        for param in parsed_doc.params:
            if param.description and param.arg_name in properties:
                properties[param.arg_name]["description"] = param.description
        description = parsed_doc.short_description or ""

        drop_titles(properties)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


def from_callable(func, *, name=None, desc=None):
    name = name or getattr(func, "__name__")
    return Tool(func, name=name, desc=desc)
