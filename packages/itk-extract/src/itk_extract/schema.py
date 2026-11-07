# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""The JSON schema that extracted objects must follow."""

from __future__ import annotations

import copy
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class RecordSchema:
    """A validated JSON schema describing one record object.

    Array fields at any depth are the "record arrays" that become output
    frames; ``array_paths`` lists them as tuples of field names.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        if schema.get("type") != "object":
            msg = "Record schema must describe an object at the root"
            raise ValueError(msg)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as e:
            msg = f"Invalid JSON schema: {e.message}"
            raise ValueError(msg) from e
        self._raw = copy.deepcopy(schema)
        self._validator = Draft202012Validator(self._raw)

    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)

    @property
    def title(self) -> str:
        return str(self._raw.get("title", "record_object"))

    @property
    def array_paths(self) -> list[tuple[str, ...]]:
        """Every array field, as the path of field names leading to it."""
        found: list[tuple[str, ...]] = []

        def walk(node: Any, path: tuple[str, ...]) -> None:
            if not isinstance(node, dict):
                return
            for name, spec in node.get("properties", {}).items():
                if not isinstance(spec, dict):
                    continue
                here = (*path, name)
                if spec.get("type") == "array":
                    found.append(here)
                    walk(spec.get("items", {}), here)
                else:
                    walk(spec, here)

        walk(self._raw, ())
        return found

    @property
    def array_labels(self) -> list[str]:
        return [".".join(path) for path in self.array_paths]

    def validate(self, data: Any) -> list[str]:
        """Human-readable violations, empty when ``data`` conforms."""
        return [
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in sorted(self._validator.iter_errors(data), key=str)
        ]

    def for_openai(self) -> dict[str, Any]:
        """Copy suitable for strict structured outputs.

        Every object gets ``additionalProperties: false`` and all of its
        properties marked required, which the API insists on.
        """
        schema = self.raw

        def normalise(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    node["additionalProperties"] = False
                    required = list(node.get("required", []))
                    for name in node.get("properties", {}):
                        if name not in required:
                            required.append(name)
                    node["required"] = required
                for value in node.values():
                    normalise(value)
            elif isinstance(node, list):
                for item in node:
                    normalise(item)

        normalise(schema)
        return schema

    def response_format(self) -> dict[str, Any]:
        """The ``response_format`` argument for an OpenAI chat call."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "record_object",
                "strict": True,
                "schema": self.for_openai(),
            },
        }
