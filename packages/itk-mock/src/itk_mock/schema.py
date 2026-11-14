# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
"""Build and edit the JSON schema that describes the records to generate.

Every editing function takes the whole schema, a ``nesting`` path of field
labels leading to the object (or array of objects) being edited (``[]`` is the
root), and usually a field label. Schemas are edited in place. A path or label
that does not resolve raises a ``SchemaEditError`` subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from itk_mock.models import (
    ArrayFieldType,
    DuplicateFieldError,
    FieldNotFoundError,
    FieldTypeError,
    PrimitiveFieldType,
    SchemaEditError,
    SchemaPathError,
    StringFormat,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

Schema = dict[str, Any]

_STRING_CONSTRAINTS = ("minLength", "maxLength", "pattern", "format")
_NUMBER_CONSTRAINTS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
)


def create_boilerplate_schema(
    schema_field: str = "http://json-schema.org/draft/2020-12/schema",
    title_field: str = "Example Schema",
    description_field: str = (
        "An example schema ready to be edited and populated with fields."
    ),
) -> Schema:
    """Root object with one empty ``records`` array of objects."""
    return {
        "$schema": schema_field,
        "title": title_field,
        "description": description_field,
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "description": "An array of records",
                "items": _empty_object(),
            }
        },
        "required": ["records"],
        "additionalProperties": False,
    }


def normalize_schema_for_openai(schema: Schema) -> Schema:
    """Make an imported schema usable with OpenAI structured outputs.

    Every object gets ``additionalProperties: false`` and lists all of its
    properties as required. Edits ``schema`` in place and returns it.
    """

    def normalize(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                normalize(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            required = node.setdefault("required", [])
            node["additionalProperties"] = False
            for name in node.get("properties", {}):
                if name not in required:
                    required.append(name)
        for value in list(node.values()):
            normalize(value)

    normalize(schema)
    return schema


def extract_array_fields(schema: Schema) -> list[list[str]]:
    """Label paths of every array field, outermost first.

    The first path is the primary record array.
    """
    array_fields: list[list[str]] = []

    def walk(node: Any, path: list[str]) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            return
        for name, value in properties.items():
            if not isinstance(value, dict):
                continue
            if value.get("type") == "array":
                array_fields.append([*path, name])
                walk(value.get("items", {}), [*path, name])
            else:
                walk(value, [*path, name])

    walk(schema, [])
    return array_fields


def get_container(schema: Schema, nesting: Sequence[str]) -> Schema:
    """Object schema reached by ``nesting``; arrays resolve to their items."""
    node = schema
    for depth, label in enumerate(nesting):
        properties = node.get("properties")
        if not isinstance(properties, dict) or label not in properties:
            msg = f"No field {label!r} at {_show(nesting[:depth])}"
            raise SchemaPathError(msg)
        node = properties[label]
        if node.get("type") == "array":
            node = node.get("items", {})
        if node.get("type") != "object":
            msg = f"{_show(nesting[: depth + 1])} is not an object or array of objects"
            raise SchemaPathError(msg)
    if not isinstance(node.get("properties"), dict):
        msg = f"{_show(nesting)} has no properties"
        raise SchemaPathError(msg)
    return node


def get_properties(schema: Schema, nesting: Sequence[str]) -> dict[str, Schema]:
    """Fields of the object at ``nesting``, by label."""
    return get_container(schema, nesting)["properties"]


def get_required_list(schema: Schema, nesting: Sequence[str]) -> list[str]:
    """The ``required`` list of the object at ``nesting``."""
    return get_container(schema, nesting).setdefault("required", [])


def get_field(schema: Schema, nesting: Sequence[str], label: str) -> Schema:
    """The schema of one field."""
    properties = get_properties(schema, nesting)
    if label not in properties:
        msg = f"No field {label!r} at {_show(nesting)}"
        raise FieldNotFoundError(msg)
    return properties[label]


def add_object_field(
    schema: Schema,
    nesting: Sequence[str],
    label: str | None = None,
    description: str = "",
    required: bool = True,
) -> str:
    """Add a nested object; returns the label used."""
    field = {"type": "object", "description": description, **_empty_object()}
    return _add(schema, nesting, label, "object", field, required)


def add_array_field(
    schema: Schema,
    nesting: Sequence[str],
    label: str | None = None,
    description: str = "",
    item_type: ArrayFieldType = ArrayFieldType.STRING,
    required: bool = True,
) -> str:
    """Add an array of objects or primitives; returns the label used."""
    items = (
        _empty_object()
        if item_type == ArrayFieldType.OBJECT
        else {"type": item_type.value}
    )
    field = {"type": "array", "description": description, "items": items}
    return _add(schema, nesting, label, f"{item_type.value}_array", field, required)


def add_primitive_field(
    schema: Schema,
    nesting: Sequence[str],
    label: str | None = None,
    description: str = "",
    field_type: PrimitiveFieldType = PrimitiveFieldType.STRING,
    required: bool = True,
) -> str:
    """Add a string, number or boolean field; returns the label used."""
    field = {"type": field_type.value, "description": description}
    return _add(schema, nesting, label, field_type.value, field, required)


def rename_field(
    schema: Schema, nesting: Sequence[str], old_label: str, new_label: str
) -> None:
    """Rename a field, keeping its position and its required status."""
    properties = get_properties(schema, nesting)
    get_field(schema, nesting, old_label)
    if new_label == old_label:
        return
    if new_label in properties:
        msg = f"Field {new_label!r} already exists at {_show(nesting)}"
        raise DuplicateFieldError(msg)
    renamed = {
        (new_label if key == old_label else key): value
        for key, value in properties.items()
    }
    properties.clear()
    properties.update(renamed)
    required = get_required_list(schema, nesting)
    if old_label in required:
        required[required.index(old_label)] = new_label
    _sort_required(schema, nesting)


def delete_field(schema: Schema, nesting: Sequence[str], label: str) -> None:
    """Remove a field and its required entry."""
    get_field(schema, nesting, label)
    get_properties(schema, nesting).pop(label)
    required = get_required_list(schema, nesting)
    if label in required:
        required.remove(label)


def move_field_up(schema: Schema, nesting: Sequence[str], label: str) -> None:
    """Swap a field with the one before it; the first field stays put."""
    _move(schema, nesting, label, -1)


def move_field_down(schema: Schema, nesting: Sequence[str], label: str) -> None:
    """Swap a field with the one after it; the last field stays put."""
    _move(schema, nesting, label, 1)


def set_required_field_status(
    schema: Schema, nesting: Sequence[str], label: str, required: bool
) -> None:
    """Mark a field required or optional; required order follows field order."""
    get_field(schema, nesting, label)
    reqs = get_required_list(schema, nesting)
    if required and label not in reqs:
        reqs.append(label)
    elif not required and label in reqs:
        reqs.remove(label)
    _sort_required(schema, nesting)


def set_enum_field_status(
    schema: Schema, nesting: Sequence[str], label: str, constrained: bool
) -> bool:
    """Start or stop constraining a field to an enum; returns whether it changed.

    A newly constrained field gets a placeholder enum (``[""]``, ``[0]`` or
    ``[True, False]``). Objects and arrays of objects or booleans cannot be
    constrained and return ``False``.
    """
    target = _value_schema(get_field(schema, nesting, label))
    if not constrained:
        return target.pop("enum", None) is not None
    if "enum" in target:
        return False
    defaults: dict[str, list[Any]] = {"string": [""], "number": [0]}
    if target is get_field(schema, nesting, label):
        defaults["boolean"] = [True, False]
    placeholder = defaults.get(target.get("type", ""))
    if placeholder is None:
        return False
    target["enum"] = placeholder
    return True


def set_enum_values(
    schema: Schema, nesting: Sequence[str], label: str, values: Sequence[Any] | None
) -> None:
    """Set the allowed values of a primitive field or array of primitives.

    ``None`` removes the constraint.
    """
    target = _value_schema(get_field(schema, nesting, label))
    if target.get("type") not in {"string", "number", "boolean"}:
        msg = f"Field {label!r} at {_show(nesting)} cannot be constrained to an enum"
        raise FieldTypeError(msg)
    if values is None:
        target.pop("enum", None)
    else:
        target["enum"] = list(values)


def set_additional_field_status(
    schema: Schema, nesting: Sequence[str], label: str, additional: bool
) -> bool:
    """Allow or forbid extra properties on an object or array of objects.

    Returns whether the schema changed; other field types return ``False``.
    """
    target = _value_schema(get_field(schema, nesting, label))
    if target.get("type") != "object":
        return False
    changed = target.get("additionalProperties") != additional
    target["additionalProperties"] = additional
    return changed


def set_string_min_length(
    schema: Schema, nesting: Sequence[str], label: str, min_length: int | None
) -> None:
    _set(_typed(schema, nesting, label, "string"), "minLength", min_length)


def set_string_max_length(
    schema: Schema, nesting: Sequence[str], label: str, max_length: int | None
) -> None:
    _set(_typed(schema, nesting, label, "string"), "maxLength", max_length)


def set_string_pattern(
    schema: Schema, nesting: Sequence[str], label: str, pattern: str | None
) -> None:
    _set(_typed(schema, nesting, label, "string"), "pattern", pattern)


def set_string_format(
    schema: Schema,
    nesting: Sequence[str],
    label: str,
    string_format: StringFormat | None,
) -> None:
    value = None if string_format is None else string_format.value
    _set(_typed(schema, nesting, label, "string"), "format", value)


def clear_string_constraints(
    schema: Schema, nesting: Sequence[str], label: str
) -> None:
    target = _typed(schema, nesting, label, "string")
    for key in _STRING_CONSTRAINTS:
        target.pop(key, None)


def set_number_minimum(
    schema: Schema,
    nesting: Sequence[str],
    label: str,
    minimum: float | None,
    exclusive: bool = False,
) -> None:
    """Set or clear the bound; setting one kind removes the other."""
    target = _typed(schema, nesting, label, "number")
    _set_bound(target, "minimum", "exclusiveMinimum", minimum, exclusive)


def set_number_maximum(
    schema: Schema,
    nesting: Sequence[str],
    label: str,
    maximum: float | None,
    exclusive: bool = False,
) -> None:
    """Set or clear the bound; setting one kind removes the other."""
    target = _typed(schema, nesting, label, "number")
    _set_bound(target, "maximum", "exclusiveMaximum", maximum, exclusive)


def set_number_multiple_of(
    schema: Schema, nesting: Sequence[str], label: str, multiple_of: float | None
) -> None:
    _set(_typed(schema, nesting, label, "number"), "multipleOf", multiple_of)


def clear_number_constraints(
    schema: Schema, nesting: Sequence[str], label: str
) -> None:
    target = _typed(schema, nesting, label, "number")
    for key in _NUMBER_CONSTRAINTS:
        target.pop(key, None)


def _empty_object() -> Schema:
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def _show(nesting: Sequence[str]) -> str:
    return ".".join(nesting) if nesting else "the schema root"


def _add(
    schema: Schema,
    nesting: Sequence[str],
    label: str | None,
    default_label: str,
    field: Schema,
    required: bool,
) -> str:
    properties = get_properties(schema, nesting)
    if label is None:
        label = _numbered_label(schema, properties, default_label)
    elif not label:
        msg = "Field label must not be empty"
        raise SchemaEditError(msg)
    elif label in properties:
        msg = f"Field {label!r} already exists at {_show(nesting)}"
        raise DuplicateFieldError(msg)
    properties[label] = field
    if required:
        set_required_field_status(schema, nesting, label, True)
    return label


def _numbered_label(schema: Schema, properties: dict[str, Schema], base: str) -> str:
    """``base_N``, N being one more than the ``base_*`` keys in the whole schema.

    Bumped further while the label is taken where it is being added.
    """
    number = _count_labels(schema, base) + 1
    while f"{base}_{number}" in properties:
        number += 1
    return f"{base}_{number}"


def _count_labels(node: Schema, base: str) -> int:
    count = 0
    for key, value in node.items():
        if key.rsplit("_", 1)[0] == base and "_" in key:
            count += 1
        if isinstance(value, dict):
            count += _count_labels(value, base)
    return count


def _move(schema: Schema, nesting: Sequence[str], label: str, step: int) -> None:
    properties = get_properties(schema, nesting)
    get_field(schema, nesting, label)
    order = list(properties)
    index = order.index(label)
    other = index + step
    if 0 <= other < len(order):
        order[index], order[other] = order[other], order[index]
        reordered = {key: properties[key] for key in order}
        properties.clear()
        properties.update(reordered)
    _sort_required(schema, nesting)


def _sort_required(schema: Schema, nesting: Sequence[str]) -> None:
    order = list(get_properties(schema, nesting))
    reqs = get_required_list(schema, nesting)
    reqs.sort(key=lambda name: order.index(name) if name in order else len(order))


def _value_schema(field: Schema) -> Schema:
    """The field itself, or the items of an array field."""
    if field.get("type") == "array":
        return field.setdefault("items", {})
    return field


def _typed(schema: Schema, nesting: Sequence[str], label: str, expected: str) -> Schema:
    target = _value_schema(get_field(schema, nesting, label))
    if target.get("type") != expected:
        msg = (
            f"Field {label!r} at {_show(nesting)} is "
            f"{target.get('type')!r}, not {expected!r}"
        )
        raise FieldTypeError(msg)
    return target


def _set(target: Schema, key: str, value: Any) -> None:
    if value is None:
        target.pop(key, None)
    else:
        target[key] = value


def _set_bound(
    target: Schema, inclusive_key: str, exclusive_key: str, value: Any, exclusive: bool
) -> None:
    key, other = (
        (exclusive_key, inclusive_key) if exclusive else (inclusive_key, exclusive_key)
    )
    if value is None:
        target.pop(key, None)
    else:
        target[key] = value
        target.pop(other, None)
