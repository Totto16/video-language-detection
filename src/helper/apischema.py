from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Literal,
    Optional,
    TypedDict,
    Unpack,
    cast,
)

from apischema import schema
from apischema.json_schema import (
    deserialization_schema,
    serialization_schema,
)
from apischema.objects import ObjectField, object_fields, set_object_fields

type EmitType = Literal["deserialize", "serialize"]


type SchemaType = MutableMapping[str, Any]


def get_schema(
    any_type: Any,
    *,
    additional_properties: Optional[bool] = None,
    all_refs: Optional[bool] = None,
    emit_type: Optional[EmitType] = None,
) -> SchemaType:
    result: Mapping[str, Any] = deserialization_schema(
        any_type,
        additional_properties=additional_properties,
        all_refs=all_refs,
    )

    result2 = serialization_schema(
        any_type,
        additional_properties=additional_properties,
        all_refs=all_refs,
    )

    if result != result2:
        if emit_type is None:
            msg = "Deserialization and Serialization scheme mismatch"
            raise RuntimeError(msg)
        if emit_type == "serialize":
            return cast(SchemaType, result2)

    return cast(SchemaType, result)


def get_sub_schema(
    any_type: Any,
    *,
    additional_properties: Optional[bool] = None,
    all_refs: Optional[bool] = None,
    emit_type: Optional[EmitType] = None,
) -> tuple[SchemaType, Optional[dict[str, Any]]]:
    schema = get_schema(
        any_type=any_type,
        additional_properties=additional_properties,
        all_refs=all_refs,
        emit_type=emit_type,
    )

    if schema.get("$schema", None) is not None:
        del schema["$schema"]

    defs = schema.get("$defs", None)
    if defs is not None:
        del schema["$defs"]

    for key in schema:
        if key.startswith("$"):
            if key == "$ref":
                continue

            msg = f"Error: invalid json meta key: {key}"
            raise RuntimeError(msg)

    return (schema, defs)


class SchemaOptions(TypedDict, total=False):
    additional_properties: bool
    all_refs: bool
    emit_type: EmitType


def narrow_type(
    replace: tuple[str, Any],
    **options: Unpack[SchemaOptions],
) -> Callable[[dict[str, Any]], None]:
    name, type_desc = replace

    def narrow_schema(schema: dict[str, Any]) -> None:
        if schema.get("properties") is not None and isinstance(
            schema["properties"],
            dict,
        ):
            resulting_type, defs = get_sub_schema(type_desc, **options)

            if defs is not None:
                msg = "Error: defs can't be used here, use another mean to get the defs into the global scope!"
                raise ValueError(msg)

            if cast(dict[str, Any], schema["properties"]).get(name) is None:
                msg = f"Narrowing type failed, type is not present. key '{name}'"
                raise RuntimeError(msg)

            schema["properties"][name] = resulting_type

    return narrow_schema


def define_schema(*fields: ObjectField) -> Callable[[Any], Any]:
    def lazy_fn() -> Sequence[ObjectField]:
        return fields

    return define_schema_lazy(lazy_fn)


def define_schema_lazy(fn: Callable[[], Sequence[ObjectField]]) -> Callable[[Any], Any]:
    def decorator(cls: Any) -> Any:
        set_object_fields(cls, fn)
        return cls

    return decorator


def use_schema_from(type_desc: Any) -> Callable[[Any], Any]:
    def decorator(cls: Any) -> Any:
        fields = [val for key, val in object_fields(type_desc).items()]
        set_object_fields(cls, fields)
        return cls

    return decorator


def replace_schema_with(
    type_desc: Any,
    **options: Unpack[SchemaOptions],
) -> Callable[[dict[str, Any]], None]:
    def replace_schema_with_impl(schema: dict[str, Any]) -> None:
        resulting_type, defs = get_sub_schema(type_desc, **options)
        if defs is not None:
            msg = "Error: defs can't be used here, use another mean to get the defs into the global scope!"
            raise ValueError(msg)

        for key in [*schema.keys()]:
            del schema[key]

        for key, value in resulting_type.items():
            schema[key] = value  # noqa: PERF403

    return replace_schema_with_impl


# from: https://wyfo.github.io/apischema/0.18/json_schema/
# schema extra can be callable to modify the schema in place
def to_one_of(schema: dict[str, Any]) -> None:
    if "anyOf" in schema:
        schema["oneOf"] = schema.pop("anyOf")


OneOf = schema(extra=to_one_of)

Deprecated = schema(deprecated=True)
