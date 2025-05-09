from collections.abc import Callable, Mapping, MutableMapping
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    cast,
)

from apischema import schema
from apischema.json_schema import (
    deserialization_schema,
    serialization_schema,
)

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


def narrow_type(replace: tuple[str, Any]) -> Callable[[dict[str, Any]], None]:
    name, type_desc = replace

    def narrow_schema(schema: dict[str, Any]) -> None:
        if schema.get("properties") is not None and isinstance(
            schema["properties"],
            dict,
        ):
            resulting_type: SchemaType = get_schema(type_desc)
            del resulting_type["$schema"]

            if cast(dict[str, Any], schema["properties"]).get(name) is None:
                msg = f"Narrowing type failed, type is not present. key '{name}'"
                raise RuntimeError(msg)

            schema["properties"][name] = resulting_type

    return narrow_schema


# from: https://wyfo.github.io/apischema/0.18/json_schema/
# schema extra can be callable to modify the schema in place
def to_one_of(schema: dict[str, Any]) -> None:
    if "anyOf" in schema:
        schema["oneOf"] = schema.pop("anyOf")


OneOf = schema(extra=to_one_of)

Deprecated = schema(deprecated=True)

type AnnotatedOptional[T] = Annotated[Optional[T], OneOf]
