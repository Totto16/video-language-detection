from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, Self, TypeIs, override

from apischema import alias, deserializer, schema, serialize, serializer
from apischema.objects import ObjectField

from helper.apischema import OneOf, define_schema_lazy
from helper.decorator import decorate_class
from helper.translation import get_translator

_ = get_translator()


@schema()
@dataclass(slots=True, repr=True)
class MetadataHandleHelper:
    provider: str
    data: Any


def make_provider_schema_tmdb() -> Sequence[ObjectField]:
    from content.metadata.provider.tmdb import TMDBProvider  # noqa: PLC0415

    return [v for k, v in TMDBProvider.get_metadata_schema().items()]


def make_provider_schema_imdb() -> Sequence[ObjectField]:
    from content.metadata.provider.imdb import IMDBProvider  # noqa: PLC0415

    return [v for k, v in IMDBProvider.get_metadata_schema().items()]

class HandleImpl(ABC):
    __provider: str
    __data: Any

    def __init__(
        self: Self,
        provider: str,
        data: Any,
    ) -> None:
        super().__init__()
        self.__provider = provider
        self.__data = data

    @property
    def provider(self: Self) -> str:
        return self.__provider

    @property
    def data(self: Self) -> Any:
        return self.__data

    @abstractmethod
    def to_handle(self: Self) -> "MetadataHandle": ...


@define_schema_lazy(fn=make_provider_schema_imdb)
class ImdbHandleImpl(HandleImpl):

    def __init__(
        self: Self,
        provider: str,
        data: Any,
    ) -> None:
        super().__init__(provider, data)

    @override
    def to_handle(self: Self) -> "MetadataHandle":
        msg = _(
            "Deserialization error: Not implemented for provider {provider}"  # noqa: COM812
        ).format(provider=self.provider)
        raise RuntimeError(msg)


@define_schema_lazy(fn=make_provider_schema_tmdb)
class TmdbHandleImpl(HandleImpl):

    def __init__(
        self: Self,
        provider: str,
        data: Any,
    ) -> None:
        super().__init__(provider, data)

    @override
    def to_handle(self: Self) -> "MetadataHandle":
        return MetadataHandle(
            self.provider,
            self.data,
        )


MetadataHandleSchema = Annotated[ImdbHandleImpl | TmdbHandleImpl, OneOf]

@schema()
@dataclass(slots=False, repr=True)
class MetadataHandle:
    __provider: str = field(metadata=alias("provider"))
    __data: Any = field(metadata=alias("data"))

    def __init__(self: Self, provider: str, data: Any) -> None:
        self.__provider = provider
        self.__data = data

    @property
    def provider(self: Self) -> str:
        return self.__provider

    @property
    def data(self: Self) -> Any:
        return self.__data

    @serializer
    def serialize(self: Self) -> dict[str, Any]:
        serialized_dict: dict[str, Any] = serialize(
            MetadataHandleHelper,
            MetadataHandleHelper(self.__provider, self.__data),
        )
        return serialized_dict

    @deserializer
    @staticmethod
    def deserialize(data: MetadataHandleSchema) -> "MetadataHandle":
        return data.to_handle()

    def __str__(self: Self) -> str:
        return f"<MetadataHandle provider: {self.__provider} data: {self.__data}>"

    def __repr__(self: Self) -> str:
        return str(self)

class SkipHandle:
    # serialize the same as None
    @serializer
    def serialize(self: Self) -> None:
        return None

    def __str__(self: Self) -> str:
        return "<SkipHandle>"

    def __repr__(self: Self) -> str:
        return str(self)


@schema()
@dataclass(slots=True, repr=True)
class SkipMetadata:
    def __str__(self: Self) -> str:
        return "<SkipMetadata>"

    def __repr__(self: Self) -> str:
        return str(self)


def should_skip_metadata(
    data: tuple[MetadataHandle, MetadataHandle] | MetadataHandle | SkipHandle | Any,
) -> TypeIs[SkipHandle | SkipMetadata]:
    if isinstance(data, (SkipHandle, SkipMetadata)):
        return True

    if isinstance(data, MetadataHandle):
        return should_skip_metadata(data.data)

    if isinstance(data, tuple):
        return should_skip_metadata(data[0]) or should_skip_metadata(data[1])

    return False


type HandlesType = Optional[list[MetadataHandle] | SkipHandle]

InternalMetadataType = Annotated[Optional[MetadataHandle | SkipHandle], OneOf]
