from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True, repr=True)
class MetadataHandle:
    provider: str
    data: dict[str, Any]


HandlesType = Optional[list[MetadataHandle]]
