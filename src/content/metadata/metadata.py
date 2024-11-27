from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, repr=True)
class MetadataHandle:
    provider: str
    data: dict[str, Any]
