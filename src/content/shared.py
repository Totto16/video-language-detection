from enum import Enum


class ScanType(Enum):
    first_scan = "first_scan"
    rescan = "rescan"


class ScanKind(Enum):
    metadata = "metadata"
    language = "language"