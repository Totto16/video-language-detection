from pathlib import Path
from typing import Optional, Self

import mutagen._file as mutagen
from mutagen._util import MutagenError


class VideoTagger:
    __instance: mutagen.FileType

    def __init__(self: Self, instance: mutagen.FileType) -> None:
        self.__instance = instance

    @staticmethod
    def get_handle(file: Path) -> Optional["VideoTagger"]:
        instance = mutagen.File(file, easy=False)

        if instance is None:
            return None

        return VideoTagger(instance)

    def write_metadata(self: Self, metadata: dict[str, str]) -> None:
        try:
            for key, value in metadata.items():
                self.__instance[key] = value

            self.__instance.save()
        except MutagenError as err:
            msg = "tag error"
            raise RuntimeError(msg) from err
