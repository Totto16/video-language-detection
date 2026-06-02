from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Self, override

import mutagen._file as mutagen
from mutagen._util import MutagenError

from helper.manager import CounterInterface, ManagerInterface


class MutagenFileWrapper(IOInterface):
    pass


VIDEO_FILE_TAG_UPDATE_BAR_FORMAT: str = (
    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:!.2j}{unit} / {total:!.2j}{unit} "
    "[{elapsed}<{eta}, {rate:!.2j}{unit}/s]"
)


class VideoTaggerWriter:
    __filething: MutagenFileWrapper
    __instance: mutagen.FileType
    __manager: ManagerInterface

    def __init__(
        self: Self,
        filething: MutagenFileWrapper,
        instance: mutagen.FileType,
        manager: ManagerInterface,
    ) -> None:
        self.__instance = instance
        self.__filething = filething
        self.__manager = manager

    def __save_impl(self: Self) -> None:
        size: float = self.__filething.new_size()

        bar: CounterInterface = self.__manager.counter(
            total=size,
            desc="update video tags",
            unit="B",
            leave=False,
            bar_format=VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
            color="red",
        )
        bar.update(0, force=True)

        try:
            self.__filething.on_progress(
                lambda data: bar.update(float(len(data))),
            )

            self.__instance.save()
        except MutagenError as err:
            msg = "tag error"
            raise RuntimeError(msg) from err
        finally:
            bar.close(clear=True)

    def write_metadata(
        self: Self,
        metadata: dict[str, str],
    ) -> None:
        for key, value in metadata.items():
            self.__instance[key] = value

        self.__save_impl()


class VideoTagger:
    __filething: MutagenFileWrapper
    __instance: mutagen.FileType

    def __init__(
        self: Self,
        filething: MutagenFileWrapper,
        instance: mutagen.FileType,
    ) -> None:
        self.__instance = instance
        self.__filething = filething

    @staticmethod
    def get_handle(file: Path) -> Optional["VideoTagger"]:

        filething = MutagenFileWrapper(file=file)

        instance = mutagen.File(filething=filething, easy=False)

        if instance is None:
            return None

        return VideoTagger(filething, instance)

    def writer(
        self: Self, manager: ManagerInterface
    ) -> AbstractContextManager[VideoTaggerWriter]:

        filething = self.__filething
        instance = self.__instance

        class VideoTaggerWriterCtx(AbstractContextManager[VideoTaggerWriter]):

            def __init__(self: Self) -> None:
                pass

            @override
            def __enter__(self: Self) -> VideoTaggerWriter:
                return VideoTaggerWriter(
                    filething=filething,
                    instance=instance,
                    manager=manager,
                )

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                return False

        return VideoTaggerWriterCtx()
