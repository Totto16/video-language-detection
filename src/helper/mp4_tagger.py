import struct
from collections.abc import Generator
from io import BufferedIOBase
from pathlib import Path
from typing import Optional, Self


def read_checked(f: BufferedIOBase, amount: int) -> bytes:
    if amount < 0:
        msg = "Invalid checked read, read amount negative"
        raise ValueError(msg)

    value = f.read(amount)
    if len(value) != amount:
        msg = f"Read failed to produce {amount} bytes, got {len(value)}"
        raise RuntimeError(msg)

    return value


class MP4Box:
    type: bytes
    start: int
    size: int
    header_size: int

    def __init__(
        self: Self, typ: bytes, start: int, size: int, header_size: int,
    ) -> None:
        self.type = typ
        self.start = start
        self.size = size
        self.header_size = header_size

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def payload_start(self) -> int:
        return self.start + self.header_size

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MP4Box":
        f.seek(offset)

        hdr = read_checked(f, 8)

        size, typ = struct.unpack(">I4s", hdr)

        if size == 1:
            ext = read_checked(f, 8)

            largesize = struct.unpack(">Q", ext)[0]

            if largesize < 16:
                msg = f"Invalid extended box size {largesize}"
                raise RuntimeError(msg)

            return MP4Box(typ, offset, largesize, 16)

        if size == 0:
            f.seek(0, 2)
            eof = f.tell()
            return MP4Box(typ, offset, eof - offset, 8)

        return MP4Box(typ, offset, size, 8)


CONTAINER_BOXES: set[bytes] = {
    b"moov",
    b"trak",
    b"mdia",
    b"minf",
    b"stbl",
    b"edts",
    b"dinf",
    b"udta",
    b"meta",
    b"moof",
    b"traf",
    b"mfra",
}


def iter_boxes(f: BufferedIOBase, start: int, end: int) -> Generator[MP4Box]:
    pos = start

    while pos < end:
        box = MP4Box.read_from_stream(f, pos)

        if box.size < 8:
            raise RuntimeError(f"Invalid box")

        if box.size < box.header_size:
            raise RuntimeError(f"Invalid box size {box.size} for {box.type!r} at {pos}")

        if pos + box.size > end:
            raise RuntimeError(
                f"Box {box.type!r} at {pos} extends past parent boundary"
            )

        if box.type == b"cmov":
            raise RuntimeError("Compressed movie box (cmov) not supported")

        if box.type == b"moof":
            raise RuntimeError("Fragmented MP4 (moof) not supported")

        yield box
        pos += box.size


class MHDBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.start, parent.size, parent.header_size)

    @staticmethod
    def get_from_normal_box(box: MP4Box) -> "MHDBox":
        if box.type != b"mdhd":
            msg = f"Invalid MHDBox box with type: {box.type!s}"
            raise RuntimeError(msg)
        return MHDBox(box)

    @staticmethod
    def find_audio_mdhd_boxes(f: BufferedIOBase) -> Generator["MHDBox"]:
        f.seek(0, 2)
        filesize = f.tell()

        stack: list[tuple[int, int, list[bytes]]] = [(0, filesize, [])]

        while stack:
            start, end, path = stack.pop()

            for box in iter_boxes(f, start, end):

                if box.type == b"trak":
                    trak_box = TrakBox.get_from_normal_box(box)

                    hdlr = trak_box.get_hdlr_type(f)

                    if hdlr is None:
                        raise RuntimeError("Missing hdlr box in trak")

                    if hdlr != b"soun":
                        print("Non sound trak", hdlr)
                        # skip non-audio tracks
                        continue

                if box.type == b"mdhd":
                    current = path[-2:]
                    if current != [
                        b"trak",
                        b"mdia",
                    ]:
                        print(current)
                        raise RuntimeError("invalid box hierarchy")

                    yield MHDBox.get_from_normal_box(box)

                if box.type in CONTAINER_BOXES:
                    stack.append((box.payload_start, box.end, [*path, box.type]))

    @staticmethod
    def decode_language(value: int) -> str:
        chars = []

        for shift in (10, 5, 0):
            v = (value >> shift) & 0x1F

            if v < 1 or v > 26:
                msg = f"Invalid ISO639 character value {v}"
                raise RuntimeError(msg)

            chars.append(chr(v + 0x60))

        return "".join(chars)

    @staticmethod
    def encode_language(code: str) -> int:
        if len(code) != 3:
            msg = "language code must be 3 characters"
            raise ValueError(msg)

        code = code.lower()

        value = 0
        for ch in code:
            n = ord(ch) - 0x60
            if not (1 <= n <= 26):
                msg = f"invalid language character: {ch}"
                raise ValueError(msg)
            value = (value << 5) | n

        return value

    def read_mdhd_language(self: Self, f: BufferedIOBase) -> tuple[int, str]:
        f.seek(self.payload_start)

        version = read_checked(f, 1)[0]

        # TODO: define the structs for the size!
        MIN_MDHD_V0 = 24
        MIN_MDHD_V1 = 36

        print("mdhd version: ", version)

        payload_size = self.size - self.header_size

        if version == 0:
            if payload_size < MIN_MDHD_V0:
                raise RuntimeError("Truncated mdhd v0")
        elif version == 1:
            if payload_size < MIN_MDHD_V1:
                raise RuntimeError("Truncated mdhd v1")
        else:
            msg = "Invalid mdhd version"
            raise RuntimeError(msg)

        # flags
        read_checked(f, 3)

        if version == 1:
            language_offset = self.payload_start + 4 + 8 + 8 + 4 + 8
        elif version == 0:
            language_offset = self.payload_start + 4 + 4 + 4 + 4 + 4
        else:
            msg = "Invalid mdhd version"
            raise RuntimeError(msg)

        if language_offset + 2 > self.end:
            msg = "Language field outside mdhd bounds"
            raise RuntimeError(msg)

        f.seek(language_offset)

        lang_bytes = read_checked(f, 2)
        packed = struct.unpack(">H", lang_bytes)[0]

        return language_offset, MHDBox.decode_language(packed)

    def patch_mdhd_language(self: Self, f: BufferedIOBase, new_language: str) -> str:
        offset, old_language = self.read_mdhd_language(f)

        packed = MHDBox.encode_language(new_language)

        f.seek(offset)
        f.write(struct.pack(">H", packed))
        f.flush()

        f.seek(offset)
        verify_bytes = read_checked(f, 2)
        verify = struct.unpack(">H", verify_bytes)[0]

        if verify != packed:
            msg = "Invalid overwrite"
            raise RuntimeError(msg)

        return old_language


class TrakBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.start, parent.size, parent.header_size)

    @staticmethod
    def get_from_normal_box(box: MP4Box) -> "TrakBox":
        if box.type != b"trak":
            msg = f"Invalid TrakBox box with type: {box.type!s}"
            raise RuntimeError(msg)
        return TrakBox(box)

    def get_hdlr_type(self: Self, f: BufferedIOBase) -> Optional[bytes]:
        """
        Returns handler type like b'soun', b'vide', etc.
        """
        mdia_start = None

        for box in iter_boxes(f, self.payload_start, end=self.end):
            if box.type == b"mdia":
                mdia_start = box
                break

        if not mdia_start:
            return None

        for box in iter_boxes(f, mdia_start.payload_start, mdia_start.end):
            if box.type == b"hdlr":
                # skip version(1)+flags(3)+predefined(4)
                f.seek(box.payload_start + 8)
                return read_checked(f, 4)

        return None


def list_languages(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []

    with path.open("rb") as f:

        for i, mdhd in enumerate(MHDBox.find_audio_mdhd_boxes(f), start=1):
            _, lang = mdhd.read_mdhd_language(f)
            result.append((i, lang))

    return result


def patch_languages(path: Path, new_language: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []

    with path.open("rb+") as f:
        for i, mdhd in enumerate(MHDBox.find_audio_mdhd_boxes(f), start=1):
            old = mdhd.patch_mdhd_language(f, new_language)
            result.append((i, old))

    return result


if __name__ == "__main__":
    mp4 = Path("test.mp4")

    list_languages(mp4)

    patch_languages(mp4, "ger")

    list_languages(mp4)


# TODO every read has to be checked, every struct unpack has to be checked, every [0] has to be checked!
