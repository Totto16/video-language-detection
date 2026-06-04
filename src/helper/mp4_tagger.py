#!/usr/bin/env python3

from collections.abc import Generator
from io import BytesIO
import struct
from pathlib import Path
from typing import Optional


class MP4Box:
    def __init__(self, typ: bytes, start: int, size: int, header_size: int) -> None:
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


CONTAINER_BOXES = {
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


def decode_language(value):
    chars = []

    for shift in (10, 5, 0):
        v = (value >> shift) & 0x1F

        if 1 > v or v > 26:
            raise RuntimeError(f"Invalid ISO639 character value {v}")

        chars.append(chr(v + 0x60))

    return "".join(chars)


def encode_language(code):
    if len(code) != 3:
        raise ValueError("language code must be 3 characters")

    code = code.lower()

    value = 0
    for ch in code:
        n = ord(ch) - 0x60
        if not (1 <= n <= 26):
            raise ValueError(f"invalid language character: {ch}")
        value = (value << 5) | n

    return value


def read_box(f: BytesIO, offset: int) -> Optional[MP4Box]:
    f.seek(offset)

    hdr = f.read(8)
    if len(hdr) != 8:
        return None

    size, typ = struct.unpack(">I4s", hdr)
    # print(typ, size)

    if size == 1:
        ext = f.read(8)
        if len(ext) != 8:
            raise RuntimeError("Truncated largesize")

        largesize = struct.unpack(">Q", ext)[0]

        if largesize < 16:
            raise RuntimeError(f"Invalid extended box size {largesize}")

        return MP4Box(typ, offset, largesize, 16)

    if size == 0:
        f.seek(0, 2)
        eof = f.tell()
        return MP4Box(typ, offset, eof - offset, 8)

    return MP4Box(typ, offset, size, 8)


def iter_boxes(f: BytesIO, start: int, end: int) -> Generator[MP4Box]:
    pos = start

    while pos < end:
        box = read_box(f, pos)
        if box is None:
            raise RuntimeError(f"Failed to read box at {pos}")

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
    def __init__(self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.start, parent.size, parent.header_size)


def get_mdhd_box(box) -> MHDBox:

    return MHDBox(box)


def get_hdlr_type(f: BytesIO, trak_box: MP4Box) -> Optional[bytes]:
    """
    Returns handler type like b'soun', b'vide', etc.
    """
    mdia_start = None

    # find mdia inside trak
    for box in iter_boxes(f, trak_box.payload_start, end=trak_box.end):
        if box.type == b"mdia":
            mdia_start = box
            break

    if not mdia_start:
        return None

    # find hdlr inside mdia
    for box in iter_boxes(f, mdia_start.payload_start, mdia_start.end):
        if box.type == b"hdlr":
            f.seek(box.payload_start + 8)  # skip version(1)+flags(3)+predefined(4)
            handler = f.read(4)
            return handler

    return None


def find_audio_mdhd_boxes(f: BytesIO) -> Generator[MP4Box]:
    f.seek(0, 2)
    filesize = f.tell()

    stack: list[tuple[int, int, list[bytes]]] = [(0, filesize, [])]

    while stack:
        start, end, path = stack.pop()

        for box in iter_boxes(f, start, end):

            if box.type == b"trak":
                hdlr = get_hdlr_type(f, trak_box=box)

                if hdlr is None:
                    raise RuntimeError("Missing hdlr box in trak")

                if hdlr != b"soun":
                    print("Non sound trak", hdlr)
                    continue  # skip non-audio tracks

            if box.type == b"mdhd":
                current = path[-2:]
                if current != [
                    b"trak",
                    b"mdia",
                ]:
                    print(current)
                    raise RuntimeError("invalid box hierarchy")

                yield get_mdhd_box(box)

            if box.type in CONTAINER_BOXES:
                stack.append((box.payload_start, box.end, [*path, box.type]))


def read_mdhd_language(f, mdhd_box: MHDBox):
    f.seek(mdhd_box.payload_start)

    version = f.read(1)[0]

    MIN_MDHD_V0 = 24
    MIN_MDHD_V1 = 36

    print("mdhd version: ", version)

    payload_size = mdhd_box.size - mdhd_box.header_size

    if version == 0:
        if payload_size < MIN_MDHD_V0:
            raise RuntimeError("Truncated mdhd v0")
    elif version == 1:
        if payload_size < MIN_MDHD_V1:
            raise RuntimeError("Truncated mdhd v1")
    else:
        raise RuntimeError("Invalid mdhd version")

    f.read(3)  # flags

    if version == 1:
        language_offset = mdhd_box.payload_start + 4 + 8 + 8 + 4 + 8
    elif version == 0:
        language_offset = mdhd_box.payload_start + 4 + 4 + 4 + 4 + 4
    else:
        raise RuntimeError("Invalid mdhd version")

    if language_offset + 2 > mdhd_box.end:
        raise RuntimeError("Language field outside mdhd bounds")

    f.seek(language_offset)

    packed = struct.unpack(">H", f.read(2))[0]

    return language_offset, decode_language(packed)


def patch_mdhd_language(f, mdhd_box, new_language):
    raise NotImplementedError("ERRO")
    offset, old_language = read_mdhd_language(f, mdhd_box)

    packed = encode_language(new_language)

    f.seek(offset)
    f.write(struct.pack(">H", packed))
    f.flush()

    f.seek(offset)
    verify = struct.unpack(">H", f.read(2))[0]

    if verify != packed:
        raise RuntimeError("Invalid overwrite")

    return old_language


def list_languages(path: Path) -> None:
    with path.open("rb") as f:

        for i, mdhd in enumerate(find_audio_mdhd_boxes(f), start=1):
            _, lang = read_mdhd_language(f, mdhd)
            print(f"Track {i}: {lang}")


def patch_languages(path: Path, new_language: str) -> None:
    with path.open("röööööb") as f:
        for i, mdhd in enumerate(find_audio_mdhd_boxes(f), start=1):
            old = patch_mdhd_language(f, mdhd, new_language)
            print(f"Track {i}: {old} -> {new_language}")


if __name__ == "__main__":
    mp4 = Path("test.mp4")

    print("Before:")
    list_languages(mp4)

    # patch_languages(mp4, "eng")

    print("\nAfter:")
    list_languages(mp4)


# TODO every read has to be checked, every struct unpack has to be checked, every [0] has to be checked!
