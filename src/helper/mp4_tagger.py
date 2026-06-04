import struct
from collections.abc import Generator
from io import BufferedIOBase
from pathlib import Path
from typing import Any, Literal, Optional, Self


def read_checked(f: BufferedIOBase, amount: int) -> bytes:
    if amount < 0:
        msg = "Invalid checked read, read amount negative"
        raise ValueError(msg)

    value = f.read(amount)
    if len(value) != amount:
        msg = f"Read failed to produce {amount} bytes, got {len(value)}"
        raise RuntimeError(msg)

    return value


BytesOrder = Literal[
    "@",
    "=",
    "<",
    ">",
    "!",
]


class Unpacker:

    @staticmethod
    def __unpack_impl(
        byte_order: BytesOrder,
        formats: str,
        value: bytes,
    ) -> tuple[Any, ...]:

        fmt: str = f"{byte_order}{formats}"

        size = struct.calcsize(fmt)

        if len(value) != size:
            msg = f"Unpacking has wrong input: expected bytes with size {size} but got {len(value)}"
            raise RuntimeError(msg)

        return struct.unpack(fmt, value)

    @staticmethod
    def unpack_default(
        formats: str,
        value: bytes,
    ) -> tuple[Any, ...]:
        return Unpacker.__unpack_impl(">", formats, value)

    @staticmethod
    def unpack_default_sized(formats: str, value: bytes, size: int) -> tuple[Any, ...]:
        result = Unpacker.unpack_default(formats, value)

        if len(result) != size:
            msg = f"Expected unpack to produce {size} values, but got {len(result)}"
            raise RuntimeError(msg)

        return result

    @staticmethod
    def unpack_default_one(
        formats: str,
        value: bytes,
    ) -> Any:
        result = Unpacker.unpack_default_sized(formats, value, size=1)

        return result[0]


class ISOAtomName:
    __value: bytes

    def __init__(self: Self, value: bytes) -> None:
        self.__value = value

        if len(value) != 4:
            msg = f"Invalid atom name length {len(value)}"
            raise ValueError(msg)

        if not value.islower():
            msg = f"Atom name is not lowercase {value!s}"
            raise ValueError(msg)

    @property
    def value(self: Self) -> bytes:
        return self.__value

    def __str__(self: Self) -> str:
        return str(self.__value)

    def __repr__(self: Self) -> str:
        return repr(self.__value)

    def __hash__(self: Self) -> int:
        return hash(self.__value)

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, ISOAtomName):
            return self.__value == other.__value

        if isinstance(other, str):
            return self.__value == other.encode()

        if isinstance(other, bytes):
            return self.__value == other

        return False


CMOV_ATOM_NAME: ISOAtomName = ISOAtomName(b"cmov")
MOOF_ATOM_NAME: ISOAtomName = ISOAtomName(b"moof")
MOOV_ATOM_NAME: ISOAtomName = ISOAtomName(b"moov")
UUID_ATOM_NAME: ISOAtomName = ISOAtomName(b"uuid")
TRAK_ATOM_NAME: ISOAtomName = ISOAtomName(b"trak")
MDIA_ATOM_NAME: ISOAtomName = ISOAtomName(b"mdia")
MINF_ATOM_NAME: ISOAtomName = ISOAtomName(b"minf")
STBL_ATOM_NAME: ISOAtomName = ISOAtomName(b"stbl")
EDTS_ATOM_NAME: ISOAtomName = ISOAtomName(b"edts")
DINF_ATOM_NAME: ISOAtomName = ISOAtomName(b"dinf")
UDTA_ATOM_NAME: ISOAtomName = ISOAtomName(b"udta")
META_ATOM_NAME: ISOAtomName = ISOAtomName(b"meta")
TRAF_ATOM_NAME: ISOAtomName = ISOAtomName(b"traf")
MFRA_ATOM_NAME: ISOAtomName = ISOAtomName(b"mfra")
MDHD_ATOM_NAME: ISOAtomName = ISOAtomName(b"mdhd")
SOUN_ATOM_NAME: ISOAtomName = ISOAtomName(b"soun")
HDLR_ATOM_NAME: ISOAtomName = ISOAtomName(b"hdlr")


class BoxSpan:
    start: int
    size: int
    header_size: int

    def __init__(
        self: Self,
        start: int,
        size: int,
        header_size: int,
    ) -> None:
        self.start = start
        self.size = size
        self.header_size = header_size

        if self.size < 8:
            msg = f"Invalid box: sitze too small: {self.size}"
            raise RuntimeError(msg)

        if self.size < self.header_size:
            msg = f"Invalid box size {self.size} at {self.start}"
            raise RuntimeError(msg)

    @property
    def end(self: Self) -> int:
        return self.start + self.size

    @property
    def payload_start(self: Self) -> int:
        return self.start + self.header_size

    @property
    def payload_size(self: Self) -> int:
        return self.size - self.header_size

    def add_header_size(self: Self, header_size: int) -> None:
        self.header_size = self.header_size + header_size
        if self.size < self.header_size:
            msg = f"Invalid box size {self.size} at {self.start}"
            raise RuntimeError(msg)


class MP4Box:
    type: ISOAtomName
    span: BoxSpan
    container: bool

    def __init__(self: Self, typ: ISOAtomName, span: BoxSpan, container: bool) -> None:
        self.type = typ
        self.span = span
        self.container = container

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MP4Box":
        # spec: ISO/IEC 14496-12
        # MP4 atom / ISO box structure:
        # size | 4 bytes | unsigned integer
        # type | 4 bytes | char[4]
        # ... extension, dependent on size

        # aligned(8) class Box (
        #     unsigned int(32) boxtype,
        #     optional unsigned int(8)[16] extended_type
        # ) {
        #     unsigned int(32) size;
        #     unsigned int(32) type = boxtype;

        #     if (size==1) {
        #         unsigned int(64) largesize;
        #     } else if (size==0) {
        #         # box extends to end of file
        #     }

        #     if (boxtype=="uuid") {
        #         unsigned int(8)[16] usertype = extended_type;
        #     }
        # }

        f.seek(offset)

        hdr = read_checked(f, 8)

        size, typ_raw = Unpacker.unpack_default_sized("I4s", hdr, size=2)

        typ = ISOAtomName(typ_raw)

        if typ == UUID_ATOM_NAME:
            msg = "'uuid' type not implemented, the box header size differs with that type"
            raise RuntimeError(msg)

        if typ == CMOV_ATOM_NAME:
            msg = f"Compressed movie box '{CMOV_ATOM_NAME}' not supported"
            raise RuntimeError(msg)

        if typ == MOOF_ATOM_NAME:
            msg = f"Fragmented MP4 '{MOOF_ATOM_NAME}' not supported"
            raise RuntimeError(msg)

        if size == 1:
            ext = read_checked(f, 8)

            largesize = Unpacker.unpack_default_one("Q", ext)

            if largesize < 16:
                msg = f"Invalid extended box size {largesize}"
                raise RuntimeError(msg)

            span = BoxSpan(offset, largesize, header_size=16)
            return MP4Box(typ, span, container=False)

        if size == 0:
            f.seek(0, 2)
            eof = f.tell()
            span = BoxSpan(offset, eof - offset, header_size=8)
            return MP4Box(typ, span, container=False)

        span = BoxSpan(offset, size, header_size=8)
        return MP4Box(typ, span, container=False)


class MP4FullBox(MP4Box):
    version: int
    flags: bytes

    def __init__(
        self: Self, parent: MP4Box, container: bool, version: int, flags: bytes
    ) -> None:
        super().__init__(parent.type, parent.span, container=container)

        self.version = version
        self.flags = flags

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "MP4FullBox":
        # spec: ISO/IEC 14496-12
        # ISO full box structure:
        # box     | <box size> bytes | parent box
        # version | 1 byte | unsigned char
        # flags   | 3 bytes | unsigned char[3]

        # aligned(8) class FullBox(
        #     unsigned int(32) boxtype,
        #     unsigned int(8) v,
        #     bit(24) f
        # ) extends Box(boxtype) {
        #     unsigned int(8) version = v;
        #     bit(24) flags = f;
        # }

        f.seek(parent.span.payload_start)

        version = read_checked(f, 1)[0]

        flags = read_checked(f, 3)

        parent.span.add_header_size(4)

        return MP4FullBox(parent, container=False, version=version, flags=flags)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MP4FullBox":
        box = MP4Box.read_from_stream(f, offset)
        return MP4FullBox.__read_from_stream_impl(f, box)


class MediaHeaderBox(MP4FullBox):
    language_offset: int

    def __init__(self: Self, parent: MP4FullBox, language_offset: int) -> None:
        super().__init__(
            parent,
            container=False,
            version=parent.version,
            flags=parent.flags,
        )
        self.language_offset = language_offset

    @staticmethod
    def __read_from_stream_impl(
        f: BufferedIOBase, parent: MP4FullBox
    ) -> "MediaHeaderBox":
        # spec: ISO/IEC 14496-12
        # ISO media header box structure:
        # full_box     | <full box size> bytes | parent full box
        # ... data, dependent on version

        # aligned(8) class MediaHeaderBox
        # extends FullBox(
        #     ‘mdhd’,
        #     version,
        #     0
        # ) {
        #     if (version==1) {
        #         unsigned int(64) creation_time;
        #         unsigned int(64) modification_time;
        #         unsigned int(32) timescale;
        #         unsigned int(64) duration;
        #     } else { # version==0
        #         unsigned int(32) creation_time;
        #         unsigned int(32) modification_time;
        #         unsigned int(32) timescale;
        #         unsigned int(32) duration;
        #     }
        #     bit(1) pad = 0;
        #     unsigned int(5)[3] language;
        #     # ISO-639-2/T language code
        #     unsigned int(16) pre_defined = 0;
        # }

        f.seek(parent.span.payload_start)

        version_dependend_size: int

        if parent.version == 0:
            version_dependend_size = 4 + 4 + 4 + 4
        elif parent.version == 1:
            additional_header_size = 1
            version_dependend_size = 8 + 8 + 4 + 8
        else:
            msg = "Invalid mdhd version"
            raise RuntimeError(msg)

        additional_header_size = version_dependend_size + (2 + 2)

        if parent.span.payload_size < additional_header_size:
            msg = f"Truncated mdhd header {parent.span.payload_size} < {additional_header_size}"
            raise RuntimeError(msg)

        language_offset = parent.span.header_size + version_dependend_size

        if parent.span.start + language_offset + 2 > parent.span.end:
            msg = "Language field outside mdhd bounds"
            raise RuntimeError(msg)

        parent.span.add_header_size(additional_header_size)

        return MediaHeaderBox(parent, language_offset)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MediaHeaderBox":
        box = MP4FullBox.read_from_stream(f, offset)
        return MediaHeaderBox.__read_from_stream_impl(f, box)

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

    def read_language(self: Self, f: BufferedIOBase) -> str:
        f.seek(self.span.start + self.language_offset)

        lang_bytes = read_checked(f, 2)
        packed = Unpacker.unpack_default_one("H", lang_bytes)

        return MediaHeaderBox.decode_language(packed)

    def patch_language(self: Self, f: BufferedIOBase, new_language: str) -> None:
        packed = MediaHeaderBox.encode_language(new_language)

        f.seek(self.span.start + self.language_offset)

        packed_bytes = struct.pack(">H", packed)
        if len(packed_bytes) != 2:
            msg = "packed bytes are not of correct size"
            raise RuntimeError(msg)

        f.write(packed_bytes)
        f.flush()

        f.seek(self.span.start + self.language_offset)
        verify_bytes = read_checked(f, 2)
        verify = Unpacker.unpack_default_one("H", verify_bytes)

        if verify != packed:
            msg = "Invalid overwrite"
            raise RuntimeError(msg)


class MediaBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, container=True)

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "MediaBox":
        # spec: ISO/IEC 14496-12
        # ISO media box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class MediaBox extends Box(
        #     ‘mdia’
        #     ) {
        # }

        return MediaBox(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MediaBox":
        box = MP4Box.read_from_stream(f, offset)
        return MediaBox.__read_from_stream_impl(f, box)


class HandlerBox(MP4FullBox):
    handler_type: ISOAtomName

    def __init__(self: Self, parent: MP4FullBox, handler_type: ISOAtomName) -> None:
        super().__init__(
            parent,
            container=False,
            version=parent.version,
            flags=parent.flags,
        )
        self.handler_type = handler_type

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4FullBox) -> "HandlerBox":
        # spec: ISO/IEC 14496-12
        # ISO handler box structure:
        # box     | <full box size> bytes | parent full box
        # ... data, see below

        # aligned(8) class HandlerBox extends FullBox(
        #     ‘hdlr’,
        #     version = 0,
        #     0)
        # {
        #     unsigned int(32) pre_defined = 0;
        #     unsigned int(32) handler_type;
        #     const unsigned int(32)[3] reserved = 0;
        #     string name;
        # }

        f.seek(parent.span.payload_start)

        if parent.version != 0:
            msg = "Invalid hdlr version"
            raise RuntimeError(msg)

        f.seek(parent.span.payload_start + 4)

        handler_type_raw = read_checked(f, 4)
        handler_type = ISOAtomName(handler_type_raw)

        # omitting dynamic sized string "name"
        additional_header_size = 4 + 4 + (4 * 3)

        parent.span.add_header_size(additional_header_size)

        return HandlerBox(parent, handler_type)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "HandlerBox":
        box = MP4FullBox.read_from_stream(f, offset)
        return HandlerBox.__read_from_stream_impl(f, box)


class TrackBox(MP4Box):
    hdlr: HandlerBox

    def __init__(self: Self, parent: MP4Box, hdlr: HandlerBox) -> None:
        super().__init__(parent.type, parent.span, container=True)

        self.hdlr = hdlr

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "TrackBox":
        # spec: ISO/IEC 14496-12
        # ISO track box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class TrackBox extends Box(
        #     ‘trak’
        # ) {
        # }

        mdia_box: Optional[MediaBox] = None

        for box in iter_boxes(f, start=parent.span.payload_start, end=parent.span.end):
            if box.type == MDIA_ATOM_NAME:
                if not isinstance(box, MediaBox):
                    msg = "Invalid MediaBox: type not dispatched to correct class"
                    raise ValueError(msg)

                mdia_box = box
                break

        if mdia_box is None:
            msg = "Missing mdia box in trak"
            raise RuntimeError(msg)

        for box in iter_boxes(
            f, start=mdia_box.span.payload_start, end=mdia_box.span.end
        ):
            if box.type == HDLR_ATOM_NAME:
                if not isinstance(box, HandlerBox):
                    msg = "Invalid HandlerBox: type not dispatched to correct class"
                    raise ValueError(msg)

                return TrackBox(parent, box)

        msg = "Missing hdlr box in trak"
        raise RuntimeError(msg)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "TrackBox":
        box = MP4Box.read_from_stream(f, offset)
        return TrackBox.__read_from_stream_impl(f, box)


# NOTE: to make all of them work, we need to define all of those and determine the header size, so that the payload start address is aligned!, otherwise some things might fail!
# but we don't need all of those, only teh ones, that are needed for finding the language
CONTAINER_BOXES: set[ISOAtomName] = {
    MOOV_ATOM_NAME,
    TRAK_ATOM_NAME,
    MDIA_ATOM_NAME,
    # MINF_ATOM_NAME,
    # STBL_ATOM_NAME,
    # EDTS_ATOM_NAME,
    # DINF_ATOM_NAME,
    # UDTA_ATOM_NAME,
    # META_ATOM_NAME,
    # MOOF_ATOM_NAME,
    # TRAF_ATOM_NAME,
    # MFRA_ATOM_NAME,
}


def read_box_from_stream(f: BufferedIOBase, pos: int) -> MP4Box:
    box = MP4Box.read_from_stream(f, pos)

    match box.type:
        case b"mdhd":
            return MediaHeaderBox.read_from_stream(f, pos)
        case b"mdia":
            return MediaBox.read_from_stream(f, pos)
        case b"hdlr":
            return HandlerBox.read_from_stream(f, pos)
        case b"trak":
            return TrackBox.read_from_stream(f, pos)
        case _:
            return box


def iter_boxes(f: BufferedIOBase, start: int, end: int) -> Generator[MP4Box]:
    pos = start

    while pos < end:
        box = read_box_from_stream(f, pos)

        if pos + box.span.size > end:
            msg = f"Box {box.type!r} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield box
        pos += box.span.size


def find_audio_mdhd_boxes(f: BufferedIOBase) -> Generator["MediaHeaderBox"]:
    f.seek(0, 2)
    filesize = f.tell()

    stack: list[tuple[int, int, list[ISOAtomName]]] = [(0, filesize, [])]

    while stack:
        start, end, path = stack.pop()

        for box in iter_boxes(f, start, end):

            if box.type == TRAK_ATOM_NAME:
                if not isinstance(box, TrackBox):
                    msg = "Invalid TrackBox: type not dispatched to correct class"
                    raise ValueError(msg)

                hdlr = box.hdlr.handler_type

                if hdlr != SOUN_ATOM_NAME:
                    continue

            if box.type == MDHD_ATOM_NAME:
                if not isinstance(box, MediaHeaderBox):
                    msg = "Invalid MediaHeaderBox: type not dispatched to correct class"
                    raise ValueError(msg)

                current = path
                if current != [
                    MOOV_ATOM_NAME,
                    TRAK_ATOM_NAME,
                    MDIA_ATOM_NAME,
                ]:
                    msg = f"invalid mdhd box hierarchy: {current}"
                    raise RuntimeError(msg)

                yield box

            if box.type in CONTAINER_BOXES:
                stack.append((box.span.payload_start, box.span.end, [*path, box.type]))


def list_languages(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []

    with path.open("rb") as f:

        for i, mdhd in enumerate(find_audio_mdhd_boxes(f), start=1):
            lang = mdhd.read_language(f)
            result.append((i, lang))

    return result


def patch_languages(path: Path, new_language: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []

    with path.open("rüüüb+") as f:
        for i, mdhd in enumerate(find_audio_mdhd_boxes(f), start=1):
            old = mdhd.read_language(f)
            mdhd.patch_language(f, new_language)
            result.append((i, old))

    return result


if __name__ == "__main__":
    mp4 = Path("test.mp4")

    languages = list_languages(mp4)
    print("prev", languages)

    # languages = patch_languages(mp4, "ger")
    # print("old", languages)

    languages = list_languages(mp4)
    print("new", languages)
