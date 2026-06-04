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


class MP4Box:
    type: ISOAtomName
    start: int
    size: int
    header_size: int

    def __init__(
        self: Self,
        typ: ISOAtomName,
        start: int,
        size: int,
        header_size: int,
    ) -> None:
        self.type = typ
        self.start = start
        self.size = size
        self.header_size = header_size

        if self.size < 8:
            msg = f"Invalid box: sitze too small: {self.size}"
            raise RuntimeError(msg)

        if self.size < self.header_size:
            msg = f"Invalid box size {self.size} for {self.type!r} at {self.start}"
            raise RuntimeError(msg)

        if self.type == CMOV_ATOM_NAME:
            msg = f"Compressed movie box '{CMOV_ATOM_NAME}' not supported"
            raise RuntimeError(msg)

        if self.type == MOOF_ATOM_NAME:
            msg = f"Fragmented MP4 '{MOOF_ATOM_NAME}' not supported"
            raise RuntimeError(msg)

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def payload_start(self) -> int:
        return self.start + self.header_size

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MP4Box":
        # spec: ISO/IEC 14496-12
        # MP4 atom / ISO box structure:
        # size | 4 bytes | unsigned integer
        # type | 4 bytes | char[4]
        # ..., extension, dependent on size

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

        size, typ = Unpacker.unpack_default_sized("I4s", hdr, size=2)

        if typ == UUID_ATOM_NAME:
            msg = "'uuid' type not implemented, the box size differs"
            raise RuntimeError(msg)

        if size == 1:
            ext = read_checked(f, 8)

            largesize = Unpacker.unpack_default_one("Q", ext)

            if largesize < 16:
                msg = f"Invalid extended box size {largesize}"
                raise RuntimeError(msg)

            return MP4Box(typ, offset, largesize, header_size=16)

        if size == 0:
            f.seek(0, 2)
            eof = f.tell()
            return MP4Box(typ, offset, eof - offset, 8)

        return MP4Box(typ, offset, size, header_size=8)

    @staticmethod
    def iter_boxes(f: BufferedIOBase, start: int, end: int) -> Generator["MP4Box"]:
        pos = start

        while pos < end:
            box = MP4Box.read_from_stream(f, pos)

            if pos + box.size > end:
                msg = f"Box {box.type!r} at {pos} extends past parent boundary"
                raise RuntimeError(msg)

            yield box
            pos += box.size


class MP4FullBox(MP4Box):
    version: int
    flags: bytes

    def __init__(self: Self, parent: MP4Box, version: int, flags: bytes) -> None:
        super().__init__(parent.type, parent.start, parent.size, parent.header_size)
        self.version = version
        self.flags = flags

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "MP4FullBox":
        # spec: ISO/IEC 14496-12
        # ISO full box structure:
        # box     | header_size bytes | parent box
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

        f.seek(parent.payload_start)

        version = read_checked(f, 1)[0]

        flags = read_checked(f, 3)

        return MP4FullBox(parent, version, flags)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MP4FullBox":
        box = MP4Box.read_from_stream(f, offset)
        return MP4FullBox.__read_from_stream_impl(f, box)


CONTAINER_BOXES: set[ISOAtomName] = {
    MOOV_ATOM_NAME,
    TRAK_ATOM_NAME,
    MDIA_ATOM_NAME,
    MINF_ATOM_NAME,
    STBL_ATOM_NAME,
    EDTS_ATOM_NAME,
    DINF_ATOM_NAME,
    UDTA_ATOM_NAME,
    META_ATOM_NAME,
    MOOF_ATOM_NAME,
    TRAF_ATOM_NAME,
    MFRA_ATOM_NAME,
}


class MHDBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.start, parent.size, parent.header_size)

    @staticmethod
    def get_from_normal_box(box: MP4Box) -> "MHDBox":
        if box.type != MDHD_ATOM_NAME:
            msg = f"Invalid MHDBox box with type: {box.type!s}"
            raise RuntimeError(msg)
        return MHDBox(box)

    @staticmethod
    def find_audio_mdhd_boxes(f: BufferedIOBase) -> Generator["MHDBox"]:
        f.seek(0, 2)
        filesize = f.tell()

        stack: list[tuple[int, int, list[ISOAtomName]]] = [(0, filesize, [])]

        while stack:
            start, end, path = stack.pop()

            for box in MP4Box.iter_boxes(f, start, end):

                if box.type == TRAK_ATOM_NAME:
                    trak_box: TrakBox = TrakBox.get_from_normal_box(box)

                    hdlr = trak_box.get_hdlr_type(f)

                    if hdlr is None:
                        msg = "Missing hdlr box in trak"
                        raise RuntimeError(msg)

                    if hdlr != SOUN_ATOM_NAME:
                        continue

                if box.type == MDHD_ATOM_NAME:
                    current = path
                    if current != [
                        MOOV_ATOM_NAME,
                        TRAK_ATOM_NAME,
                        MDIA_ATOM_NAME,
                    ]:
                        msg = f"invalid mdhd box hierarchy: {current}"
                        raise RuntimeError(msg)

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

        # TODO: full box!
        version = read_checked(f, 1)[0]

        # TODO: define the structs for the size!
        MIN_MDHD_V0 = 24
        MIN_MDHD_V1 = 36

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
        packed = Unpacker.unpack_default_one("H", lang_bytes)

        return language_offset, MHDBox.decode_language(packed)

    def patch_mdhd_language(self: Self, f: BufferedIOBase, new_language: str) -> str:
        offset, old_language = self.read_mdhd_language(f)

        packed = MHDBox.encode_language(new_language)

        f.seek(offset)

        packed_bytes = struct.pack(">H", packed)
        if len(packed_bytes) != 2:
            msg = "packed bytes are not of correct size"
            raise RuntimeError(msg)

        f.write(packed_bytes)
        f.flush()

        f.seek(offset)
        verify_bytes = read_checked(f, 2)
        verify = Unpacker.unpack_default_one("H", verify_bytes)

        if verify != packed:
            msg = "Invalid overwrite"
            raise RuntimeError(msg)

        return old_language


class TrakBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.start, parent.size, parent.header_size)

    @staticmethod
    def get_from_normal_box(box: MP4Box) -> "TrakBox":
        if box.type != TRAK_ATOM_NAME:
            msg = f"Invalid TrakBox box with type: {box.type!s}"
            raise RuntimeError(msg)
        return TrakBox(box)

    def get_hdlr_type(self: Self, f: BufferedIOBase) -> Optional[bytes]:
        """
        Returns handler type like b'soun', b'vide', etc.
        """
        mdia_start = None

        for box in MP4Box.iter_boxes(f, self.payload_start, end=self.end):
            if box.type == MDIA_ATOM_NAME:
                mdia_start = box
                break

        if not mdia_start:
            return None

        for box in MP4Box.iter_boxes(f, mdia_start.payload_start, mdia_start.end):
            if box.type == HDLR_ATOM_NAME:
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

    languages = list_languages(mp4)
    print("prev", languages)

    languages = patch_languages(mp4, "ger")
    print("old", languages)

    languages = list_languages(mp4)
    print("new", languages)
