import json
from collections.abc import Generator
from contextlib import AbstractContextManager
from enum import Enum
from io import BufferedIOBase, BytesIO
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, Self, final, override
from uuid import UUID

from content.language import ShortLanguageStr
from content.tagger.parser import (
    ByteOrder,
    Packable,
    Packer,
    Unpacker,
    UnsignedInt,
    UnsignedLongLong,
    UnsignedShort,
    read_checked,
    uuid_from_bytes,
    uuid_to_bytes,
)
from content.tagger.video_tagger import (
    VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
    MetadataTags,
    MetadataTagsRead,
    SerializableDict,
    VideoTagger,
    VideoTaggerWriter,
)
from helper.manager import CounterInterface, ManagerInterface
from helper.result import Err, Ok, Result
from helper.translation import get_translator

_ = get_translator()


class ISOMAtomName:
    __value: bytes

    def __init__(self: Self, value: bytes) -> None:
        self.__value = value

        if len(value) != 4:
            msg = f"Invalid atom name length {len(value)}"
            raise ValueError(msg)

        def is_valid_byte(byte: int, pos: int) -> bool:
            val = bytes([byte])

            if val.islower():
                return True

            if val.isupper():
                return True

            if val.isupper():
                return True

            if pos != 0 and val.isdigit():
                return True

            return val in [b"\xa9", b"-"]

        if not all(is_valid_byte(val, i) for i, val in enumerate(value)):
            msg = f"Atom name not valid {value!r}"
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
        if isinstance(other, ISOMAtomName):
            return self.__value == other.__value

        if isinstance(other, str):
            return self.__value == other.encode()

        if isinstance(other, bytes):
            return self.__value == other

        return False


class PackableISOMAtomName(Packable[bytes]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "4s"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 4


CMOV_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"cmov")
MOOF_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"moof")
MOOV_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"moov")
UUID_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"uuid")
TRAK_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"trak")
MDIA_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mdia")
MINF_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"minf")
STBL_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"stbl")
EDTS_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"edts")
DINF_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"dinf")
UDTA_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"udta")
META_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"meta")
TRAF_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"traf")
MFRA_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mfra")
MDHD_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mdhd")
SOUN_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"soun")
HDLR_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"hdlr")
VIDE_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"vide")
FTYP_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"ftyp")
FREE_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"free")
SKIP_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"skip")
ILST_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"ilst")
DATA_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"data")
MVHD_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mvhd")
IODS_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"iods")
TKHD_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"tkhd")
MDAT_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mdat")


class MP4BoxSpan:
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

    def __str__(self: Self) -> str:
        return f"<MP4BoxSpan start: {self.start} size: {self.size} header: [0, {self.header_size}] payload: [{self.payload_start}, {self.payload_size}]>"

    def __repr__(self: Self) -> str:
        return str(self)


ISOM_BYTE_ORDER = ByteOrder.Big


class MP4Box:
    type: ISOMAtomName
    span: MP4BoxSpan
    is_container: bool

    def __init__(
        self: Self,
        typ: ISOMAtomName,
        span: MP4BoxSpan,
        *,
        is_container: bool,
    ) -> None:
        self.type = typ
        self.span = span
        self.is_container = is_container

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

        size, typ_raw = Unpacker.unpack_two(
            ISOM_BYTE_ORDER,
            (UnsignedInt(), PackableISOMAtomName()),
            hdr,
        )

        typ = ISOMAtomName(typ_raw)

        if typ == CMOV_ATOM_NAME:
            msg = f"Compressed movie box '{CMOV_ATOM_NAME}' not supported"
            raise RuntimeError(msg)

        if typ == MOOF_ATOM_NAME:
            msg = f"Fragmented MP4 '{MOOF_ATOM_NAME}' not supported"
            raise RuntimeError(msg)

        final_size: int = size
        header_size: int = 8

        if size == 1:
            ext = read_checked(f, 8)

            largesize = Unpacker.unpack_one(
                ISOM_BYTE_ORDER,
                UnsignedLongLong(),
                ext,
            )

            if largesize < 16:
                msg = f"Invalid extended box size {largesize}"
                raise RuntimeError(msg)

            final_size = largesize
            header_size = 16
        elif size == 0:
            f.seek(0, 2)
            eof = f.tell()
            final_size = eof - offset
            header_size = 8

        if typ == UUID_ATOM_NAME:
            usertype_raw = read_checked(f, 16)

            usertype = uuid_from_bytes(ISOM_BYTE_ORDER, usertype_raw)

            header_size = header_size + 16

            span = MP4BoxSpan(offset, size=final_size, header_size=header_size)
            box = MP4Box(typ, span, is_container=False)
            user_box = UserExtensionBox(box, usertype, is_container=False)
            return user_extension_box_determine_correct_extension(f, user_box)

        span = MP4BoxSpan(offset, size=final_size, header_size=header_size)
        return MP4Box(typ, span, is_container=False)

    @staticmethod
    def __impl_write_to_buffer_mp4_box(typ: ISOMAtomName, data: bytes) -> bytes:
        buf = BytesIO()

        final_size: int = 4 + 4 + len(data)
        additional_size: Optional[int] = None

        if final_size > ((2**32) - 1):
            additional_size = final_size + 8
            final_size = 1

        hdr = Packer.pack_two(
            ISOM_BYTE_ORDER,
            (UnsignedInt(), PackableISOMAtomName()),
            (
                final_size,
                typ.value,
            ),
            8,
        )

        buf.write(hdr)

        if additional_size is not None:
            largsize = Packer.pack_one(
                ISOM_BYTE_ORDER,
                UnsignedLongLong(),
                additional_size,
                8,
            )

            buf.write(largsize)

        buf.write(data)

        return buf.getvalue()

    @staticmethod
    def write_to_buffer_mp4_box(
        typ: ISOMAtomName,
        data: bytes,
        *,
        usertype: Optional[UUID] = None,
    ) -> bytes:

        if typ == UUID_ATOM_NAME:
            if usertype is None:
                msg = f"usertype HAS TO BE given, if we have th UUID atom name: {usertype}"
                raise RuntimeError(msg)

            uuid_bytes = uuid_to_bytes(ISOM_BYTE_ORDER, usertype)

            return MP4Box.__impl_write_to_buffer_mp4_box(typ, uuid_bytes + data)

        if usertype is not None:
            msg = f"usertype can only be given, if we have th UUID atom name: {typ}"
            raise RuntimeError(msg)

        return MP4Box.__impl_write_to_buffer_mp4_box(typ, data)

    def __str__(self: Self) -> str:
        return f"<MP4Box type: {self.type} span: {self.span} is_container: {self.is_container}>"

    def __repr__(self: Self) -> str:
        return str(self)


class UserExtensionBox(MP4Box):
    usertype: UUID

    def __init__(
        self: Self,
        parent: MP4Box,
        usertype: UUID,
        *,
        is_container: bool,
    ) -> None:
        super().__init__(parent.type, parent.span, is_container=is_container)

        self.usertype = usertype

    @staticmethod
    def write_to_buffer_uuid_box(uuid: UUID, data: bytes) -> bytes:

        return MP4Box.write_to_buffer_mp4_box(UUID_ATOM_NAME, data, usertype=uuid)

    def __str__(self: Self) -> str:
        return f"<UserExtensionBox parent: {MP4Box.__str__(self)} usertype: {self.usertype.hex}>"

    def __repr__(self: Self) -> str:
        return str(self)


UUIDExtension_UUID = UUID("90e175d1-efdb-4144-a214-ebfab6258ae7")
JSONExtension_UUID = UUID("90e175d1-efdb-4144-a214-ebfab6258ae8")


class UserExtensions:
    UUIDExtension_UUID = UUIDExtension_UUID
    JSONExtension_UUID = JSONExtension_UUID


@final
class UUIDExtensionBox(UserExtensionBox):
    uuid: UUID

    def __init__(
        self: Self,
        parent: UserExtensionBox,
        uuid: UUID,
    ) -> None:
        super().__init__(parent, parent.usertype, is_container=False)

        self.uuid = uuid

    @staticmethod
    def read_from_stream_parent(
        f: BufferedIOBase,
        parent: UserExtensionBox,
    ) -> "UUIDExtensionBox":

        # this is a custom user box, it contains one UUID

        f.seek(parent.span.payload_start)

        if parent.span.payload_size != 16:
            msg = f"UUIDExtensionBox has not the correct payload size: {parent.span.payload_size}"
            raise RuntimeError(msg)

        uuid_raw = read_checked(f, 16)

        uuid = uuid_from_bytes(ISOM_BYTE_ORDER, uuid_raw)

        parent.span.add_header_size(16)

        return UUIDExtensionBox(parent, uuid)

    @staticmethod
    def write_to_buffer(uuid: UUID) -> bytes:

        data: bytes = uuid_to_bytes(ISOM_BYTE_ORDER, uuid)

        return UserExtensionBox.write_to_buffer_uuid_box(UUIDExtension_UUID, data)

    def __str__(self: Self) -> str:
        return f"<UUIDExtensionBox parent: {UserExtensionBox.__str__(self)} uuid: {self.uuid.hex}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class JsonExtensionBox(UserExtensionBox):
    data: SerializableDict

    def __init__(
        self: Self,
        parent: UserExtensionBox,
        data: SerializableDict,
    ) -> None:
        super().__init__(parent, parent.usertype, is_container=False)

        self.data = data

    @staticmethod
    def read_from_stream_parent(
        f: BufferedIOBase,
        parent: UserExtensionBox,
    ) -> "JsonExtensionBox":

        # this is a custom user box, it contains a json payload

        f.seek(parent.span.payload_start)

        data_raw = read_checked(f, parent.span.payload_size)

        data = json.loads(data_raw.decode())

        parent.span.add_header_size(16)

        return JsonExtensionBox(parent, data)

    @staticmethod
    def write_to_buffer(data: SerializableDict) -> bytes:

        byte_data: bytes = json.dumps(data).encode()

        return UserExtensionBox.write_to_buffer_uuid_box(JSONExtension_UUID, byte_data)

    def __str__(self: Self) -> str:
        return f"<JsonExtensionBox parent: {UserExtensionBox.__str__(self)} data: {self.data!r}>"

    def __repr__(self: Self) -> str:
        return str(self)


def user_extension_box_determine_correct_extension(
    f: BufferedIOBase,
    box: UserExtensionBox,
) -> UserExtensionBox:
    match box.usertype:
        case UserExtensions.UUIDExtension_UUID:
            return UUIDExtensionBox.read_from_stream_parent(f, box)
        case UserExtensions.JSONExtension_UUID:
            return JsonExtensionBox.read_from_stream_parent(f, box)
        case _:
            return box


class MP4FullBox(MP4Box):
    version: int
    flags: bytes

    def __init__(
        self: Self,
        parent: MP4Box,
        version: int,
        flags: bytes,
        *,
        is_container: bool,
    ) -> None:
        super().__init__(parent.type, parent.span, is_container=is_container)

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

        return MP4FullBox(parent, version, flags, is_container=False)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MP4FullBox":
        box = MP4Box.read_from_stream(f, offset)
        return MP4FullBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent_mp4_full_box(
        f: BufferedIOBase, parent: MP4Box
    ) -> "MP4FullBox":
        return MP4FullBox.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<MP4FullBox parent: {MP4Box.__str__(self)} version: {self.version} flags: {self.flags.hex()}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class FileTypeBox(MP4Box):
    major_brand: ISOMAtomName
    minor_version: int
    compatible_brands: bytes

    def __init__(
        self: Self,
        parent: MP4Box,
        major_brand: ISOMAtomName,
        minor_version: int,
        compatible_brands: bytes,
    ) -> None:
        super().__init__(parent.type, parent.span, is_container=False)

        self.major_brand = major_brand
        self.minor_version = minor_version
        self.compatible_brands = compatible_brands

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "FileTypeBox":
        # spec: ISO/IEC 14496-12
        # ISO file type structure:
        # box     | <box size> bytes | parent box
        # major_brand   | 4 bytes | char[4]
        # minor_version   | 4 bytes | char[4]
        # .. string data for the rest of the size

        # aligned(8) class FileTypeBox extends Box(
        #     'ftyp'
        # ) {
        #     unsigned int(32) major_brand;
        #     unsigned int(32) minor_version;
        #     unsigned int(32) compatible_brands[];
        # }

        f.seek(parent.span.payload_start)

        major_brand_raw = read_checked(f, 4)
        major_brand = ISOMAtomName(major_brand_raw)

        minor_version_bytes = read_checked(f, 4)

        minor_version = Unpacker.unpack_one(
            ISOM_BYTE_ORDER,
            UnsignedInt(),
            minor_version_bytes,
        )

        compatible_brands_size = parent.span.payload_size - (4 + 4)

        if compatible_brands_size < 0:
            msg = f"Invalid box size: not enough data for complete FileTypeBox: have {parent.span.payload_size} but need at least {(4 + 4)}"
            raise RuntimeError(msg)

        compatible_brands = read_checked(f, compatible_brands_size)

        additional_header_size = 4 + 4 + compatible_brands_size

        parent.span.add_header_size(additional_header_size)

        return FileTypeBox(parent, major_brand, minor_version, compatible_brands)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "FileTypeBox":
        box = MP4Box.read_from_stream(f, offset)
        return FileTypeBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "FileTypeBox":
        return FileTypeBox.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<FileTypeBox parent: {MP4Box.__str__(self)} major_brand: {self.major_brand} minor_version: {self.minor_version} compatible_brands: {self.compatible_brands!s}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class FreeSpaceBox(MP4Box):
    data: bytes

    def __init__(
        self: Self,
        parent: MP4Box,
        data: bytes,
    ) -> None:
        super().__init__(parent.type, parent.span, is_container=False)

        self.data = data

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "FreeSpaceBox":
        # spec: ISO/IEC 14496-12
        # ISO free space structure:
        # box     | <box size> bytes | parent box
        # .. string data for the rest of the size

        # free_type may be 'free' or 'skip'.
        # aligned(8) class FreeSpaceBox extends Box(
        #     free_type
        # ) {
        #     unsigned int(8) data[];
        # }

        f.seek(parent.span.payload_start)

        data = read_checked(f, parent.span.payload_size)

        parent.span.add_header_size(parent.span.payload_size)

        return FreeSpaceBox(parent, data)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "FreeSpaceBox":
        box = MP4Box.read_from_stream(f, offset)
        return FreeSpaceBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "FreeSpaceBox":
        return FreeSpaceBox.__read_from_stream_impl(f, parent)

    @staticmethod
    def write_to_buffer(data: bytes) -> bytes:

        return MP4Box.write_to_buffer_mp4_box(FREE_ATOM_NAME, data)

    def __str__(self: Self) -> str:
        return f"<FreeSpaceBox parent: {MP4Box.__str__(self)} data: {self.data!s}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class MediaHeaderBox(MP4FullBox):
    language_offset: int

    def __init__(self: Self, parent: MP4FullBox, language_offset: int) -> None:
        super().__init__(
            parent,
            parent.version,
            parent.flags,
            is_container=False,
        )
        self.language_offset = language_offset

    @staticmethod
    def __read_from_stream_impl(
        f: BufferedIOBase,
        parent: MP4FullBox,
    ) -> "MediaHeaderBox":
        # spec: ISO/IEC 14496-12
        # ISO media header box structure:
        # full_box     | <full box size> bytes | parent full box
        # ... data, dependent on version

        # aligned(8) class MediaHeaderBox
        # extends FullBox(
        #     'mdhd',
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
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "MediaHeaderBox":
        box = MP4FullBox.read_from_stream_parent_mp4_full_box(f, parent)
        return MediaHeaderBox.__read_from_stream_impl(f, box)

    @staticmethod
    def __decode_language_impl(value: int) -> ShortLanguageStr | str:
        chars = []

        for shift in (10, 5, 0):
            v = (value >> shift) & 0x1F

            if v < 1 or v > 26:
                msg = f"Invalid ISO639 character value {v}"
                raise RuntimeError(msg)

            chars.append(chr(v + 0x60))

        val = "".join(chars)
        short_str = ShortLanguageStr.from_str(val)
        if short_str is not None:
            return short_str

        return val

    @staticmethod
    def __encode_language_impl(lang: ShortLanguageStr) -> int:
        code = str(lang.to_alpha3())

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

    def read_language(self: Self, f: BufferedIOBase) -> ShortLanguageStr | str:
        f.seek(self.span.start + self.language_offset)

        lang_bytes = read_checked(f, 2)
        packed = Unpacker.unpack_one(ISOM_BYTE_ORDER, UnsignedShort(), lang_bytes)

        return MediaHeaderBox.__decode_language_impl(packed)

    def patch_language(
        self: Self,
        f: BufferedIOBase,
        new_language: ShortLanguageStr,
    ) -> None:
        packed = MediaHeaderBox.__encode_language_impl(new_language)

        f.seek(self.span.start + self.language_offset)

        packed_bytes = Packer.pack_one(ISOM_BYTE_ORDER, UnsignedShort(), packed, 2)

        f.write(packed_bytes)
        f.flush()

        f.seek(self.span.start + self.language_offset)
        verify_bytes = read_checked(f, 2)
        verify = Unpacker.unpack_one(ISOM_BYTE_ORDER, UnsignedShort(), verify_bytes)

        if verify != packed:
            msg = "Invalid overwrite"
            raise RuntimeError(msg)

    def __str__(self: Self) -> str:
        return f"<MediaHeaderBox parent: {MP4FullBox.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class MediaBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "MediaBox":
        # spec: ISO/IEC 14496-12
        # ISO media box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class MediaBox extends Box(
        #     'mdia'
        #     ) {
        # }

        return MediaBox(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MediaBox":
        box = MP4Box.read_from_stream(f, offset)
        return MediaBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "MediaBox":
        return MediaBox.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<MediaBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class MovieBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "MovieBox":
        # spec: ISO/IEC 14496-12
        # ISO movie box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class MovieBox extends Box(
        #     'moov'
        #     ){
        # }

        return MovieBox(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MovieBox":
        box = MP4Box.read_from_stream(f, offset)
        return MovieBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "MovieBox":
        return MovieBox.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<MovieBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class HandlerBox(MP4FullBox):
    handler_type: ISOMAtomName
    name: str

    def __init__(
        self: Self, parent: MP4FullBox, handler_type: ISOMAtomName, name: str
    ) -> None:
        super().__init__(
            parent,
            parent.version,
            parent.flags,
            is_container=False,
        )
        self.handler_type = handler_type
        self.name = name

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4FullBox) -> "HandlerBox":
        # spec: ISO/IEC 14496-12
        # ISO handler box structure:
        # box     | <full box size> bytes | parent full box
        # ... data, see below

        # aligned(8) class HandlerBox extends FullBox(
        #     'hdlr',
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

        pre_defined = read_checked(f, 4)

        if pre_defined != b"\x00" * 4:
            msg = f"HandlerBox: pre_defined has to be 0, but was: {pre_defined!r}"
            raise ValueError(msg)

        handler_type_raw = read_checked(f, 4)
        handler_type = ISOMAtomName(handler_type_raw)

        reserved = read_checked(f, 4 * 3)

        if reserved != b"\x00" * (3 * 4):
            # make exception for apples usage of these types
            if reserved.startswith(b"appl"):
                pass
            else:
                msg = f"HandlerBox: reserved has to be 0, but was: {reserved!r}"
                raise ValueError(msg)

        # omitting dynamic sized string "name"
        fixed_header_size = 4 + 4 + (4 * 3)

        name_size = parent.span.payload_size - fixed_header_size

        name_raw = read_checked(f, name_size)

        name = name_raw.decode()

        parent.span.add_header_size(parent.span.payload_size)

        return HandlerBox(parent, handler_type, name)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "HandlerBox":
        box = MP4FullBox.read_from_stream(f, offset)
        return HandlerBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "HandlerBox":
        box = MP4FullBox.read_from_stream_parent_mp4_full_box(f, parent)
        return HandlerBox.__read_from_stream_impl(f, box)

    def __str__(self: Self) -> str:
        return f"<HandlerBox parent: {MP4FullBox.__str__(self)} handler_type: {self.handler_type} name: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class TrackBox(MP4Box):
    hdlr: HandlerBox

    def __init__(self: Self, parent: MP4Box, hdlr: HandlerBox) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

        self.hdlr = hdlr

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "TrackBox":
        # spec: ISO/IEC 14496-12
        # ISO track box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class TrackBox extends Box(
        #     'trak'
        # ) {
        # }

        mdia_box: Optional[MediaBox] = None

        for box in mp4_iter_boxes(
            f,
            start=parent.span.payload_start,
            end=parent.span.end,
        ):
            if box.type == MDIA_ATOM_NAME:
                if not isinstance(box, MediaBox):
                    msg = "Invalid MediaBox: type not dispatched to correct class"
                    raise TypeError(msg)

                mdia_box = box
                break

        if mdia_box is None:
            msg = "Missing mdia box in trak"
            raise RuntimeError(msg)

        for box in mp4_iter_boxes(
            f,
            start=mdia_box.span.payload_start,
            end=mdia_box.span.end,
        ):
            if box.type == HDLR_ATOM_NAME:
                if not isinstance(box, HandlerBox):
                    msg = "Invalid HandlerBox: type not dispatched to correct class"
                    raise TypeError(msg)

                return TrackBox(parent, box)

        msg = "Missing hdlr box in trak"
        raise RuntimeError(msg)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "TrackBox":
        box = MP4Box.read_from_stream(f, offset)
        return TrackBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "TrackBox":
        return TrackBox.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<TrackBox parent: {MP4Box.__str__(self)} hdlr: {self.hdlr}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class UserDataBox(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_from_stream_impl(f: BufferedIOBase, parent: MP4Box) -> "UserDataBox":
        # spec: ISO/IEC 14496-12
        # ISO user data box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class UserDataBox extends Box(
        #     'udta'
        #     ) {
        # }

        return UserDataBox(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "UserDataBox":
        box = MP4Box.read_from_stream(f, offset)
        return UserDataBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "UserDataBox":
        return UserDataBox.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<UserDataBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class MetaBox(MP4FullBox):
    handler_box: HandlerBox

    def __init__(
        self: Self, parent: MP4FullBox, handler_box: HandlerBox, *, is_container: bool
    ) -> None:
        super().__init__(
            parent,
            parent.version,
            parent.flags,
            is_container=is_container,
        )

        self.handler_box = handler_box

    @staticmethod
    def __read_from_stream_impl(
        f: BufferedIOBase,
        parent: MP4FullBox,
    ) -> "MetaBox":
        # spec: ISO/IEC 14496-12
        # ISO meta box structure:
        # full_box     | <full box size> bytes | parent full box
        # ... data

        # aligned(8) class MetaBox (handler_type)
        # extends FullBox(
        #     'meta',
        #     version = 0,
        #     0)
        # {
        #     HandlerBox(handler_type) theHandler;
        #     PrimaryItemBox primary_resource; // optional
        #     DataInformationBox file_locations; // optional
        #     ItemLocationBox item_locations; // optional
        #     ItemProtectionBox protections; // optional
        #     ItemInfoBox item_infos; // optional
        #     IPMPControlBox IPMP_control; // optional
        #     ItemReferenceBox item_refs; // optional
        #     ItemDataBox item_data; // optional
        #     Box other_boxes[]; // optional
        # }

        f.seek(parent.span.payload_start)

        if parent.version != 0:
            msg = "Invalid meta version"
            raise RuntimeError(msg)

        handler_box = HandlerBox.read_from_stream(f, parent.span.payload_start)

        parent.span.add_header_size(handler_box.span.size)

        # peek the next box, if it's an optional box, we read that, otherwise we are at the end and read the last box array
        OPTIONAL_BOXES: list[ISOMAtomName] = [
            ISOMAtomName(b"pitm"),  # PrimaryItemBox
            ISOMAtomName(b"dinf"),  # DataInformationBox
            ISOMAtomName(b"iloc"),  # ItemLocationBox
            ISOMAtomName(b"ipro"),  # ItemProtectionBox
            ISOMAtomName(b"iinf"),  # ItemInfoBox
            ISOMAtomName(b"ipmc"),  # IPMPControlBox
            ISOMAtomName(b"iref"),  # ItemReferenceBox
            ISOMAtomName(b"idat"),  # ItemDataBox
        ]
        while True:
            f.seek(parent.span.payload_start)

            simple_box = MP4Box.read_from_stream(f, parent.span.payload_start)

            if simple_box.type not in OPTIONAL_BOXES:
                break

            msg = f"MetaBox: parsing of optional box {simple_box.type} not implemented yet"
            raise RuntimeError(msg)

        # treat the box as container, if there is some payload left
        is_container = parent.span.payload_size != 0

        return MetaBox(parent, handler_box, is_container=is_container)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MetaBox":
        box = MP4FullBox.read_from_stream(f, offset)
        return MetaBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(f: BufferedIOBase, parent: MP4Box) -> "MetaBox":
        box = MP4FullBox.read_from_stream_parent_mp4_full_box(f, parent)
        return MetaBox.__read_from_stream_impl(f, box)

    def __str__(self: Self) -> str:
        return f"<MetaBox parent: {MP4FullBox.__str__(self)} handler_box: {self.handler_box}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class AppleItunesItemList(MP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_from_stream_impl(
        f: BufferedIOBase,
        parent: MP4Box,
    ) -> "AppleItunesItemList":
        # spec: https://developer.apple.com/documentation/quicktime-file-format/metadata_item_list_atom
        # Apple Itunes Item List structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class AppleItunesItemList extends Box(
        #     'ilst'
        #     ) {
        # }

        return AppleItunesItemList(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "AppleItunesItemList":
        box = MP4Box.read_from_stream(f, offset)
        return AppleItunesItemList.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(
        f: BufferedIOBase, parent: MP4Box
    ) -> "AppleItunesItemList":
        return AppleItunesItemList.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemList parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


AppleItunesItemDataContent = str | int | UUID


# see: https://developer.apple.com/documentation/quicktime-file-format/well-known_types
class AppleItunesItemDataType(Enum):
    # NOTE: only some are implemented here
    reserved = 0
    utf_8 = 1
    utf_16 = 2

    # see https://taglib.org/api/namespaceTagLib_1_1MP4.html#a86c3870b24b4cdb2887d3e47f5f9a39b
    uuid = 8

    jpeg = 13
    png = 14

    be_signed_integer_var = 21
    be_unsigned_integer_var = 22


@final
class AppleItunesItemDataBox(MP4FullBox):
    type_indicator: int
    locale_indicator: int
    value: AppleItunesItemDataContent

    def __init__(
        self: Self,
        parent: MP4FullBox,
        type_indicator: int,
        locale_indicator: int,
        value: AppleItunesItemDataContent,
    ) -> None:
        super().__init__(parent, parent.version, parent.flags, is_container=False)

        self.type_indicator = type_indicator
        self.locale_indicator = locale_indicator
        self.value = value

    @staticmethod
    def __decode_value(
        type_indicator: int,
        value: bytes,
    ) -> AppleItunesItemDataContent:
        match type_indicator:
            case AppleItunesItemDataType.utf_8.value:
                return value.decode("utf-8")
            case AppleItunesItemDataType.utf_16.value:
                return value.decode("utf-8")
            case AppleItunesItemDataType.uuid.value:
                return UUID(bytes=value)
            case _:
                msg = f"Not implemented type_indicator conversion: {type_indicator}"
                raise RuntimeError(msg)

    @staticmethod
    def __read_from_stream_impl(
        f: BufferedIOBase,
        parent: MP4FullBox,
    ) -> "AppleItunesItemDataBox":
        # spec: https://developer.apple.com/documentation/quicktime-file-format/data_atom
        # Apple Itunes Item Box structure:
        # box     | <full box size> bytes | parent full box
        # ... data

        # aligned(8) class AppleItunesItemDataBox extends FullBox(
        #     'data'
        #     ) {
        # }

        f.seek(parent.span.payload_start)

        # see: https://developer.apple.com/documentation/quicktime-file-format/type_indicator
        if parent.version != 0:
            msg = f"AppleItunesItemDataBox: type indicator byte 0 has to be 0, but was {parent.version} (it is the FullBox version field)"
            raise ValueError(msg)

        type_indicator = Unpacker.unpack_one(
            ISOM_BYTE_ORDER,
            UnsignedInt(),
            b"\x00" + parent.flags,
        )

        locale_indicator_raw = read_checked(f, 4)

        # see: https://developer.apple.com/documentation/quicktime-file-format/locale_indicator
        locale_indicator = Unpacker.unpack_one(
            ISOM_BYTE_ORDER,
            UnsignedInt(),
            locale_indicator_raw,
        )

        # omitting dynamic sized string "value"
        fixed_header_size = 4

        value_size = parent.span.payload_size - fixed_header_size

        value_raw = read_checked(f, value_size)

        value = AppleItunesItemDataBox.__decode_value(type_indicator, value_raw)

        parent.span.add_header_size(parent.span.payload_size)

        return AppleItunesItemDataBox(parent, type_indicator, locale_indicator, value)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "AppleItunesItemDataBox":
        box: MP4FullBox = MP4FullBox.read_from_stream(f, offset)
        return AppleItunesItemDataBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(
        f: BufferedIOBase,
        parent: MP4Box,
    ) -> "AppleItunesItemDataBox":
        box: MP4FullBox = MP4FullBox.read_from_stream_parent_mp4_full_box(f, parent)
        return AppleItunesItemDataBox.__read_from_stream_impl(f, box)

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemDataBox parent: {MP4FullBox.__str__(self)} type_indicator: {self.type_indicator} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class AppleItunesItemBox(MP4Box):
    data: AppleItunesItemDataBox

    def __init__(self: Self, parent: MP4Box, data: AppleItunesItemDataBox) -> None:
        super().__init__(parent.type, parent.span, is_container=False)

        self.data = data

    @staticmethod
    def __read_from_stream_impl(
        f: BufferedIOBase,
        parent: MP4Box,
    ) -> "AppleItunesItemBox":
        # spec: https://developer.apple.com/documentation/quicktime-file-format/value_atom
        # Apple Itunes Item Box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class AppleItunesItemBox extends Box(
        #     '<any>' // any type specified, see below
        #     ) {
        # }

        f.seek(parent.span.payload_start)

        data = AppleItunesItemDataBox.read_from_stream(f, parent.span.payload_start)

        parent.span.add_header_size(data.span.size)

        return AppleItunesItemBox(parent, data)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "AppleItunesItemBox":
        box = MP4Box.read_from_stream(f, offset)
        return AppleItunesItemBox.__read_from_stream_impl(f, box)

    @staticmethod
    def read_from_stream_parent(
        f: BufferedIOBase,
        parent: MP4Box,
    ) -> "AppleItunesItemBox":
        return AppleItunesItemBox.__read_from_stream_impl(f, parent)

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemBox parent: {MP4Box.__str__(self)} data: {self.data}>"

    def __repr__(self: Self) -> str:
        return str(self)


AppleItunesItemBoxAtoms: list[ISOMAtomName] = [
    ISOMAtomName(b"----"),  # TODO: special case this!
    ISOMAtomName(b"trkn"),
    ISOMAtomName(b"disk"),
    ISOMAtomName(b"gnre"),
    ISOMAtomName(b"plID"),
    ISOMAtomName(b"cnID"),
    ISOMAtomName(b"geID"),
    ISOMAtomName(b"atID"),
    ISOMAtomName(b"sfID"),
    ISOMAtomName(b"cmID"),
    ISOMAtomName(b"akID"),
    ISOMAtomName(b"tvsn"),
    ISOMAtomName(b"tves"),
    ISOMAtomName(b"tmpo"),
    ISOMAtomName(b"\xa9mvi"),
    ISOMAtomName(b"\xa9mvc"),
    ISOMAtomName(b"cpil"),
    ISOMAtomName(b"pgap"),
    ISOMAtomName(b"pcst"),
    ISOMAtomName(b"shwm"),
    ISOMAtomName(b"stik"),
    ISOMAtomName(b"hdvd"),
    ISOMAtomName(b"rtng"),
    ISOMAtomName(b"covr"),
    ISOMAtomName(b"purl"),
    ISOMAtomName(b"egid"),
    ISOMAtomName(b"\xa9nam"),
    ISOMAtomName(b"\xa9alb"),
    ISOMAtomName(b"\xa9ART"),
    ISOMAtomName(b"aART"),
    ISOMAtomName(b"\xa9wrt"),
    ISOMAtomName(b"\xa9day"),
    ISOMAtomName(b"\xa9cmt"),
    ISOMAtomName(b"desc"),
    ISOMAtomName(b"purd"),
    ISOMAtomName(b"\xa9grp"),
    ISOMAtomName(b"\xa9gen"),
    ISOMAtomName(b"\xa9lyr"),
    ISOMAtomName(b"catg"),
    ISOMAtomName(b"keyw"),
    ISOMAtomName(b"\xa9too"),
    ISOMAtomName(b"cprt"),
    ISOMAtomName(b"soal"),
    ISOMAtomName(b"soaa"),
    ISOMAtomName(b"soar"),
    ISOMAtomName(b"sonm"),
    ISOMAtomName(b"soco"),
    ISOMAtomName(b"sosn"),
    ISOMAtomName(b"tvsh"),
]


class SupportedBoxes:
    MDHD = MDHD_ATOM_NAME
    MDIA = MDIA_ATOM_NAME
    HDLR = HDLR_ATOM_NAME
    TRAK = TRAK_ATOM_NAME
    MOOV = MOOV_ATOM_NAME
    FTYP = FTYP_ATOM_NAME
    FREE = FREE_ATOM_NAME
    SKIP = SKIP_ATOM_NAME
    UDTA = UDTA_ATOM_NAME
    META = META_ATOM_NAME

    ILST = ILST_ATOM_NAME

    AppleItunesItemBox = AppleItunesItemBoxAtoms

    DATA = DATA_ATOM_NAME


def read_box_from_stream(f: BufferedIOBase, pos: int) -> MP4Box:
    box = MP4Box.read_from_stream(f, pos)

    match box.type:
        case SupportedBoxes.MDHD:
            return MediaHeaderBox.read_from_stream_parent(f, box)
        case SupportedBoxes.MDIA:
            return MediaBox.read_from_stream_parent(f, box)
        case SupportedBoxes.HDLR:
            return HandlerBox.read_from_stream_parent(f, box)
        case SupportedBoxes.TRAK:
            return TrackBox.read_from_stream_parent(f, box)
        case SupportedBoxes.MOOV:
            return MovieBox.read_from_stream_parent(f, box)
        case SupportedBoxes.FTYP:
            return FileTypeBox.read_from_stream_parent(f, box)
        case SupportedBoxes.FREE:
            return FreeSpaceBox.read_from_stream_parent(f, box)
        case SupportedBoxes.SKIP:
            return FreeSpaceBox.read_from_stream_parent(f, box)
        case SupportedBoxes.UDTA:
            return UserDataBox.read_from_stream_parent(f, box)
        case SupportedBoxes.META:
            return MetaBox.read_from_stream_parent(f, box)
        case SupportedBoxes.ILST:
            return AppleItunesItemList.read_from_stream_parent(f, box)
        case _ if box.type in SupportedBoxes.AppleItunesItemBox:
            return AppleItunesItemBox.read_from_stream_parent(f, box)
        case SupportedBoxes.DATA:
            return AppleItunesItemDataBox.read_from_stream_parent(f, box)
        case _:
            return box


def mp4_iter_boxes(f: BufferedIOBase, start: int, end: int) -> Generator[MP4Box]:
    pos = start

    while pos < end:
        box = read_box_from_stream(f, pos)

        if pos + box.span.size > end:
            msg = f"Box {box.type!r} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield box
        pos += box.span.size


def find_mdhd_boxes_with_type(
    f: BufferedIOBase,
    types: list[ISOMAtomName],
) -> Generator[MediaHeaderBox]:
    f.seek(0, 2)
    filesize = f.tell()

    stack: list[tuple[int, int, list[ISOMAtomName]]] = [(0, filesize, [])]

    while stack:
        start, end, path = stack.pop()

        for box in mp4_iter_boxes(f, start, end):

            if box.type == TRAK_ATOM_NAME:
                if not isinstance(box, TrackBox):
                    msg = "Invalid TrackBox: type not dispatched to correct class"
                    raise TypeError(msg)

                hdlr = box.hdlr.handler_type

                if hdlr not in types:
                    continue

            if box.type == MDHD_ATOM_NAME:
                if not isinstance(box, MediaHeaderBox):
                    msg = "Invalid MediaHeaderBox: type not dispatched to correct class"
                    raise TypeError(msg)

                current = path
                if current != [
                    MOOV_ATOM_NAME,
                    TRAK_ATOM_NAME,
                    MDIA_ATOM_NAME,
                ]:
                    msg = f"invalid mdhd box hierarchy: {current}"
                    raise RuntimeError(msg)

                yield box

            if box.is_container:
                stack.append((box.span.payload_start, box.span.end, [*path, box.type]))


def is_mp4_file(f: BufferedIOBase) -> Optional[str]:
    f.seek(0)

    try:

        first_box = read_box_from_stream(f, 0)

        if not isinstance(first_box, FileTypeBox):
            return _("Not a valid ISOM / MP4 file")

        if first_box.major_brand not in [b"isom", b"mp42"]:
            return _(
                "ISOM/MP42 file has valid box, but invalid major_brand: {major_brand!s}"  # noqa: COM812
            ).format(major_brand=first_box.major_brand)

        f.seek(0)
    except RuntimeError as err:
        return str(err)
    except ValueError as err:
        return str(err)
    return None


class Mp4MetadataHandler:
    __uuid_box: Optional[UUIDExtensionBox]
    __our_boxes: list[MP4Box]

    def __init__(
        self: Self,
        uuid_box: Optional[UUIDExtensionBox],
        our_boxes: list[MP4Box],
    ) -> None:
        self.__uuid_box = uuid_box
        self.__our_boxes = our_boxes

    def remove_old_metadata(self: Self, f: BufferedIOBase) -> None:
        # delete old metadata
        if len(self.__our_boxes) != 0:
            f.truncate(self.__our_boxes[0].span.start)

    def write_new_matadata(
        self: Self,
        f: BufferedIOBase,
        metadata: list[SerializableDict],
        uuid: UUID,
    ) -> None:
        f.seek(0, 2)

        # note: can write 0 or more free space or user extension boxes, and its allowed everywhere

        if self.__uuid_box is not None:
            buffer = UUIDExtensionBox.write_to_buffer(self.__uuid_box.uuid)
            f.write(buffer)
        else:
            buffer = UUIDExtensionBox.write_to_buffer(uuid)

            f.write(buffer)

        # TODO: also write some metadata into these boxes
        # use either top level "meta" or "meco" boxes

        # meta:
        # location, file (0 or 1), inside meco (1 or more, per handler, one meta box!)

        # meco:
        # location, file (0 or 1)

        for mdt in metadata:
            buffer = JsonExtensionBox.write_to_buffer(mdt)

            f.write(buffer)

        f.flush()

    def read_matadata(
        self: Self,
    ) -> tuple[list[SerializableDict], Optional[UUID]]:
        uuid = None if self.__uuid_box is None else self.__uuid_box.uuid
        metadata: list[SerializableDict] = []

        for box in self.__our_boxes:
            if isinstance(box, JsonExtensionBox):
                metadata.append(box.data)
            elif isinstance(box, UUIDExtensionBox):
                if uuid is None:
                    msg = f"Found uuid box manually, but constructor didn't find it: {box}"
                    raise RuntimeError(msg)

            else:
                msg = f"Invalid box for tags found: {type(box)}"
                raise TypeError(msg)

        return (metadata, uuid)

    @staticmethod
    def get_metadata_handler(f: BufferedIOBase) -> "Mp4MetadataHandler":

        uuid_box: Optional[UUIDExtensionBox] = None

        def free_box_is_written_by_us(box: FreeSpaceBox) -> bool:
            free_tag = b"vld\x42\x42\x69-->"

            # NOTE. there are some old legacy ones, that were only used during testing and the new one (which is shorter)
            free_tags: list[bytes] = [
                b"video_language_detect_",
                b"see other metadata for more info by video_language_detect",
                b"vld\x42\x42\x69",
                free_tag,
            ]

            return any(box.data.startswith(tag) for tag in free_tags)

        def uuid_box_is_written_by_us(box: UserExtensionBox) -> bool:
            match box.usertype:
                case UserExtensions.UUIDExtension_UUID:
                    if not isinstance(box, UUIDExtensionBox):
                        msg = "Invalid UUIDExtensionBox: type not dispatched to correct class"
                        raise TypeError(msg)
                    nonlocal uuid_box

                    if uuid_box is not None:
                        msg = "Duplicate uuid box found"
                        raise RuntimeError(msg)

                    uuid_box = box
                    return True
                case UserExtensions.JSONExtension_UUID:
                    return True
                case _:
                    return False

        def box_is_written_by_us(box: MP4Box) -> bool:
            if box.type == FREE_ATOM_NAME:
                if not isinstance(box, FreeSpaceBox):
                    msg = "Invalid FreeSpaceBox: type not dispatched to correct class"
                    raise TypeError(msg)

                return free_box_is_written_by_us(box)

            if box.type == UUID_ATOM_NAME:
                if not isinstance(box, UserExtensionBox):
                    msg = (
                        "Invalid UserExtensionBox: type not dispatched to correct class"
                    )
                    raise TypeError(msg)

                return uuid_box_is_written_by_us(box)

            return False

        f.seek(0, 2)
        end = f.tell()

        f.seek(0)

        top_boxes: list[MP4Box] = list(mp4_iter_boxes(f, 0, end=end))

        our_boxes_reversed: list[MP4Box] = []
        other_box_encountered = False
        for box in reversed(top_boxes):
            if other_box_encountered:
                break

            if box_is_written_by_us(box):
                our_boxes_reversed.append(box)
            else:
                other_box_encountered = True
                break

        return Mp4MetadataHandler(uuid_box, list(reversed(our_boxes_reversed)))


class VideoTaggerWriterMP4(VideoTaggerWriter):
    __writer: BufferedIOBase
    __streams: int
    __types: list[ISOMAtomName]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        writer: BufferedIOBase,
        streams: int,
        types: list[ISOMAtomName],
    ) -> None:
        super().__init__(manager)
        self.__writer = writer
        self.__streams = streams
        self.__types = types

    @override
    def write_tags(
        self: Self,
        tags: MetadataTags,
    ) -> None:

        new_language = tags.language.short

        bar: CounterInterface = self.manager.counter(
            total=float(self.__streams + 1),
            desc="update mp4 language",
            unit="B",
            leave=False,
            bar_format=VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
            color="red",
        )
        bar.update(0, force=True)

        try:
            self.__writer.seek(0)
            for mdhd in find_mdhd_boxes_with_type(self.__writer, self.__types):

                should_write_language = True

                lang = mdhd.read_language(self.__writer)
                if isinstance(lang, ShortLanguageStr) and new_language == lang:
                    should_write_language = False

                if should_write_language:
                    mdhd.patch_language(self.__writer, new_language)

                bar.update(1, force=True)

            mp4_metadata_handler = Mp4MetadataHandler.get_metadata_handler(
                f=self.__writer,
            )

            mp4_metadata_handler.remove_old_metadata(self.__writer)

            metadata_dict: SerializableDict = {
                "comment": tags.comment,
                "metadata": tags.metadata,
            }

            mp4_metadata_handler.write_new_matadata(
                self.__writer,
                [metadata_dict],
                tags.uuid,
            )

            self.__writer.flush()
        finally:
            bar.close(clear=True)

    @override
    def get_tags(
        self: Self,
    ) -> MetadataTagsRead:

        def decode_as_str(value: Any) -> str:
            if not isinstance(value, str):
                msg = f"Invalid type in decode_as_str: {type(value)}"
                raise TypeError(msg)

            return value

        def decode_as_dict(value: Any) -> SerializableDict:
            if not isinstance(value, dict):
                msg = f"Invalid type in decode_as_dict: {type(value)}"
                raise TypeError(msg)

            return value

        mp4_metadata_handler = Mp4MetadataHandler.get_metadata_handler(
            f=self.__writer,
        )

        metadata, uuid = mp4_metadata_handler.read_matadata()

        result: MetadataTagsRead = MetadataTagsRead(None, None, {}, [])

        if uuid is not None:
            result.uuid = uuid

        for mdt in metadata:
            for key, value in mdt.items():

                if key == "comment":
                    if result.comment is not None:
                        msg = f"Duplicate comment tag read: {value}"
                        raise RuntimeError(msg)

                    result.comment = decode_as_str(value)

                elif key == "metadata":
                    if len(result.metadata.items()) != 0:
                        msg = f"Duplicate metadata tag read: {value}"
                        raise RuntimeError(msg)

                    result.metadata = decode_as_dict(value)
                else:
                    result.unrecognized.append(
                        (key, decode_as_str(value)),
                    )

        return result


class VideoTaggerMP4(VideoTagger):
    __streams: int
    __types: list[ISOMAtomName]

    def __init__(
        self: Self,
        file: Path,
        streams: int,
        types: list[ISOMAtomName],
    ) -> None:
        super().__init__(file)
        self.__streams = streams
        self.__types = types

        streams = 0

    @staticmethod
    def get_handle(file: Path) -> Result["VideoTagger", str]:

        try:

            with file.open("rb") as f:
                mp4_res = is_mp4_file(f)
                if mp4_res is not None:
                    return Err(mp4_res)

                f.seek(0)

                streams = 0
                types: list[ISOMAtomName] = [SOUN_ATOM_NAME, VIDE_ATOM_NAME]

                # read the file, so that we check if we can parse it correctly and that it is an mp4
                for mdhd in find_mdhd_boxes_with_type(f, types):
                    streams = streams + 1
                    lang = mdhd.read_language(f)
                    # check if this lang is valid

                    if isinstance(lang, str):
                        msg = _("Invalid language in mp4 detected: {lang}").format(
                            lang=lang,
                        )
                        return Err(msg)

                return Ok(
                    VideoTaggerMP4(file, streams, types),
                )
        except RuntimeError as err:
            return Err(str(err))
        except ValueError as err:
            return Err(str(err))
        except TypeError as err:
            return Err(str(err))

    @override
    def writer(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerWriter]:

        file = self.file
        streams = self.__streams
        types = self.__types

        class VideoTaggerWriterCtx(AbstractContextManager[VideoTaggerWriter]):
            __writer: Optional[BufferedIOBase]

            def __init__(self: Self) -> None:
                super().__init__()
                self.__writer = None

            @override
            def __enter__(self: Self) -> VideoTaggerWriter:
                self.__writer = file.open("rb+")

                return VideoTaggerWriterMP4(manager, self.__writer, streams, types)

            @override
            def __exit__(
                self: Self,
                _exc_type: Optional[type[BaseException]],
                _exc_val: Optional[BaseException],
                _exc_tb: Optional[TracebackType],
            ) -> Literal[False]:  # actually bool
                if self.__writer is not None:
                    self.__writer.close()
                return False

        return VideoTaggerWriterCtx()
