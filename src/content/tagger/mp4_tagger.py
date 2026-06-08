import json
from collections.abc import Generator
from contextlib import AbstractContextManager
from io import BufferedIOBase, BytesIO
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Self, final, override
from uuid import UUID

from content.language import Language, ShortLanguageStr
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
        f: BufferedIOBase, parent: UserExtensionBox
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
        #     ‘ftyp’
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

        # free_type may be ‘free’ or ‘skip’.
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
        #     ‘mdia’
        #     ) {
        # }

        return MediaBox(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MediaBox":
        box = MP4Box.read_from_stream(f, offset)
        return MediaBox.__read_from_stream_impl(f, box)

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
        #     ‘moov’
        #     ){
        # }

        return MovieBox(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "MovieBox":
        box = MP4Box.read_from_stream(f, offset)
        return MovieBox.__read_from_stream_impl(f, box)

    def __str__(self: Self) -> str:
        return f"<MovieBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
class HandlerBox(MP4FullBox):
    handler_type: ISOMAtomName

    def __init__(self: Self, parent: MP4FullBox, handler_type: ISOMAtomName) -> None:
        super().__init__(
            parent,
            parent.version,
            parent.flags,
            is_container=False,
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
        handler_type = ISOMAtomName(handler_type_raw)

        # omitting dynamic sized string "name"
        additional_header_size = 4 + 4 + (4 * 3)

        parent.span.add_header_size(additional_header_size)

        return HandlerBox(parent, handler_type)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "HandlerBox":
        box = MP4FullBox.read_from_stream(f, offset)
        return HandlerBox.__read_from_stream_impl(f, box)

    def __str__(self: Self) -> str:
        return f"<HandlerBox parent: {MP4FullBox.__str__(self)} handler_type: {self.handler_type}>"

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
        #     ‘trak’
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
        #     ‘udta’
        #     ) {
        # }

        return UserDataBox(parent)

    @staticmethod
    def read_from_stream(f: BufferedIOBase, offset: int) -> "UserDataBox":
        box = MP4Box.read_from_stream(f, offset)
        return UserDataBox.__read_from_stream_impl(f, box)

    def __str__(self: Self) -> str:
        return f"<UserDataBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


def read_box_from_stream(f: BufferedIOBase, pos: int) -> MP4Box:
    box = MP4Box.read_from_stream(f, pos)

    match box.type.value:
        case b"mdhd":
            return MediaHeaderBox.read_from_stream(f, pos)
        case b"mdia":
            return MediaBox.read_from_stream(f, pos)
        case b"hdlr":
            return HandlerBox.read_from_stream(f, pos)
        case b"trak":
            return TrackBox.read_from_stream(f, pos)
        case b"moov":
            return MovieBox.read_from_stream(f, pos)
        case b"ftyp":
            return FileTypeBox.read_from_stream(f, pos)
        case b"free":
            return FreeSpaceBox.read_from_stream(f, pos)
        case b"skip":
            return FreeSpaceBox.read_from_stream(f, pos)
        case b"udta":
            return UserDataBox.read_from_stream(f, pos)
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
    __our_boxes_start: Optional[MP4Box]

    def __init__(
        self: Self,
        uuid_box: Optional[UUIDExtensionBox],
        our_boxes_start: Optional[MP4Box],
    ):
        self.__uuid_box = uuid_box
        self.__our_boxes_start = our_boxes_start

    def remove_old_metadata(self: Self, f: BufferedIOBase) -> None:
        # delete old metadata
        if self.__our_boxes_start is not None:
            f.truncate(self.__our_boxes_start.span.start)

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

        first_box_written_by_us: Optional[MP4Box] = None
        other_box_encountered = False
        for box in reversed(top_boxes):
            if other_box_encountered:
                break

            if box_is_written_by_us(box):
                first_box_written_by_us = box
            else:
                other_box_encountered = True
                break

        return Mp4MetadataHandler(uuid_box, first_box_written_by_us)


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
    def write_metadata(
        self: Self,
        comment: str,
        uuid: UUID,
        language: Language,
        metadata: SerializableDict,
    ) -> None:

        new_language = language.short

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
                f=self.__writer
            )

            mp4_metadata_handler.remove_old_metadata(self.__writer)

            metadata_list: list[SerializableDict] = [
                {"comment": comment},
                {"metadata": metadata},
            ]

            mp4_metadata_handler.write_new_matadata(self.__writer, metadata_list, uuid)
        finally:
            bar.close(clear=True)


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
