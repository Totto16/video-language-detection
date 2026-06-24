import json
from collections.abc import Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Literal,
    Optional,
    Self,
    assert_never,
    cast,
    final,
    override,
)
from uuid import UUID

from content.language import Language, ShortLanguageStr
from content.tagger.parser import (
    ISOM_BYTE_ORDER,
    BoundedIO,
    Packable,
    Packer,
    SimpleSpan,
    Unpacker,
    UnsignedInt,
    UnsignedLongLong,
    UnsignedShort,
    uuid_from_bytes,
    uuid_to_bytes,
)
from content.tagger.utils import merge_dicts
from content.tagger.video_tagger import (
    TAGGER_DOMAIN,
    VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
    AppleItunesFreeformKey,
    ContextType,
    InspectElement,
    InspectNotImplemented,
    InspectPrinter,
    InspectPriority,
    MetadataTags,
    MetadataTagsRead,
    RestoreFileNotSupported,
    SerializableDict,
    SerializableDictValue,
    TaggerDomain,
    VideoTagger,
    VideoTaggerContextCtxGeneric,
    VideoTaggerContextReadable,
    VideoTaggerContextRW,
    VideoTaggerContextWriteable,
    uuid_from_str,
    uuid_to_str,
)
from helper.decorator import decorate_class
from helper.manager import CounterInterface, ManagerInterface
from helper.result import Err, Ok, Result
from helper.translation import get_translator

_ = get_translator()


@final
@decorate_class(slots=True)
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


@final
@decorate_class(slots=True)
class PackableISOMAtomName(Packable[ISOMAtomName, bytes]):
    @property
    @override
    def pack_str(self: Self) -> str:
        return "4s"

    @property
    @override
    def pack_size(self: Self) -> int:
        return 4

    @override
    def to_underlying(self: Self, value: ISOMAtomName) -> bytes:
        return value.value

    @override
    def from_underlying(self: Self, value: bytes) -> ISOMAtomName:
        return ISOMAtomName(value)


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
MEAN_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mean")
NAME_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"name")
MVHD_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mvhd")
IODS_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"iods")
TKHD_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"tkhd")
MDAT_ATOM_NAME: ISOMAtomName = ISOMAtomName(b"mdat")
PITM_ATOM_NAME: ISOMAtomName = ISOMAtomName(value=b"pitm")
MDIR_ATOM_NAME: ISOMAtomName = ISOMAtomName(value=b"mdir")


@final
@decorate_class(slots=True)
class MP4BoxSpan:
    __total: SimpleSpan

    __intervals: list[int]

    def __init__(
        self: Self,
        span: SimpleSpan,
        header_size: int,
    ) -> None:
        self.__total = span
        self.__intervals = [header_size]

        if self.__total.size < 8:
            msg = f"Invalid box: size too small: {self.__total.size}"
            raise RuntimeError(msg)

        if self.__total.size < header_size:
            msg = f"Invalid box size {self.__total.size} at {self.__total.start}"
            raise RuntimeError(msg)

    def __interval_span_impl(self: Self, depth: int = 0) -> SimpleSpan:
        if len(self.__intervals) == 0:
            msg = "Implementation error: intervals list is empty"
            raise RuntimeError(msg)

        if depth < 0:
            msg = f"Invalid depth, it is negative: {depth}"
            raise RuntimeError(msg)

        if depth > len(self.__intervals):
            msg = f"Invalid depth of interval size: {depth}, max is {len(self.__intervals)}"
            raise RuntimeError(msg)

        interval_start = self.__total.start
        interval_end = self.__total.end

        if depth != 0:
            interval_start = self.__total.start + sum(self.__intervals[0:depth])

        if depth != len(self.__intervals):
            interval_end = self.__total.start + sum(self.__intervals[0 : depth + 1])

        interval_size = interval_end - interval_start

        if interval_size > self.__total.size or interval_size < 0:
            msg = f"Implementation error, interval_size out of bounds [0, {self.__total.size}]: {interval_size}"

        return SimpleSpan(interval_start, interval_size)

    def header_span(self: Self, depth: int = 0) -> SimpleSpan:
        if depth == -1:
            return self.header_span(len(self.__intervals) - 1)

        if depth >= len(self.__intervals):
            msg = f"Invalid depth of header size: {depth}, max is {len(self.__intervals) - 1}"
            raise RuntimeError(msg)

        return self.__interval_span_impl(depth)

    @property
    def payload_span(self: Self) -> SimpleSpan:
        return self.__interval_span_impl(len(self.__intervals))

    def add_header(self: Self, header_size: int) -> None:
        self.__intervals.append(header_size)

        if self.__total.size < sum(self.__intervals):
            msg = f"Invalid total header size {self.__total.size} < {sum(self.__intervals)} at {self.__total.start}"
            raise RuntimeError(msg)

    @property
    def total(self: Self) -> SimpleSpan:
        return self.__total

    def __str__(self: Self) -> str:
        header_string = ", ".join(
            str(self.header_span(i)) for i in range(len(self.__intervals))
        )
        return f"<MP4BoxSpan total: {self.__total} header: [ {header_string} ] payload: {self.payload_span}>"

    def __repr__(self: Self) -> str:
        return str(self)


@decorate_class(slots=True)
class FinalMP4Box:
    __final__mp4_box__ = True


@decorate_class(slots=True)
class NonFinalMP4Box:
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        is_final = getattr(cls, "__final__mp4_box__", False)

        if not is_final:
            for fn_name in [
                "read",
                "write_to_buffer",
                "read_from_parent",
            ]:
                if fn_name in cls.__dict__:
                    msg = f"{cls.__name__} defines {fn_name}(), but only final classes may do so"
                    raise TypeError(msg)


# ruff: disable[ERA001]


@decorate_class(slots=True)
class MP4Box(NonFinalMP4Box):
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
    def read_mp4_box(io: BoundedIO) -> "MP4Box":
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

        with io.r_ctx(force_entire_read=False) as f:

            hdr = f.read(8)

            size, typ = Unpacker.unpack_two(
                ISOM_BYTE_ORDER,
                (UnsignedInt(), PackableISOMAtomName()),
                hdr,
            )

            if typ == CMOV_ATOM_NAME:
                msg = f"Compressed movie box '{CMOV_ATOM_NAME}' not supported"
                raise RuntimeError(msg)

            if typ == MOOF_ATOM_NAME:
                msg = f"Fragmented MP4 '{MOOF_ATOM_NAME}' not supported"
                raise RuntimeError(msg)

            final_size: int = size
            header_size: int = 8

            if size == 1:
                ext = f.read(8)

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
                final_size = io.special_checked_filesize()
                header_size = 8

            if final_size > io.span.size:
                msg = f"Invalid MP4 Box size: It overflows the parent box: {final_size} > {io.span.size}"
                raise RuntimeError(msg)

            if typ != UUID_ATOM_NAME:
                span = MP4BoxSpan(
                    io.span.sub_span(final_size),
                    header_size=header_size,
                )
                return MP4Box(typ, span, is_container=False)

            usertype_raw = f.read(16)

            usertype = uuid_from_bytes(ISOM_BYTE_ORDER, usertype_raw)

            header_size = header_size + 16

            span = MP4BoxSpan(
                io.span.sub_span(final_size),
                header_size=header_size,
            )
            box = MP4Box(typ, span, is_container=False)
            user_box = UserExtensionBox(box, usertype, is_container=False)

        # out of the context manager, we can read again :)
        return user_extension_box_determine_correct_extension(
            user_box.payload_io(io),
            user_box,
        )

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
                typ,
            ),
            8,
        )

        buf.write(hdr)

        if additional_size is not None:
            largesize = Packer.pack_one(
                ISOM_BYTE_ORDER,
                UnsignedLongLong(),
                additional_size,
                8,
            )

            buf.write(largesize)

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

    @final
    def payload_io(self: Self, io: BoundedIO) -> BoundedIO:
        return io.new_span_io(
            self.span.payload_span,
        )

    @final
    def header_io(self: Self, io: BoundedIO, depth: int = 0) -> BoundedIO:
        return io.new_span_io(
            self.span.header_span(depth),
        )

    def __str__(self: Self) -> str:
        return f"<MP4Box type: {self.type} span: {self.span} is_container: {self.is_container}>"

    def __repr__(self: Self) -> str:
        return str(self)


@decorate_class(slots=True)
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


UUIDExtension_UUID = UUID(hex="90e175d1-efdb-4144-a214-ebfab6258ae7")
JSONExtension_UUID = UUID(hex="90e175d1-efdb-4144-a214-ebfab6258ae8")


@decorate_class(slots=True)
class UserExtensions:
    UUIDExtension_UUID = UUIDExtension_UUID
    JSONExtension_UUID = JSONExtension_UUID


@final
@decorate_class(slots=True)
class UUIDExtensionBox(UserExtensionBox, FinalMP4Box):
    uuid: UUID

    def __init__(
        self: Self,
        parent: UserExtensionBox,
        uuid: UUID,
    ) -> None:
        super().__init__(parent, parent.usertype, is_container=False)

        self.uuid = uuid

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: UserExtensionBox,
    ) -> "UUIDExtensionBox":

        # this is a custom user box, it contains one UUID

        with io.r_ctx(force_entire_read=True) as f:

            if parent.span.payload_span.size != 16:
                msg = f"UUIDExtensionBox has not the correct payload size: {parent.span.payload_span.size}"
                raise RuntimeError(msg)

            uuid_raw = f.read(16)

            uuid = uuid_from_bytes(ISOM_BYTE_ORDER, uuid_raw)

            parent.span.add_header(16)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

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
@decorate_class(slots=True)
class JsonExtensionBox(UserExtensionBox, FinalMP4Box):
    data: SerializableDict

    def __init__(
        self: Self,
        parent: UserExtensionBox,
        data: SerializableDict,
    ) -> None:
        super().__init__(parent, parent.usertype, is_container=False)

        self.data = data

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: UserExtensionBox,
    ) -> "JsonExtensionBox":

        # this is a custom user box, it contains a json payload

        with io.r_ctx(force_entire_read=True) as f:

            data_raw = f.read(parent.span.payload_span.size)

            data = json.loads(data_raw.decode())

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

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
    io: BoundedIO,
    box: UserExtensionBox,
) -> UserExtensionBox:
    match box.usertype:
        case UserExtensions.UUIDExtension_UUID:
            return UUIDExtensionBox.read_from_parent(io, box)
        case UserExtensions.JSONExtension_UUID:
            return JsonExtensionBox.read_from_parent(io, box)
        case _:
            return box


@decorate_class(slots=True)
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
    def __read_impl(io: BoundedIO, parent: MP4Box) -> "MP4FullBox":
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

        with io.r_ctx(force_entire_read=False) as f:

            version = f.read(1)[0]

            flags = f.read(3)

            parent.span.add_header(4)

            return MP4FullBox(parent, version, flags, is_container=False)

    @staticmethod
    def read_mp4_full_box(io: BoundedIO) -> "MP4FullBox":
        box = MP4Box.read_mp4_box(io)
        return MP4FullBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent_mp4_full_box(io: BoundedIO, parent: MP4Box) -> "MP4FullBox":
        return MP4FullBox.__read_impl(io, parent)

    @staticmethod
    def __impl_write_to_buffer_mp4_full_box(
        typ: ISOMAtomName,
        version: int,
        flags: bytes,
        data: bytes,
    ) -> bytes:
        buf = BytesIO()

        if version < 0 or version > 0xFF:
            msg = f"Invalid version, only 1 byte values allowed, but got: {version}"
            raise RuntimeError(msg)

        buf.write(bytes([version]))

        if len(flags) != 3:
            msg = f"Invalid flags, have to be 3 bytes long: {flags!r}"
            raise RuntimeError(msg)

        buf.write(flags)
        buf.write(data)

        final_data = buf.getvalue()

        return MP4Box.write_to_buffer_mp4_box(typ, final_data, usertype=None)

    @staticmethod
    def write_to_buffer_mp4_full_box(
        typ: ISOMAtomName,
        version: int,
        flags: bytes,
        data: bytes,
    ) -> bytes:
        return MP4FullBox.__impl_write_to_buffer_mp4_full_box(typ, version, flags, data)

    def __str__(self: Self) -> str:
        return f"<MP4FullBox parent: {MP4Box.__str__(self)} version: {self.version} flags: {self.flags.hex()}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class FileTypeBox(MP4Box, FinalMP4Box):
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
    def __read_impl(io: BoundedIO, parent: MP4Box) -> "FileTypeBox":
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

        with io.r_ctx(force_entire_read=True) as f:

            major_brand_raw = f.read(4)
            major_brand = ISOMAtomName(major_brand_raw)

            minor_version_bytes = f.read(4)

            minor_version = Unpacker.unpack_one(
                ISOM_BYTE_ORDER,
                UnsignedInt(),
                minor_version_bytes,
            )

            compatible_brands_size = parent.span.payload_span.size - (4 + 4)

            if compatible_brands_size < 0:
                msg = f"Invalid box size: not enough data for complete FileTypeBox: have {parent.span.payload_span.size} but need at least {(4 + 4)}"
                raise RuntimeError(msg)

            compatible_brands = f.read(compatible_brands_size)

            additional_header_size = 4 + 4 + compatible_brands_size

            parent.span.add_header(additional_header_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return FileTypeBox(parent, major_brand, minor_version, compatible_brands)

    @staticmethod
    def read(io: BoundedIO) -> "FileTypeBox":
        box = MP4Box.read_mp4_box(io)
        return FileTypeBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "FileTypeBox":
        return FileTypeBox.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<FileTypeBox parent: {MP4Box.__str__(self)} major_brand: {self.major_brand} minor_version: {self.minor_version} compatible_brands: {self.compatible_brands!s}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class FreeSpaceBox(MP4Box, FinalMP4Box):
    data: bytes

    def __init__(
        self: Self,
        parent: MP4Box,
        data: bytes,
    ) -> None:
        super().__init__(parent.type, parent.span, is_container=False)

        self.data = data

    @staticmethod
    def __read_impl(io: BoundedIO, parent: MP4Box) -> "FreeSpaceBox":
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

        with io.r_ctx(force_entire_read=True) as f:
            data = f.read(parent.span.payload_span.size)

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return FreeSpaceBox(parent, data)

    @staticmethod
    def read(io: BoundedIO) -> "FreeSpaceBox":
        box = MP4Box.read_mp4_box(io)
        return FreeSpaceBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "FreeSpaceBox":
        return FreeSpaceBox.__read_impl(io, parent)

    @staticmethod
    def write_to_buffer(data: bytes) -> bytes:
        return MP4Box.write_to_buffer_mp4_box(FREE_ATOM_NAME, data)

    def __str__(self: Self) -> str:
        return f"<FreeSpaceBox parent: {MP4Box.__str__(self)} data: {self.data!s}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class MediaHeaderBox(MP4FullBox, FinalMP4Box):
    # offset from the own header start, not the start of the whole box!
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
    def __read_impl(
        io: BoundedIO,
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

        with io.r_ctx(force_entire_read=True) as f:

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

            if parent.span.payload_span.size < additional_header_size:
                msg = f"Truncated mdhd header {parent.span.payload_span.size} < {additional_header_size}"
                raise RuntimeError(msg)

            parent.span.add_header(additional_header_size)

            header_language_offset = version_dependend_size

            if (
                parent.span.header_span(-1).start + header_language_offset + 2
                > parent.span.header_span(-1).end
            ):
                msg = "Language field outside mdhd header bounds"
                raise RuntimeError(msg)

            f.skip(additional_header_size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return MediaHeaderBox(parent, header_language_offset)

    @staticmethod
    def read(io: BoundedIO) -> "MediaHeaderBox":
        box = MP4FullBox.read_mp4_full_box(io)
        return MediaHeaderBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "MediaHeaderBox":
        box: MP4FullBox
        if isinstance(parent, MP4FullBox):
            box = parent
        else:
            box = MP4FullBox.read_from_parent_mp4_full_box(io, parent)
        return MediaHeaderBox.__read_impl(box.payload_io(io), box)

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

    def read_language(self: Self, io_base: BinaryIO) -> ShortLanguageStr | str:
        io = self.header_io(BoundedIO.get_new(io_base, self.span.total), -1)

        with io.r_ctx(force_entire_read=False) as f:
            f.skip(self.language_offset)

            lang_bytes = f.read(2)
            packed = Unpacker.unpack_one(ISOM_BYTE_ORDER, UnsignedShort(), lang_bytes)

            return MediaHeaderBox.__decode_language_impl(packed)

    def patch_language(
        self: Self,
        io_base: BinaryIO,
        new_language: ShortLanguageStr,
    ) -> None:
        packed = MediaHeaderBox.__encode_language_impl(new_language)

        io = self.header_io(BoundedIO.get_new(io_base, self.span.total), -1)

        with io.rw_ctx(force_entire_read=False) as f:
            f.skip(self.language_offset)

            packed_bytes = Packer.pack_one(ISOM_BYTE_ORDER, UnsignedShort(), packed, 2)

            f.write(packed_bytes)
            f.flush()

        with io.r_ctx(force_entire_read=False) as f:
            f.skip(self.language_offset)
            verify_bytes = f.read(2)
            verify = Unpacker.unpack_one(ISOM_BYTE_ORDER, UnsignedShort(), verify_bytes)

            if verify != packed:
                msg = "Invalid overwrite"
                raise RuntimeError(msg)

    def __str__(self: Self) -> str:
        return f"<MediaHeaderBox parent: {MP4FullBox.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class MediaBox(MP4Box, FinalMP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_impl(io: BoundedIO, parent: MP4Box) -> "MediaBox":
        # spec: ISO/IEC 14496-12
        # ISO media box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class MediaBox extends Box(
        #     'mdia'
        #     ) {
        # }

        with io.r_ctx(force_entire_read=True) as f:
            f.skip(parent.span.payload_span.size)

        return MediaBox(parent)

    @staticmethod
    def read(io: BoundedIO) -> "MediaBox":
        box = MP4Box.read_mp4_box(io)
        return MediaBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "MediaBox":
        return MediaBox.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<MediaBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class MovieBox(MP4Box, FinalMP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_impl(io: BoundedIO, parent: MP4Box) -> "MovieBox":
        # spec: ISO/IEC 14496-12
        # ISO movie box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class MovieBox extends Box(
        #     'moov'
        #     ){
        # }

        with io.r_ctx(force_entire_read=True) as f:
            f.skip(parent.span.payload_span.size)

        return MovieBox(parent)

    @staticmethod
    def read(io: BoundedIO) -> "MovieBox":
        box = MP4Box.read_mp4_box(io)
        return MovieBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "MovieBox":
        return MovieBox.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<MovieBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class HandlerBox(MP4FullBox, FinalMP4Box):
    handler_type: ISOMAtomName
    name: str

    def __init__(
        self: Self,
        parent: MP4FullBox,
        handler_type: ISOMAtomName,
        name: str,
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
    def __read_impl(io: BoundedIO, parent: MP4FullBox) -> "HandlerBox":
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

        with io.r_ctx(force_entire_read=True) as f:

            if parent.version != 0:
                msg = f"Invalid hdlr version: {parent.version}"
                raise RuntimeError(msg)

            pre_defined = f.read(4)

            if pre_defined != b"\x00" * 4:
                msg = f"HandlerBox: pre_defined has to be 0, but was: {pre_defined!r}"
                raise ValueError(msg)

            handler_type = Unpacker.unpack_one(
                ISOM_BYTE_ORDER,
                PackableISOMAtomName(),
                f.read(4),
            )

            reserved = f.read(4 * 3)

            if reserved != b"\x00" * (3 * 4):
                # make exception for apples usage of these types
                if reserved.startswith(b"appl"):
                    pass
                else:
                    msg = f"HandlerBox: reserved has to be 0, but was: {reserved!r}"
                    raise ValueError(msg)

            # omitting dynamic sized string "name"
            fixed_header_size = 4 + 4 + (4 * 3)

            name_size = parent.span.payload_span.size - fixed_header_size

            name_raw = f.read(name_size)

            name = name_raw.decode()

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return HandlerBox(parent, handler_type, name)

    @staticmethod
    def read(io: BoundedIO) -> "HandlerBox":
        box = MP4FullBox.read_mp4_full_box(io)
        return HandlerBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "HandlerBox":
        box: MP4FullBox
        if isinstance(parent, MP4FullBox):
            box = parent
        else:
            box = MP4FullBox.read_from_parent_mp4_full_box(io, parent)
        return HandlerBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def write_to_buffer(
        handler_type: ISOMAtomName,
        name: str,
        *,
        reserved: Optional[bytes] = None,
    ) -> bytes:
        buf = BytesIO()

        buf.write(b"\x00" * 4)  # pre_defined

        handler_type_bytes = Packer.pack_one(
            ISOM_BYTE_ORDER,
            PackableISOMAtomName(),
            handler_type,
            4,
        )

        buf.write(handler_type_bytes)

        if reserved is not None:
            if len(reserved) != (3 * 4):
                msg = f"Invalid reserved, only {3*4} byte values allowed, but got: {reserved!r}"
                raise RuntimeError(msg)
            buf.write(reserved)
        else:
            buf.write(b"\x00" * (3 * 4))

        buf.write(name.encode())
        # note: 1 '\x00' byte for the name!
        buf.write(b"\x00")

        final_data = buf.getvalue()

        return MP4FullBox.write_to_buffer_mp4_full_box(
            HDLR_ATOM_NAME,
            0,
            b"\x00" * 3,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<HandlerBox parent: {MP4FullBox.__str__(self)} handler_type: {self.handler_type} name: {self.name}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class TrackBox(MP4Box, FinalMP4Box):
    hdlr: HandlerBox

    def __init__(self: Self, parent: MP4Box, hdlr: HandlerBox) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

        self.hdlr = hdlr

    @staticmethod
    def __read_impl(io: BoundedIO, parent: MP4Box) -> "TrackBox":
        # spec: ISO/IEC 14496-12
        # ISO track box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class TrackBox extends Box(
        #     'trak'
        # ) {
        # }

        mdia_box: Optional[MediaBox] = None

        for box in mp4_iter_boxes_io(
            parent.payload_io(io),
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

        for box in mp4_iter_boxes_io(
            mdia_box.payload_io(io),
        ):
            if box.type == HDLR_ATOM_NAME:
                if not isinstance(box, HandlerBox):
                    msg = "Invalid HandlerBox: type not dispatched to correct class"
                    raise TypeError(msg)

                return TrackBox(parent, box)

        msg = "Missing hdlr box in trak"
        raise RuntimeError(msg)

    @staticmethod
    def read(io: BoundedIO) -> "TrackBox":
        box = MP4Box.read_mp4_box(io)
        return TrackBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "TrackBox":
        return TrackBox.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<TrackBox parent: {MP4Box.__str__(self)} hdlr: {self.hdlr}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class UserDataBox(MP4Box, FinalMP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_impl(io: BoundedIO, parent: MP4Box) -> "UserDataBox":
        # spec: ISO/IEC 14496-12
        # ISO user data box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class UserDataBox extends Box(
        #     'udta'
        #     ) {
        # }

        with io.rw_ctx(force_entire_read=True) as f:
            f.skip(parent.span.payload_span.size)

        return UserDataBox(parent)

    @staticmethod
    def read(io: BoundedIO) -> "UserDataBox":
        box = MP4Box.read_mp4_box(io)
        return UserDataBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "UserDataBox":
        return UserDataBox.__read_impl(io, parent)

    def __str__(self: Self) -> str:
        return f"<UserDataBox parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class PrimaryItemBox(MP4FullBox, FinalMP4Box):
    item_id: int

    def __init__(
        self: Self,
        parent: MP4FullBox,
        item_id: int,
    ) -> None:
        super().__init__(
            parent,
            parent.version,
            parent.flags,
            is_container=False,
        )
        self.item_id = item_id

    @staticmethod
    def __read_impl(io: BoundedIO, parent: MP4FullBox) -> "PrimaryItemBox":
        # spec: ISO/IEC 14496-12
        # ISO Primary item box structure:
        # box     | <full box size> bytes | parent full box
        # ... data

        # aligned(8) class PrimaryItemBox
        #     extends FullBox(
        #         'pitm',
        #         version = 0,
        #         0)
        # {
        #     unsigned int(16) item_ID;
        # }

        # see https://mpeggroup.github.io/FileFormatConformance/?query=%3D%22pitm%22
        # for known ids

        with io.r_ctx(force_entire_read=True) as f:

            if parent.version != 0:
                msg = f"Invalid pitm version: {parent.version}"
                raise RuntimeError(msg)

            item_id_raw = f.read(2)

            item_id = Unpacker.unpack_one(ISOM_BYTE_ORDER, UnsignedShort(), item_id_raw)

            parent.span.add_header(2)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return PrimaryItemBox(parent, item_id)

    @staticmethod
    def read(io: BoundedIO) -> "PrimaryItemBox":
        box = MP4FullBox.read_mp4_full_box(io)
        return PrimaryItemBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "PrimaryItemBox":
        box: MP4FullBox
        if isinstance(parent, MP4FullBox):
            box = parent
        else:
            box = MP4FullBox.read_from_parent_mp4_full_box(io, parent)
        return PrimaryItemBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def write_to_buffer(item_id: int) -> bytes:
        buf = BytesIO()

        if item_id < 0 or item_id > 0xFFFF:
            msg = f"Invalid item_id, only 2 byte values allowed, but got: {item_id}"
            raise RuntimeError(msg)

        item_id_bytes = Packer.pack_one(ISOM_BYTE_ORDER, UnsignedShort(), item_id, 2)
        buf.write(item_id_bytes)

        final_data = buf.getvalue()

        return MP4FullBox.write_to_buffer_mp4_full_box(
            PITM_ATOM_NAME,
            0,
            b"\x00" * 3,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<PrimaryItemBox parent: {MP4FullBox.__str__(self)} item_id: {self.item_id}"

    def __repr__(self: Self) -> str:
        return str(self)


@dataclass(slots=True, repr=True)
class MetaBoxHandlerInformation:
    handler_type: ISOMAtomName
    name: str
    reserved: Optional[bytes] = None


@dataclass(slots=True, repr=True)
class PrimaryItemInformation:
    item_id: int


@dataclass(slots=True, repr=True)
class OptionalMetaBoxes:
    pitm: Optional[PrimaryItemBox]  # PrimaryItemBox
    dinf: Optional[MP4Box]  # DataInformationBox
    iloc: Optional[MP4Box]  # ItemLocationBox
    ipro: Optional[MP4Box]  # ItemProtectionBox
    iinf: Optional[MP4Box]  # ItemInfoBox
    ipmc: Optional[MP4Box]  # IPMPControlBox
    iref: Optional[MP4Box]  # ItemReferenceBox
    idat: Optional[MP4Box]  # ItemDataBox

    @staticmethod
    def empty() -> "OptionalMetaBoxes":
        return OptionalMetaBoxes(
            pitm=None,
            dinf=None,
            iloc=None,
            ipro=None,
            iinf=None,
            ipmc=None,
            iref=None,
            idat=None,
        )

    def __str__(self: Self) -> str:
        values: list[str] = [
            str(val)
            for val in [
                self.pitm,
                self.dinf,
                self.iloc,
                self.ipro,
                self.iinf,
                self.ipmc,
                self.iref,
                self.idat,
            ]
            if val is not None
        ]

        return f"<OptionalMetaBoxes {{{", ".join(values)}}}>"

    def __repr__(self: Self) -> str:
        return str(self)


META_OPTIONAL_BOXES: list[ISOMAtomName] = [
    PITM_ATOM_NAME,  # PrimaryItemBox
    ISOMAtomName(b"dinf"),  # DataInformationBox
    ISOMAtomName(b"iloc"),  # ItemLocationBox
    ISOMAtomName(b"ipro"),  # ItemProtectionBox
    ISOMAtomName(b"iinf"),  # ItemInfoBox
    ISOMAtomName(b"ipmc"),  # IPMPControlBox
    ISOMAtomName(b"iref"),  # ItemReferenceBox
    ISOMAtomName(b"idat"),  # ItemDataBox
]


@decorate_class(slots=True)
class MetaOptionalBoxes:
    PITM = PITM_ATOM_NAME


@final
@decorate_class(slots=True)
class MetaBox(MP4FullBox, FinalMP4Box):
    handler_box: HandlerBox
    optional_boxes: OptionalMetaBoxes

    def __init__(
        self: Self,
        parent: MP4FullBox,
        handler_box: HandlerBox,
        optional_boxes: OptionalMetaBoxes,
        *,
        is_container: bool,
    ) -> None:
        super().__init__(
            parent,
            parent.version,
            parent.flags,
            is_container=is_container,
        )

        self.handler_box = handler_box
        self.optional_boxes = optional_boxes

    @staticmethod
    def __read_impl(
        io: BoundedIO,
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

        if parent.version != 0:
            msg = f"Invalid meta version: {parent.version}"
            raise RuntimeError(msg)

        handler_box = HandlerBox.read(parent.payload_io(io))
        parent.span.add_header(handler_box.span.total.size)

        optional_boxes = OptionalMetaBoxes.empty()

        # peek the next box, if it's an optional box, we read that, otherwise we are at the end and read the last box array
        while True:
            simple_box: MP4Box = MP4Box.read_mp4_box(
                parent.payload_io(io),
            )

            if simple_box.type not in META_OPTIONAL_BOXES:
                break
            match simple_box.type:
                case MetaOptionalBoxes.PITM:
                    if optional_boxes.pitm is not None:
                        msg = "Duplicate PrimaryItemBox in MetaBox"
                        raise RuntimeError(msg)

                    pitm_box = PrimaryItemBox.read_from_parent(
                        simple_box.payload_io(io),
                        simple_box,
                    )
                    parent.span.add_header(pitm_box.span.total.size)
                    optional_boxes.pitm = pitm_box
                case _:
                    msg = f"MetaBox: parsing of optional box {simple_box.type} not implemented yet"
                    raise RuntimeError(msg)

        # treat the box as container, if there is some payload left
        is_container = parent.span.payload_span.size != 0

        return MetaBox(parent, handler_box, optional_boxes, is_container=is_container)

    @staticmethod
    def read(io: BoundedIO) -> "MetaBox":
        box = MP4FullBox.read_mp4_full_box(io)
        return MetaBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "MetaBox":
        box: MP4FullBox
        if isinstance(parent, MP4FullBox):
            box = parent
        else:
            box = MP4FullBox.read_from_parent_mp4_full_box(io, parent)
        return MetaBox.__read_impl(io, box)

    @staticmethod
    def write_to_buffer(
        handler: MetaBoxHandlerInformation,
        primary_item_info: Optional[PrimaryItemInformation],
        data: bytes,
    ) -> bytes:
        buf = BytesIO()

        handler_bytes = HandlerBox.write_to_buffer(
            handler_type=handler.handler_type,
            name=handler.name,
            reserved=handler.reserved,
        )

        buf.write(handler_bytes)

        if primary_item_info is not None:
            primary_item_bytes = PrimaryItemBox.write_to_buffer(
                primary_item_info.item_id,
            )
            buf.write(primary_item_bytes)

        buf.write(data)

        final_data = buf.getvalue()

        return MP4FullBox.write_to_buffer_mp4_full_box(
            META_ATOM_NAME,
            0,
            b"\x00" * 3,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<MetaBox parent: {MP4FullBox.__str__(self)} handler_box: {self.handler_box}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class AppleItunesItemList(MP4Box, FinalMP4Box):
    def __init__(self: Self, parent: MP4Box) -> None:
        super().__init__(parent.type, parent.span, is_container=True)

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: MP4Box,
    ) -> "AppleItunesItemList":
        # spec: https://developer.apple.com/documentation/quicktime-file-format/metadata_item_list_atom
        # Apple Itunes Item List structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class AppleItunesItemList extends Box(
        #     'ilst'
        #     ) {
        # }

        with io.rw_ctx(force_entire_read=True) as f:
            f.skip(parent.span.payload_span.size)

        return AppleItunesItemList(parent)

    @staticmethod
    def read(io: BoundedIO) -> "AppleItunesItemList":
        box = MP4Box.read_mp4_box(io)
        return AppleItunesItemList.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(io: BoundedIO, parent: MP4Box) -> "AppleItunesItemList":
        return AppleItunesItemList.__read_impl(io, parent)

    @staticmethod
    def write_to_buffer(
        data: bytes,
    ) -> bytes:
        return MP4Box.write_to_buffer_mp4_box(ILST_ATOM_NAME, data)

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemList parent: {MP4Box.__str__(self)}>"

    def __repr__(self: Self) -> str:
        return str(self)


AppleItunesItemDataContent = str | int | UUID


# see: https://developer.apple.com/documentation/quicktime-file-format/well-known_types
class AppleItunesItemDataType(Enum):
    # NOTE: only some are implemented here
    RESERVED = 0
    IMPLICIT = RESERVED

    UTF8 = 1
    UTF16 = 2

    # see https://taglib.org/api/namespaceTagLib_1_1MP4.html#a86c3870b24b4cdb2887d3e47f5f9a39b
    UUID = 8

    JPEG = 13
    PNG = 14

    BE_SIGNED_INTEGER_VAR = 21
    INTEGER = BE_SIGNED_INTEGER_VAR

    BE_UNSIGNED_INTEGER_VAR = 22


@final
@decorate_class(slots=True)
class AppleItunesItemDataBox(MP4FullBox, FinalMP4Box):
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
    def __decode_value_impl(
        type_indicator: AppleItunesItemDataType,
        value: bytes,
    ) -> AppleItunesItemDataContent:

        match type_indicator:
            case AppleItunesItemDataType.IMPLICIT | AppleItunesItemDataType.RESERVED:
                msg = "No implicit type allowed here!"
                raise RuntimeError(msg)
            case AppleItunesItemDataType.UTF8:
                return value.decode("utf-8")
            case AppleItunesItemDataType.UTF16:
                return value.decode("utf-16")
            case AppleItunesItemDataType.UUID:
                return uuid_from_bytes(ISOM_BYTE_ORDER, value)
            case (
                AppleItunesItemDataType.BE_SIGNED_INTEGER_VAR
                | AppleItunesItemDataType.INTEGER
            ):
                if len(value) not in [1, 2, 4, 8]:
                    msg = f"Invalid integer conversion length: {len(value)}"
                    raise RuntimeError(msg)

                return int.from_bytes(value, byteorder="big", signed=True)
            case AppleItunesItemDataType.BE_UNSIGNED_INTEGER_VAR:
                if len(value) not in [1, 2, 4, 8]:
                    msg = f"Invalid integer conversion length: {len(value)}"
                    raise RuntimeError(msg)

                return int.from_bytes(value, byteorder="big", signed=False)
            case _:
                msg = f"Not implemented type_indicator conversion: {type_indicator}"
                raise RuntimeError(msg)

    @staticmethod
    def __decode_value(
        typ: ISOMAtomName,
        type_indicator: int,
        value: bytes,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> AppleItunesItemDataContent:

        if type_indicator == AppleItunesItemDataType.IMPLICIT.value:
            if expected_type is None:
                msg = _(
                    "No implicit type known for value: {value!r}of type {type}"  # noqa: COM812
                ).format(value=value, type=typ)
                raise RuntimeError(msg)

            return AppleItunesItemDataBox.__decode_value_impl(expected_type, value)

        return AppleItunesItemDataBox.__decode_value_impl(
            AppleItunesItemDataType(type_indicator),
            value,
        )

    @staticmethod
    def __encode_value_impl(
        type_indicator: int,
        value: AppleItunesItemDataContent,
    ) -> Result[bytes, str]:
        match type_indicator:
            case AppleItunesItemDataType.IMPLICIT.value:
                return Err("Can't encode implict value")
            case AppleItunesItemDataType.UTF8.value:
                if isinstance(value, str):
                    return Ok(value.encode("utf-8"))

                return Err(f"Can't encode value of type {type(value)} with UTF-8")
            case AppleItunesItemDataType.UTF16.value:
                if isinstance(value, str):
                    return Ok(value.encode("utf-16"))

                return Err(f"Can't encode value of type {type(value)} with UTF-16")
            case AppleItunesItemDataType.UUID.value:
                if isinstance(value, UUID):
                    return Ok(uuid_to_bytes(ISOM_BYTE_ORDER, value))

                return Err(f"Can't encode value of type {type(value)} ass UUID")
            case AppleItunesItemDataType.BE_SIGNED_INTEGER_VAR.value:
                if isinstance(value, int):
                    if value < -(1 << 63):
                        return Err(
                            f"Can't encode int value as signed integer, value too big for 8 bytes: {value}",
                        )

                    if value > (1 << 63) - 1:
                        return Err(
                            f"Can't encode int value as signed integer, value too big for 8 bytes: {value}",
                        )

                    length_s: int
                    if value < 0:
                        if value >= -(1 << 7):
                            length_s = 1
                        elif value >= -(1 << 15):
                            length_s = 2
                        elif value >= -(1 << 31):
                            length_s = 4
                        else:
                            length_s = 8
                    else:  # noqa: PLR5501
                        if value <= (1 << 7) - 1:
                            length_s = 1
                        elif value <= (1 << 15) - 1:
                            length_s = 2
                        elif value <= (1 << 31) - 1:
                            length_s = 4
                        else:
                            length_s = 8

                    return Ok(
                        value.to_bytes(length=length_s, byteorder="big", signed=True),
                    )

                return Err(f"Can't encode value of type {type(value)} as int")
            case AppleItunesItemDataType.BE_UNSIGNED_INTEGER_VAR.value:
                if isinstance(value, int):
                    if value < 0:
                        return Err(
                            f"Can't encode negative int value as unsigned integer: {value}",
                        )

                    if value > (1 << 64) - 1:
                        return Err(
                            f"Can't encode int value as unsigned integer, value too big for 8 bytes: {value}",
                        )

                    length_u: int
                    if value <= 0xFF:
                        length_u = 1
                    elif value <= 0xFFFF:
                        length_u = 2
                    elif value <= (1 << 32) - 1:
                        length_u = 4
                    else:
                        length_u = 8

                    return Ok(
                        value.to_bytes(length=length_u, byteorder="big", signed=False),
                    )

                return Err(f"Can't encode value of type {type(value)} as int")
            case _:
                msg = f"Not implemented type_indicator conversion: {type_indicator}"
                raise RuntimeError(msg)

    @staticmethod
    def __encode_value(
        type_indicator: AppleItunesItemDataType,
        value: AppleItunesItemDataContent,
    ) -> bytes:
        encoded = AppleItunesItemDataBox.__encode_value_impl(
            type_indicator.value,
            value,
        )
        if encoded.err():
            msg = f"Encoding error: {encoded.as_err()}"
            raise RuntimeError(msg)

        return encoded.as_ok()

    @staticmethod
    def can_encode_value(
        type_indicator: AppleItunesItemDataType,
        value: AppleItunesItemDataContent,
    ) -> Optional[str]:
        encoded = AppleItunesItemDataBox.__encode_value_impl(
            type_indicator.value,
            value,
        )

        return encoded.err_or(None)

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: MP4FullBox,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> "AppleItunesItemDataBox":
        # spec: https://developer.apple.com/documentation/quicktime-file-format/data_atom
        # Apple Itunes Item Data Box structure:
        # box     | <full box size> bytes | parent full box
        # ... data

        # aligned(8) class AppleItunesItemDataBox extends FullBox(
        #     'data'
        #     ) {
        # }

        with io.rw_ctx(force_entire_read=True) as f:

            # see: https://developer.apple.com/documentation/quicktime-file-format/type_indicator
            if parent.version != 0:
                msg = f"AppleItunesItemDataBox: type indicator byte 0 has to be 0, but was {parent.version} (it is the FullBox version field)"
                raise ValueError(msg)

            type_indicator = Unpacker.unpack_one(
                ISOM_BYTE_ORDER,
                UnsignedInt(),
                b"\x00" + parent.flags,
            )

            locale_indicator_raw = f.read(4)

            # see: https://developer.apple.com/documentation/quicktime-file-format/locale_indicator
            locale_indicator = Unpacker.unpack_one(
                ISOM_BYTE_ORDER,
                UnsignedInt(),
                locale_indicator_raw,
            )

            # omitting dynamic sized string "value"
            fixed_header_size = 4

            value_size = parent.span.payload_span.size - fixed_header_size

            value_raw = f.read(value_size)

            value = AppleItunesItemDataBox.__decode_value(
                parent.type,
                type_indicator,
                value_raw,
                expected_type,
            )

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return AppleItunesItemDataBox(
                parent,
                type_indicator,
                locale_indicator,
                value,
            )

    @staticmethod
    def read(
        io: BoundedIO,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> "AppleItunesItemDataBox":
        box: MP4FullBox = MP4FullBox.read_mp4_full_box(io)
        return AppleItunesItemDataBox.__read_impl(
            box.payload_io(io),
            box,
            expected_type,
        )

    @staticmethod
    def read_checked(
        io: BoundedIO,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> "AppleItunesItemDataBox":
        box: MP4FullBox = MP4FullBox.read_mp4_full_box(io)
        if box.type != DATA_ATOM_NAME:
            msg = f"Invalid AppleItunesItemDataBox tag: {box.type}"
            raise RuntimeError(msg)

        return AppleItunesItemDataBox.__read_impl(
            box.payload_io(io),
            box,
            expected_type,
        )

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: MP4Box,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> "AppleItunesItemDataBox":
        box: MP4FullBox
        if isinstance(parent, MP4FullBox):
            box = parent
        else:
            box = MP4FullBox.read_from_parent_mp4_full_box(io, parent)
        return AppleItunesItemDataBox.__read_impl(
            box.payload_io(io),
            box,
            expected_type,
        )

    @staticmethod
    def write_to_buffer(
        type_indicator: AppleItunesItemDataType,
        value: AppleItunesItemDataContent,
    ) -> bytes:
        type_indicator_bytes_raw = Packer.pack_one(
            ISOM_BYTE_ORDER,
            UnsignedInt(),
            type_indicator.value,
            4,
        )
        if type_indicator_bytes_raw[0] != 0:
            msg = (
                f"Implementation error: first byet not 0: {type_indicator_bytes_raw[0]}"
            )
            raise RuntimeError(msg)

        type_indicator_bytes = type_indicator_bytes_raw[1:4]

        buf = BytesIO()

        # see: https://developer.apple.com/documentation/quicktime-file-format/locale_indicator
        locale_indicator_bytes = Packer.pack_one(
            ISOM_BYTE_ORDER,
            UnsignedInt(),
            0,
            4,
        )

        buf.write(locale_indicator_bytes)

        value_bytes = AppleItunesItemDataBox.__encode_value(type_indicator, value)

        buf.write(value_bytes)

        final_data = buf.getvalue()

        return MP4FullBox.write_to_buffer_mp4_full_box(
            DATA_ATOM_NAME,
            0,
            type_indicator_bytes,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemDataBox parent: {MP4FullBox.__str__(self)} type_indicator: {self.type_indicator} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class AppleItunesItemMeanBox(MP4FullBox, FinalMP4Box):
    value: str

    def __init__(
        self: Self,
        parent: MP4FullBox,
        value: str,
    ) -> None:
        super().__init__(parent, parent.version, parent.flags, is_container=False)

        self.value = value

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: MP4FullBox,
    ) -> "AppleItunesItemMeanBox":
        # spec: N/A
        # Apple Itunes Item Mean Box structure:
        # box     | <full box size> bytes | parent full box
        # ... data

        # aligned(8) class AppleItunesItemMeanBox extends FullBox(
        #     'mean'
        #     ) {
        # }

        with io.r_ctx(force_entire_read=True) as f:

            if parent.version != 0:
                msg = f"Invalid AppleItunesItemMeanBox version: {parent.version}"
                raise RuntimeError(msg)

            data = f.read(parent.span.payload_span.size)

            value = data.decode()

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return AppleItunesItemMeanBox(parent, value)

    @staticmethod
    def read(
        io: BoundedIO,
    ) -> "AppleItunesItemMeanBox":
        box: MP4FullBox = MP4FullBox.read_mp4_full_box(io)
        return AppleItunesItemMeanBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_checked(
        io: BoundedIO,
    ) -> "AppleItunesItemMeanBox":
        box: MP4FullBox = MP4FullBox.read_mp4_full_box(io)
        if box.type != MEAN_ATOM_NAME:
            msg = f"Invalid AppleItunesItemMeanBox tag: {box.type}"
            raise RuntimeError(msg)

        return AppleItunesItemMeanBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: MP4Box,
    ) -> "AppleItunesItemMeanBox":
        box: MP4FullBox
        if isinstance(parent, MP4FullBox):
            box = parent
        else:
            box = MP4FullBox.read_from_parent_mp4_full_box(io, parent)
        return AppleItunesItemMeanBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def write_to_buffer(
        value: str,
    ) -> bytes:
        buf = BytesIO()

        buf.write(value.encode())

        final_data = buf.getvalue()

        return MP4FullBox.write_to_buffer_mp4_full_box(
            MEAN_ATOM_NAME,
            0,
            b"\x00" * 3,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemMeanBox parent: {MP4FullBox.__str__(self)} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class AppleItunesItemNameBox(MP4FullBox, FinalMP4Box):
    value: str

    def __init__(
        self: Self,
        parent: MP4FullBox,
        value: str,
    ) -> None:
        super().__init__(parent, parent.version, parent.flags, is_container=False)

        self.value = value

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: MP4FullBox,
    ) -> "AppleItunesItemNameBox":
        # spec: https://developer.apple.com/documentation/quicktime-file-format/name_atom
        # Apple Itunes Item Name Box structure:
        # box     | <full box size> bytes | parent full box
        # ... data

        # aligned(8) class AppleItunesItemNameBox extends FullBox(
        #     'name'
        #     ) {
        # }

        with io.r_ctx(force_entire_read=True) as f:

            if parent.version != 0:
                msg = f"Invalid AppleItunesItemNameBox version: {parent.version}"
                raise RuntimeError(msg)

            data = f.read(parent.span.payload_span.size)

            value = data.decode()

            parent.span.add_header(parent.span.payload_span.size)

            if parent.span.payload_span.size != 0:
                msg = f"Expected empty payload but got:{parent.span.payload_span.size}"
                raise RuntimeError(msg)

            return AppleItunesItemNameBox(parent, value)

    @staticmethod
    def read(
        io: BoundedIO,
    ) -> "AppleItunesItemNameBox":
        box: MP4FullBox = MP4FullBox.read_mp4_full_box(io)
        return AppleItunesItemNameBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_checked(
        io: BoundedIO,
    ) -> "AppleItunesItemNameBox":
        box: MP4FullBox = MP4FullBox.read_mp4_full_box(io)
        if box.type != NAME_ATOM_NAME:
            msg = f"Invalid AppleItunesItemNameBox tag: {box.type}"
            raise RuntimeError(msg)

        return AppleItunesItemNameBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: MP4Box,
    ) -> "AppleItunesItemNameBox":
        box: MP4FullBox
        if isinstance(parent, MP4FullBox):
            box = parent
        else:
            box = MP4FullBox.read_from_parent_mp4_full_box(io, parent)
        return AppleItunesItemNameBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def write_to_buffer(
        value: str,
    ) -> bytes:
        buf = BytesIO()

        buf.write(value.encode())

        final_data = buf.getvalue()

        return MP4FullBox.write_to_buffer_mp4_full_box(
            NAME_ATOM_NAME,
            0,
            b"\x00" * 3,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemNameBox parent: {MP4FullBox.__str__(self)} value: {self.value}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class AppleItunesItemBox(MP4Box, FinalMP4Box):
    data: AppleItunesItemDataBox

    def __init__(self: Self, parent: MP4Box, data: AppleItunesItemDataBox) -> None:
        super().__init__(parent.type, parent.span, is_container=False)

        self.data = data

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: MP4Box,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> "AppleItunesItemBox":
        # spec: https://developer.apple.com/documentation/quicktime-file-format/value_atom
        # Apple Itunes Item Box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class AppleItunesItemBox extends Box(
        #     '<any>' // any type specified, see below
        #     ) {
        # }

        data = AppleItunesItemDataBox.read_checked(
            parent.payload_io(io),
            expected_type,
        )

        parent.span.add_header(data.span.total.size)

        if parent.span.payload_span.size != 0:
            msg = f"AppleItunesItemBox isn't fully filled by the data box: {parent.span.payload_span.size} leftover data"
            raise RuntimeError(msg)

        return AppleItunesItemBox(parent, data)

    @staticmethod
    def read(
        io: BoundedIO,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> "AppleItunesItemBox":
        box = MP4Box.read_mp4_box(io)
        return AppleItunesItemBox.__read_impl(box.payload_io(io), box, expected_type)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: MP4Box,
        expected_type: Optional[AppleItunesItemDataType],
    ) -> "AppleItunesItemBox":
        return AppleItunesItemBox.__read_impl(io, parent, expected_type)

    @staticmethod
    def write_to_buffer(
        name: ISOMAtomName,
        type_indicator: AppleItunesItemDataType,
        value: AppleItunesItemDataContent,
    ) -> bytes:
        buf = BytesIO()

        data_bytes = AppleItunesItemDataBox.write_to_buffer(
            type_indicator,
            value,
        )

        buf.write(data_bytes)
        final_data = buf.getvalue()

        return MP4Box.write_to_buffer_mp4_box(
            name,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemBox parent: {MP4Box.__str__(self)} data: {self.data}>"

    def __repr__(self: Self) -> str:
        return str(self)


@final
@decorate_class(slots=True)
class AppleItunesItemFreeformBox(MP4Box, FinalMP4Box):
    mean: AppleItunesItemMeanBox
    name: AppleItunesItemNameBox
    data: AppleItunesItemDataBox

    def __init__(
        self: Self,
        parent: MP4Box,
        mean: AppleItunesItemMeanBox,
        name: AppleItunesItemNameBox,
        data: AppleItunesItemDataBox,
    ) -> None:
        super().__init__(parent.type, parent.span, is_container=False)

        self.mean = mean
        self.name = name
        self.data = data

    @staticmethod
    def __read_impl(
        io: BoundedIO,
        parent: MP4Box,
    ) -> "AppleItunesItemFreeformBox":
        # spec: N/A
        # Apple Itunes Item Freeform Box structure:
        # box     | <box size> bytes | parent box

        # aligned(8) class AppleItunesItemBox extends Box(
        #     '----'
        #     ) {
        # }

        mean = AppleItunesItemMeanBox.read_checked(
            parent.payload_io(io),
        )

        parent.span.add_header(mean.span.total.size)

        name: AppleItunesItemNameBox = AppleItunesItemNameBox.read_checked(
            parent.payload_io(io),
        )

        parent.span.add_header(name.span.total.size)

        data = AppleItunesItemDataBox.read_checked(
            parent.payload_io(io),
            None,
        )

        parent.span.add_header(data.span.total.size)

        if parent.span.payload_span.size != 0:
            msg = f"AppleItunesItemBox isn't fully filled by the data box: {parent.span.payload_span.size} leftover data"
            raise RuntimeError(msg)

        return AppleItunesItemFreeformBox(parent, mean, name, data)

    @staticmethod
    def read(
        io: BoundedIO,
    ) -> "AppleItunesItemFreeformBox":
        box = MP4Box.read_mp4_box(io)
        return AppleItunesItemFreeformBox.__read_impl(box.payload_io(io), box)

    @staticmethod
    def read_from_parent(
        io: BoundedIO,
        parent: MP4Box,
    ) -> "AppleItunesItemFreeformBox":
        return AppleItunesItemFreeformBox.__read_impl(io, parent)

    @staticmethod
    def write_to_buffer(
        mean: str,
        name: str,
        type_indicator: AppleItunesItemDataType,
        value: AppleItunesItemDataContent,
    ) -> bytes:
        buf = BytesIO()

        mean_bytes = AppleItunesItemMeanBox.write_to_buffer(mean)

        buf.write(mean_bytes)

        name_bytes = AppleItunesItemNameBox.write_to_buffer(name)

        buf.write(name_bytes)

        data_bytes = AppleItunesItemDataBox.write_to_buffer(type_indicator, value)

        buf.write(data_bytes)

        final_data = buf.getvalue()

        return MP4Box.write_to_buffer_mp4_box(
            AppleItunesItemBoxAtomFreeform,
            final_data,
        )

    def __str__(self: Self) -> str:
        return f"<AppleItunesItemFreeformBox parent: {MP4Box.__str__(self)} mean: {self.mean} name: {self.name} data: {self.data}>"

    def __repr__(self: Self) -> str:
        return str(self)


# ruff: enable[ERA001]


@dataclass(slots=True, repr=True)
class ApplItunesTagsData:
    type: AppleItunesItemDataType
    value: AppleItunesItemDataContent

    @staticmethod
    def from_data_box(box: AppleItunesItemDataBox) -> "ApplItunesTagsData":
        return ApplItunesTagsData(
            AppleItunesItemDataType(box.type_indicator),
            box.value,
        )


@dataclass(slots=True, repr=True)
class ApplItunesTags:
    key: ISOMAtomName | AppleItunesFreeformKey
    data: ApplItunesTagsData

    @staticmethod
    def validate_init(
        key: ISOMAtomName | AppleItunesFreeformKey,
        data: ApplItunesTagsData,
    ) -> "ApplItunesTags":
        encode_res = AppleItunesItemDataBox.can_encode_value(data.type, data.value)
        if encode_res is not None:
            msg = f"Atom {key} not encodable: {encode_res}"
            raise RuntimeError(msg)

        return ApplItunesTags(key=key, data=data)

    @staticmethod
    def from_known_atom(
        name: ISOMAtomName,
        value: AppleItunesItemDataContent,
    ) -> "ApplItunesTags":
        data_type = AppleItunesItemBoxAtoms.get(name, None)  # noqa: SIM910
        if data_type is None:
            msg = f"Atom name not known: {name}"
            raise RuntimeError(msg)

        return ApplItunesTags.validate_init(
            key=name,
            data=ApplItunesTagsData(data_type, value),
        )


@decorate_class(slots=True)
class AppleItunesMetaBoxBuilder:
    __tags: dict[str, ApplItunesTags]

    def __init__(self: Self) -> None:
        self.__tags = {}

    @staticmethod
    def _key_str_impl(key: ISOMAtomName | AppleItunesFreeformKey) -> str:
        # note: this is never serialized, it is only to detect duplicates in internal regeneration from an old meta box, so this doesn#t have to match the serialization behavior, but it's close, as the string is unique then

        if isinstance(key, ISOMAtomName):
            return key.value.decode("latin-1")

        if isinstance(key, AppleItunesFreeformKey):
            return f"----:{key.mean}:{key.name}"

        assert_never(key)

    def add_tag(
        self: Self,
        tag: ApplItunesTags,
        duplicate_behavior: Literal["overwrite", "error", "ignore"],
    ) -> None:

        key = AppleItunesMetaBoxBuilder._key_str_impl(tag.key)

        if self.__tags.get(key, None) is not None:
            if duplicate_behavior == "error":
                msg = f"Trying to add duplicate tag key: {key}"
                raise RuntimeError(msg)

            if duplicate_behavior == "overwrite":
                self.__tags[key] = tag
            elif duplicate_behavior == "ignore":
                pass
            else:
                assert_never(duplicate_behavior)
        else:
            self.__tags[key] = tag

    @staticmethod
    def __render_tag_impl(tag: ApplItunesTags) -> bytes:
        if isinstance(tag.key, ISOMAtomName):
            return AppleItunesItemBox.write_to_buffer(
                tag.key,
                tag.data.type,
                tag.data.value,
            )

        if isinstance(tag.key, AppleItunesFreeformKey):
            return AppleItunesItemFreeformBox.write_to_buffer(
                tag.key.mean,
                tag.key.name,
                tag.data.type,
                tag.data.value,
            )

        assert_never(tag.key)

    def build(self: Self) -> bytes:

        tags_rendered: bytes = b""

        for tag in self.__tags.values():
            tags_rendered += self.__render_tag_impl(tag)

        ilst_bytes = AppleItunesItemList.write_to_buffer(tags_rendered)

        return MetaBox.write_to_buffer(
            MetaBoxHandlerInformation(
                handler_type=MDIR_ATOM_NAME,
                name="",
                reserved=b"appl" + (b"\x00" * 8),
            ),
            PrimaryItemInformation(item_id=META_BOX_VIDEO_LANGUAGE_DETECTION_ID),
            ilst_bytes,
        )


AppleItunesItemBoxAtomFreeform = ISOMAtomName(b"----")

AppleItunesItemBoxAtoms: dict[ISOMAtomName, Optional[AppleItunesItemDataType]] = {
    ISOMAtomName(b"trkn"): None,
    ISOMAtomName(b"disk"): None,
    ISOMAtomName(value=b"gnre"): None,
    ISOMAtomName(b"plID"): None,
    ISOMAtomName(b"cnID"): None,
    ISOMAtomName(b"geID"): None,
    ISOMAtomName(b"atID"): None,
    ISOMAtomName(b"sfID"): None,
    ISOMAtomName(b"cmID"): None,
    ISOMAtomName(b"akID"): None,
    ISOMAtomName(b"tvsh"): AppleItunesItemDataType.UTF8,  # TV Show, show name
    ISOMAtomName(b"tven"): AppleItunesItemDataType.INTEGER,  # TV Episode id
    ISOMAtomName(b"tvsn"): AppleItunesItemDataType.INTEGER,  # -- TV Season
    ISOMAtomName(b"tves"): AppleItunesItemDataType.INTEGER,  # -- TV Episode
    ISOMAtomName(b"tmpo"): None,
    ISOMAtomName(b"\xa9mvi"): None,
    ISOMAtomName(b"\xa9mvc"): None,
    ISOMAtomName(b"cpil"): None,
    ISOMAtomName(b"pgap"): None,
    ISOMAtomName(b"pcst"): None,
    ISOMAtomName(b"shwm"): None,
    ISOMAtomName(b"stik"): AppleItunesItemDataType.INTEGER,  # -- MediaKind
    ISOMAtomName(b"hdvd"): None,
    ISOMAtomName(b"rtng"): None,
    ISOMAtomName(b"covr"): None,
    ISOMAtomName(b"purl"): None,
    ISOMAtomName(b"egid"): None,
    ISOMAtomName(b"\xa9nam"): None,
    ISOMAtomName(b"\xa9alb"): None,
    ISOMAtomName(b"\xa9ART"): None,
    ISOMAtomName(b"aART"): None,
    ISOMAtomName(b"\xa9wrt"): None,
    ISOMAtomName(b"\xa9day"): None,
    ISOMAtomName(b"\xa9cmt"): AppleItunesItemDataType.UTF8,  # -- comment
    ISOMAtomName(b"desc"): None,
    ISOMAtomName(b"purd"): None,
    ISOMAtomName(b"\xa9grp"): None,
    ISOMAtomName(b"\xa9gen"): None,
    ISOMAtomName(b"\xa9lyr"): None,
    ISOMAtomName(b"catg"): None,
    ISOMAtomName(b"keyw"): None,
    ISOMAtomName(b"\xa9too"): AppleItunesItemDataType.UTF8,  # -- encoded by, tool
    ISOMAtomName(b"cprt"): None,
    ISOMAtomName(b"soal"): None,
    ISOMAtomName(b"soaa"): None,
    ISOMAtomName(b"soar"): None,
    ISOMAtomName(b"sonm"): None,
    ISOMAtomName(b"soco"): None,
    ISOMAtomName(b"sosn"): None,
}


@decorate_class(slots=True)
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
    AppleItunesItemBoxAtomFreeform = AppleItunesItemBoxAtomFreeform

    DATA = DATA_ATOM_NAME
    MEAN = MEAN_ATOM_NAME
    NAME = NAME_ATOM_NAME


def read_box(io: BoundedIO) -> MP4Box:
    box = MP4Box.read_mp4_box(io)

    match box.type:
        case SupportedBoxes.MDHD:
            return MediaHeaderBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.MDIA:
            return MediaBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.HDLR:
            return HandlerBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.TRAK:
            return TrackBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.MOOV:
            return MovieBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.FTYP:
            return FileTypeBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.FREE:
            return FreeSpaceBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.SKIP:
            return FreeSpaceBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.UDTA:
            return UserDataBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.META:
            return MetaBox.read_from_parent(box.payload_io(io), box)
        case SupportedBoxes.ILST:
            return AppleItunesItemList.read_from_parent(box.payload_io(io), box)
        case _ if box.type in SupportedBoxes.AppleItunesItemBox:
            value = SupportedBoxes.AppleItunesItemBox[box.type]
            return AppleItunesItemBox.read_from_parent(box.payload_io(io), box, value)
        case SupportedBoxes.AppleItunesItemBoxAtomFreeform:
            return AppleItunesItemFreeformBox.read_from_parent(box.payload_io(io), box)
        case _:
            return box


def mp4_iter_boxes(
    io_base: BinaryIO,
    span: SimpleSpan,
) -> Generator[MP4Box]:
    pos = span.start

    while pos < span.end:
        io = BoundedIO.get_new(io_base, SimpleSpan(pos, span.end - pos))
        box = read_box(io)

        if pos + box.span.total.size > span.end:
            msg = f"Box {box.type!r} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield box
        pos += box.span.total.size


def mp4_iter_boxes_io(io: BoundedIO) -> Generator[MP4Box]:
    pos = io.span.start
    end = io.span.end

    while pos < end:
        new_io = io.new_span_io(SimpleSpan(pos, end - pos))
        box = read_box(new_io)

        if pos + box.span.total.size > end:
            msg = f"Box {box.type!r} at {pos} extends past parent boundary"
            raise RuntimeError(msg)

        yield box
        pos += box.span.total.size


def find_mdhd_boxes_with_type(
    f: BinaryIO,
    types: list[ISOMAtomName],
) -> Generator[MediaHeaderBox]:
    f.seek(0, 2)
    filesize = f.tell()

    stack: list[tuple[SimpleSpan, list[ISOMAtomName]]] = [(SimpleSpan(0, filesize), [])]

    while stack:
        span, path = stack.pop()

        for box in mp4_iter_boxes(f, span):

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
                stack.append((box.span.payload_span, [*path, box.type]))


def is_mp4_file(
    f: BinaryIO,
) -> Optional[str]:
    f.seek(0)

    try:
        f.seek(0, 2)
        filesize = f.tell()
        first_box = read_box(BoundedIO.get_new(f, SimpleSpan(0, filesize)))

        if not isinstance(first_box, FileTypeBox):
            return _("Not a valid ISOM / MP4 file")

        if first_box.major_brand not in [b"isom", b"mp42"]:
            return _(
                "ISOM/MP42 file has valid box, but invalid major_brand: {major_brand!s}"  # noqa: COM812
            ).format(major_brand=first_box.major_brand)

        f.seek(0)
    except (RuntimeError, ValueError) as err:
        return str(err)
    return None


# see also: https://mpeggroup.github.io/FileFormatConformance/?query=%3D%22pitm%22
META_BOX_VIDEO_LANGUAGE_DETECTION_ID: int = 0x41DC

META_PADDING_SIZE = 512
SIZE_OF_FREE_BOX_HEADER = 8

ReadMetaBoxValues = list[ApplItunesTags]

MetaValues = Result[Optional[ReadMetaBoxValues], str]


@dataclass(slots=True, repr=True)
class ReadMetadataImpl:
    metadata: SerializableDict
    uuid: Optional[UUID]


@decorate_class(slots=True)
class MP4MetadataHandler:
    __uuid_box: Optional[UUIDExtensionBox]
    __meta_values: MetaValues
    __our_boxes: list[MP4Box]

    def __init__(
        self: Self,
        uuid_box: Optional[UUIDExtensionBox],
        meta_values: MetaValues,
        our_boxes: list[MP4Box],
    ) -> None:
        self.__uuid_box = uuid_box
        self.__meta_values = meta_values
        self.__our_boxes = our_boxes

    def __write_metadata_toplevel_meta(
        self: Self,
        f: BinaryIO,
        tags: MetadataTags,
    ) -> None:
        if self.__meta_values.err():
            # ignore this and write no top level meta box
            pass
        else:
            meta_values = self.__meta_values.as_ok()

            # NOTE: using top level meta box

            # meta:
            # location: file , amount: 0 or 1

            meta_box = AppleItunesMetaBoxBuilder()

            if meta_values is not None:
                # restore the old tags
                for meta_value in meta_values:
                    meta_box.add_tag(
                        meta_value,
                        duplicate_behavior="error",
                    )

        # add or overwrite tags, if not present, so that the new data gets written all the time, except uuid, that is never replaced
        meta_box.add_tag(
            ApplItunesTags.from_known_atom(
                ISOMAtomName(b"\xa9cmt"),
                tags.comment,
            ),
            duplicate_behavior="overwrite",
        )

        for key, value in tags.metadata.items():
            value_str = json.dumps(value)
            meta_box.add_tag(
                ApplItunesTags.validate_init(
                    key=TaggerDomain.get_freeform(key),
                    data=ApplItunesTagsData(
                        type=AppleItunesItemDataType.UTF8,
                        value=value_str,
                    ),
                ),
                duplicate_behavior="overwrite",
            )

        # Note, these are not necessary in sync, which is bad, but that should never happen
        meta_box.add_tag(
            ApplItunesTags.validate_init(
                key=TaggerDomain.UUID_RAW_KEY_FREEFORM,
                data=ApplItunesTagsData(
                    type=AppleItunesItemDataType.UUID,
                    value=tags.uuid,
                ),
            ),
            "ignore",
        )

        meta_box.add_tag(
            ApplItunesTags.validate_init(
                key=TaggerDomain.UUID_HEX_KEY_FREEFORM,
                data=ApplItunesTagsData(
                    type=AppleItunesItemDataType.UTF8,
                    value=uuid_to_str(tags.uuid),
                ),
            ),
            "ignore",
        )

        buffer = meta_box.build()

        f.seek(0, 2)
        f.write(buffer)

        # write padding
        padding_size = META_PADDING_SIZE - (len(buffer) % META_PADDING_SIZE)

        if padding_size < SIZE_OF_FREE_BOX_HEADER:
            padding_size = META_PADDING_SIZE + padding_size - SIZE_OF_FREE_BOX_HEADER
        else:
            padding_size = padding_size - SIZE_OF_FREE_BOX_HEADER

        if padding_size < 0:
            msg = f"Implementation error: padding size negative: {padding_size}"
            raise RuntimeError(msg)

        buffer = FreeSpaceBox.write_to_buffer(data=b"\x00" * padding_size)
        f.write(buffer)

        f.flush()

    def __write_metadata_custom(
        self: Self,
        f: BinaryIO,
        tags: MetadataTags,
    ) -> None:
        # note: can write 0 or more free space or user extension boxes, and there both allowed everywhere

        f.seek(0, 2)

        if self.__uuid_box is not None:
            buffer = UUIDExtensionBox.write_to_buffer(self.__uuid_box.uuid)
            f.write(buffer)
        else:
            buffer = UUIDExtensionBox.write_to_buffer(tags.uuid)

            f.write(buffer)

        metadata_dicts: list[SerializableDict] = [
            {
                "comment": tags.comment,
                "metadata": tags.metadata,
            },
        ]

        for mdt in metadata_dicts:
            buffer = JsonExtensionBox.write_to_buffer(mdt)

            f.write(buffer)

        f.flush()

    def write_new_metadata(self: Self, f: BinaryIO, tags: MetadataTags) -> None:

        # delete old metadata, this can be done, as the top level consists of multiple boxes, and no sizes need to be changed
        if len(self.__our_boxes) != 0:
            f.truncate(self.__our_boxes[0].span.total.start)

        f.seek(0, 2)

        self.__write_metadata_toplevel_meta(f, tags)

        self.__write_metadata_custom(f, tags)

        f.flush()

    def __read_metadata_custom(
        self: Self,
        boxes: list[JsonExtensionBox | UUIDExtensionBox],
    ) -> ReadMetadataImpl:
        metadata_result: ReadMetadataImpl = ReadMetadataImpl(
            {},
            None if self.__uuid_box is None else self.__uuid_box.uuid,
        )

        for box in boxes:
            if isinstance(box, JsonExtensionBox):
                metadata_result.metadata = merge_dicts(
                    metadata_result.metadata,
                    box.data,
                    "error",
                )
            elif isinstance(box, UUIDExtensionBox):
                if metadata_result.uuid is None:
                    msg = f"Found uuid box manually, but constructor didn't find it: {box}"
                    raise RuntimeError(msg)
            else:
                assert_never(box)

        return metadata_result

    def __read_metadata_toplevel_meta(  # noqa: PLR0915
        self: Self,
        box: MetaBox,
        f: BinaryIO,
    ) -> ReadMetadataImpl:
        metadata_result: ReadMetadataImpl = ReadMetadataImpl({}, None)

        meta_child_boxes = list(
            mp4_iter_boxes(f, box.span.payload_span),
        )

        if len(meta_child_boxes) != 1:
            msg = f"Invalid meta box: expected only one child, but got {len(meta_child_boxes)}"
            raise RuntimeError(msg)

        meta_child_box = meta_child_boxes[0]

        if meta_child_box.type != ILST_ATOM_NAME or not isinstance(
            meta_child_box,
            AppleItunesItemList,
        ):
            msg = f"Invalid meta child, expected AppleItunesItemList but got: {meta_child_box}"
            raise RuntimeError(msg)

        for data_box in mp4_iter_boxes(f, meta_child_box.span.payload_span):
            if not isinstance(
                data_box,
                (AppleItunesItemFreeformBox, AppleItunesItemBox),
            ):
                msg = f"Invalid data box in AppleItunesItemList: {data_box}"
                raise TypeError(msg)

            if isinstance(data_box, AppleItunesItemFreeformBox):

                mean = data_box.mean.value
                name = data_box.name.value

                if mean != TAGGER_DOMAIN:
                    msg = f"Invalid AppleItunesItemFreeformBox mean in meta box: {mean} != {TAGGER_DOMAIN}"
                    raise RuntimeError(msg)

                if name in [
                    TaggerDomain.UUID_RAW_KEY_FREEFORM.name,
                    TaggerDomain.UUID_HEX_KEY_FREEFORM.name,
                ]:
                    uuid: UUID

                    if name == TaggerDomain.UUID_RAW_KEY_FREEFORM.name:
                        if not isinstance(data_box.data.value, UUID):
                            msg = f"Invalid uuid (raw) key type: {type(data_box.data.value)} {data_box.data.value}"
                            raise RuntimeError(msg)

                        uuid = data_box.data.value
                    else:
                        if not isinstance(data_box.data.value, str):
                            msg = f"Invalid uuid (str) key type: {type(data_box.data.value)} {data_box.data.value}"
                            raise RuntimeError(msg)

                        uuid = uuid_from_str(data_box.data.value)

                    if metadata_result.uuid is not None:
                        if metadata_result.uuid != uuid:
                            msg = f"Duplicate uuid tag read, that are not the same: {uuid}"
                            raise RuntimeError(msg)
                    else:
                        metadata_result.uuid = uuid

                else:
                    raw_name = TaggerDomain.get_raw_name(name)

                    if not isinstance(data_box.data.value, str):
                        msg = f"Invalid value for metadata: expected str type, got {type(data_box.data.value)}"
                        raise RuntimeError(msg)

                    raw_value = json.loads(data_box.data.value)

                    mdt1: SerializableDict = cast(
                        SerializableDict,
                        metadata_result.metadata.get("metadata", {}),
                    )
                    mdt2: SerializableDict = {raw_name: raw_value}

                    metadata_result.metadata = merge_dicts(
                        metadata_result.metadata,
                        {
                            "metadata": merge_dicts(
                                mdt1,
                                mdt2,
                                "error",
                            ),
                        },
                        "overwrite",
                    )

            elif isinstance(data_box, AppleItunesItemBox):
                key: str
                if data_box.type == ISOMAtomName(b"\xa9cmt"):
                    key = "comment"
                else:
                    key = data_box.type.value.decode("latin-1")

                if isinstance(data_box.data.value, UUID):
                    msg = f"UUID currently not allowed in freeform boxes outsaide of the expected ones: {data_box.data.value}"
                    raise TypeError(msg)

                metadata_result.metadata = merge_dicts(
                    metadata_result.metadata,
                    {key: data_box.data.value},
                    "error",
                )

            else:
                assert_never(data_box)

        return metadata_result

    @staticmethod
    def __merge_metadata(
        mdt1: ReadMetadataImpl,
        mdt2: ReadMetadataImpl,
    ) -> ReadMetadataImpl:

        metadata_result: ReadMetadataImpl = ReadMetadataImpl({}, None)

        if mdt1.uuid is not None:
            if mdt2.uuid is not None and mdt1.uuid != mdt2.uuid:
                msg = f"UUID doesn't match: {mdt1.uuid} != {mdt2.uuid}"
                raise RuntimeError(msg)

            metadata_result.uuid = mdt1.uuid
        elif mdt2.uuid is not None:
            metadata_result.uuid = mdt2.uuid

        metadata_result.metadata = merge_dicts(
            metadata_result.metadata,
            mdt1.metadata,
            "error",
        )

        def is_value_eq(
            val1: SerializableDictValue,
            val2: SerializableDictValue,
        ) -> bool:
            return json.dumps(val1) == json.dumps(val2)

        for key, value in mdt2.metadata.items():
            if metadata_result.metadata.get(key, None) is not None:
                if not is_value_eq(metadata_result.metadata[key], value):
                    msg = f"Duplicate key '{key}' value doesn't match: {metadata_result.metadata[key]} != {value}"
                    raise RuntimeError(msg)
            else:
                metadata_result.metadata[key] = value

        return metadata_result

    def read_metadata(
        self: Self,
        f: BinaryIO,
    ) -> ReadMetadataImpl:
        custom_boxes: list[JsonExtensionBox | UUIDExtensionBox] = []
        meta_box: Optional[MetaBox] = None

        for box in self.__our_boxes:
            if isinstance(box, (JsonExtensionBox, UUIDExtensionBox)):
                custom_boxes.append(box)
            elif isinstance(box, MetaBox):
                if meta_box is not None:
                    msg = f"Duplicate 'meta' box at the top level, only one allowed: {box}"
                    raise RuntimeError(msg)

                if box.optional_boxes.pitm is None:
                    msg = "Meta box not written by us: missing pitm box"
                    raise RuntimeError(msg)

                pitm = box.optional_boxes.pitm

                if pitm.item_id != META_BOX_VIDEO_LANGUAGE_DETECTION_ID:
                    msg = "Meta box not written by us: invalid item id"
                    raise RuntimeError(msg)

                meta_box = box
            elif isinstance(box, FreeSpaceBox):
                # just ignore free space boxs, if they are only zero
                if not all(x == 0 for x in box.data):
                    msg = f"Not all zeros in padding box: {box.data!r}"
                    raise RuntimeError(msg)
            else:
                msg = f"Invalid box for tags found: {type(box)}"
                raise TypeError(msg)

        if meta_box is None:
            return self.__read_metadata_custom(custom_boxes)

        result_custom = self.__read_metadata_custom(custom_boxes)
        result_toplevel_meta = self.__read_metadata_toplevel_meta(meta_box, f)

        return MP4MetadataHandler.__merge_metadata(result_custom, result_toplevel_meta)

    @staticmethod
    def __read_meta_box_info(
        meta_box: MetaBox,
        f: BinaryIO,
    ) -> ReadMetaBoxValues:
        meta_child_boxes = list(
            mp4_iter_boxes(f, meta_box.span.payload_span),
        )

        if len(meta_child_boxes) != 1:
            msg = f"Invalid meta box: expected only one child, but got {len(meta_child_boxes)}"
            raise RuntimeError(msg)

        meta_child_box = meta_child_boxes[0]

        if meta_child_box.type != ILST_ATOM_NAME or not isinstance(
            meta_child_box,
            AppleItunesItemList,
        ):
            msg = f"Invalid meta child, expected AppleItunesItemList but got: {meta_child_box}"
            raise RuntimeError(msg)

        result: ReadMetaBoxValues = []

        for data_box in mp4_iter_boxes(f, meta_child_box.span.payload_span):
            if not isinstance(
                data_box,
                (AppleItunesItemFreeformBox, AppleItunesItemBox),
            ):
                msg = f"Invalid data box in AppleItunesItemList: {data_box}"
                raise TypeError(msg)

            if isinstance(data_box, AppleItunesItemFreeformBox):
                result.append(
                    ApplItunesTags.validate_init(
                        AppleItunesFreeformKey(
                            mean=data_box.mean.value,
                            name=data_box.name.value,
                        ),
                        ApplItunesTagsData.from_data_box(data_box.data),
                    ),
                )
            elif isinstance(data_box, AppleItunesItemBox):
                result.append(
                    ApplItunesTags.validate_init(
                        data_box.type,
                        ApplItunesTagsData.from_data_box(data_box.data),
                    ),
                )
            else:
                assert_never(data_box)

        return result

    @staticmethod
    def get_metadata_handler(  # noqa: PLR0915
        f: BinaryIO,
    ) -> "MP4MetadataHandler":

        uuid_box: Optional[UUIDExtensionBox] = None
        meta_values: MetaValues = Ok(None)

        def free_box_is_written_by_us(box: FreeSpaceBox) -> bool:
            # check if it's a padding box
            if all(x == 0 for x in box.data):
                return True

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

        def meta_box_is_written_by_us(box: MetaBox) -> bool:
            nonlocal meta_values
            if not (meta_values.ok() and meta_values.as_ok() is None):
                msg = f"Duplicate 'meta' box at the top level, only one allowed: {box}"
                raise RuntimeError(msg)

            if box.optional_boxes.pitm is None:
                meta_values = Err("Not written by us: missing pitm box")
                return False

            pitm = box.optional_boxes.pitm

            if pitm.item_id != META_BOX_VIDEO_LANGUAGE_DETECTION_ID:
                meta_values = Err("Not written by us: invalid item id")
                return False

            meta_values = Ok(MP4MetadataHandler.__read_meta_box_info(box, f))
            return True

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

            if box.type == META_ATOM_NAME:
                if not isinstance(box, MetaBox):
                    msg = "Invalid MetaBox: type not dispatched to correct class"
                    raise TypeError(msg)

                return meta_box_is_written_by_us(box)

            return False

        f.seek(0, 2)
        filesize = f.tell()

        f.seek(0)

        top_boxes: list[MP4Box] = list(mp4_iter_boxes(f, SimpleSpan(0, filesize)))

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

        return MP4MetadataHandler(
            uuid_box,
            meta_values,
            list(reversed(our_boxes_reversed)),
        )


@decorate_class(slots=True)
class VideoTaggerContextMP4(VideoTaggerContextRW):
    __writer: BinaryIO
    __streams: int
    __types: list[ISOMAtomName]

    def __init__(
        self: Self,
        manager: ManagerInterface,
        file: Path,
        writer: BinaryIO,
        streams: int,
        types: list[ISOMAtomName],
    ) -> None:
        super().__init__(manager, file=file)
        self.__writer = writer
        self.__streams = streams
        self.__types = types

    @override
    def write_tags(
        self: Self,
        tags: MetadataTags,
    ) -> None:

        # TODO. replace VIDEO_FILE_TAG_UPDATE_BAR_FORMAT everywhere, as we don't use bytes here!
        bar: CounterInterface = self.manager.counter(
            total=float(2),
            desc="update mp4 metadata tags",
            unit="B",
            leave=False,
            bar_format=VIDEO_FILE_TAG_UPDATE_BAR_FORMAT,
            color="red",
        )
        bar.update(0, force=True)

        try:
            mp4_metadata_handler = MP4MetadataHandler.get_metadata_handler(
                f=self.__writer,
            )

            bar.update(1, force=True)

            mp4_metadata_handler.write_new_metadata(
                self.__writer,
                tags,
            )

            bar.update(1, force=True)

            self.__writer.flush()
        finally:
            bar.close(clear=True)

    @override
    def restore_file(
        self: Self,
    ) -> RestoreFileNotSupported | Optional[str]:
        return RestoreFileNotSupported()

    @override
    def write_language(
        self: Self,
        language: Language,
    ) -> bool:
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

            self.__writer.flush()
        finally:
            bar.close(clear=True)

        return True

    @override
    def read_language(
        self: Self,
    ) -> Result[Optional[Language], str]:
        return Err("Not Implemented yet")

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

        mp4_metadata_handler = MP4MetadataHandler.get_metadata_handler(
            f=self.__writer,
        )

        metadata_result = mp4_metadata_handler.read_metadata(self.__writer)

        result: MetadataTagsRead = MetadataTagsRead(None, None, {}, [])

        if metadata_result.uuid is not None:
            result.uuid = metadata_result.uuid

        for key, value in metadata_result.metadata.items():

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


@decorate_class(slots=True)
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

                # read the file, so that we check if we can parse it correctly and that it is an mp4 file
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
        except (RuntimeError, ValueError, TypeError) as err:
            return Err(str(err))

    def __context_impl(
        self: Self,
        manager: ManagerInterface,
        ctx: ContextType,
    ) -> AbstractContextManager[VideoTaggerContextRW]:

        file = self.file
        streams = self.__streams
        types = self.__types

        @decorate_class(slots=True)
        class VideoTaggerContextCtx(VideoTaggerContextCtxGeneric):

            def __init__(self: Self) -> None:
                super().__init__(file, ctx, manager)

            @override
            def get_context(
                self: Self,
                manager: ManagerInterface,
                writer: BinaryIO,
            ) -> VideoTaggerContextMP4:
                return VideoTaggerContextMP4(manager, file, writer, streams, types)

        return VideoTaggerContextCtx()

    @override
    def r_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextReadable]:
        return self.__context_impl(manager, "r")

    @override
    def w_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextWriteable]:
        return self.__context_impl(manager, "w")

    @override
    def rw_ctx(
        self: Self,
        manager: ManagerInterface,
    ) -> AbstractContextManager[VideoTaggerContextRW]:
        return self.__context_impl(manager, "rw")

    @override
    def inspect(
        self: Self,
        printer: InspectPrinter,
    ) -> Optional[InspectNotImplemented]:

        def print_box(box: MP4Box, *, depth: int) -> None:

            priority = (
                InspectPriority.Important
                if box.is_container
                else InspectPriority.Normal
            )

            name: str = f"{box.type}"

            element = InspectElement(name, priority)

            printer.element(element, depth)

        with self.file.open(mode="rb") as f:

            def iterate_boxes_recursive(span: SimpleSpan, *, depth: int) -> None:
                for box in mp4_iter_boxes(f, span):

                    print_box(box, depth=depth)

                    if box.is_container:
                        iterate_boxes_recursive(
                            box.span.payload_span,
                            depth=depth + 1,
                        )

            f.seek(0, 2)
            filesize = f.tell()

            printer.start()
            iterate_boxes_recursive(SimpleSpan(0, filesize), depth=0)
            printer.end()

        return None
