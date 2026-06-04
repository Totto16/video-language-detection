# inside strh
# see https://learn.microsoft.com/de-de/previous-versions/windows/desktop/api/avifmt/ns-avifmt-avistreamheader#syntax

import struct
from pathlib import Path


RIFF_HEADER = b"RIFF"
LIST = b"LIST"
STRH = b"strh"
STRL = b"strl"


def read_chunk_header(f, offset):
    f.seek(offset)
    header = f.read(8)
    if len(header) < 8:
        return None

    chunk_id, size = struct.unpack("<4sI", header)
    return chunk_id, size, offset + 8


def iter_riff_chunks(f, start, end):
    pos = start

    while pos + 8 <= end:
        hdr = read_chunk_header(f, pos)
        if not hdr:
            return

        chunk_id, size, data_start = hdr

        # RIFF chunks are word-aligned
        padded_size = size + (size & 1)
        next_pos = data_start + padded_size

        if next_pos > end:
            raise ValueError(f"Corrupt chunk {chunk_id} exceeds bounds")

        yield chunk_id, pos, data_start, size, next_pos
        pos = next_pos


def find_riff_root(f):
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)

    header = f.read(12)
    if len(header) < 12:
        raise ValueError("Not a valid RIFF file")

    riff, file_size, riff_type = struct.unpack("<4sI4s", header)

    if riff != RIFF_HEADER or riff_type != b"AVI ":
        raise ValueError("Not an AVI RIFF file")

    return 12, size


def decode_language(val):
    # 16-bit packed: 5-bit chars (a=1)
    chars = []
    for shift in (10, 5, 0):
        v = (val >> shift) & 0x1F
        if v == 0:
            chars.append(" ")
        else:
            chars.append(chr(v + 0x60))
    return "".join(chars).strip()


def encode_language(code):
    if len(code) != 3:
        raise ValueError("AVI language must be 3 letters")

    code = code.lower()
    val = 0

    for ch in code:
        n = ord(ch) - 0x60
        if not (1 <= n <= 26):
            raise ValueError(f"Invalid language char: {ch}")
        val = (val << 5) | n

    return val


def find_strh_offsets(f):
    start, end = find_riff_root(f)

    for chunk_id, offset, data_start, size, next_pos in iter_riff_chunks(f, start, end):

        # We only care about LIST 'strl'
        if chunk_id == LIST:
            f.seek(data_start)
            list_type = f.read(4)

            if list_type != STRL:
                continue

            list_end = offset + 8 + size

            # search inside strl
            for cid, off, data_start2, size2, next2 in iter_riff_chunks(
                f, data_start + 4, list_end
            ):
                if cid == STRH:
                    yield off, data_start2, size2


def read_strh_language(f, strh_data_start):
    # STRH layout is fixed for AVI:
    # 0..?
    # language is at offset 24 (common AVI VFW layout)
    LANG_OFFSET = 24

    f.seek(strh_data_start + LANG_OFFSET)
    raw = f.read(2)

    if len(raw) != 2:
        raise ValueError("Truncated strh")

    return struct.unpack("<H", raw)[0], strh_data_start + LANG_OFFSET


def patch_avi_language(path, new_lang="eng"):
    new_val = encode_language(new_lang)

    with open(path, "r+b") as f:
        for strh_off, data_start, size in find_strh_offsets(f):

            old_val, lang_pos = read_strh_language(f, data_start)

            old_lang = decode_language(old_val)

            # sanity: ensure we don't overwrite outside box
            assert lang_pos + 2 <= data_start + size, "Language field out of bounds"

            f.seek(lang_pos)
            f.write(struct.pack("<H", new_val))

            print(f"{old_lang} -> {new_lang}")

            # verify
            f.seek(lang_pos)
            check = struct.unpack("<H", f.read(2))[0]
            assert check == new_val, "Write verification failed"
