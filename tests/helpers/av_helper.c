/*
 * Copyright (c) The FFmpeg developers
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */


#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "libavutil/base64.h"
#include "libavutil/bprint.h"
#include "libavutil/error.h"
#include "libavutil/hash.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/macros.h"
#include "libavutil/opt.h"

// Partially Copied from https://github.com/FFmpeg/FFmpeg/blob/n8.1.1/fftools/textformat/avtextformat.c
// modified for our usage

static void bprint_bytes(AVBPrint *bp, const uint8_t *ubuf, size_t ubuf_size)
{
    av_bprintf(bp, "0X");
    for (unsigned i = 0; i < ubuf_size; i++)
        av_bprintf(bp, "%02X", ubuf[i]);
}


//see https://github.com/FFmpeg/FFmpeg/blob/n8.1.1/fftools/textformat/avtextformat.c#L307
static inline int validate_string(char **dstp, const char *src, unsigned int string_validation_utf8_flags)
{
    const uint8_t *p, *endp, *srcp = (const uint8_t *)src;
    AVBPrint dstbuf;
    AVBPrint invalid_seq;
    int invalid_chars_nb = 0, ret = 0;

    *dstp = NULL;
    av_bprint_init(&dstbuf, 0, AV_BPRINT_SIZE_UNLIMITED);
    av_bprint_init(&invalid_seq, 0, AV_BPRINT_SIZE_UNLIMITED);

    endp = srcp + strlen(src);
    for (p = srcp; *p;) {
        int32_t code;
        int invalid = 0;
        const uint8_t *p0 = p;

        if (av_utf8_decode(&code, &p, endp, string_validation_utf8_flags) < 0) {

            av_bprint_clear(&invalid_seq);

            bprint_bytes(&invalid_seq, p0, p - p0);

            //av_log(tctx, AV_LOG_DEBUG, "Invalid UTF-8 sequence '%s' found in string '%s'\n", invalid_seq.str, src);
            invalid = 1;
        }

        if (invalid) {
            invalid_chars_nb++;

            av_bprintf(&dstbuf, "%s", string_validation_replacement);
        }

    }

end:
    av_bprint_finalize(&dstbuf, dstp);
    av_bprint_finalize(&invalid_seq, NULL);
    return ret;
}


// see: https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/mov.c#L346
