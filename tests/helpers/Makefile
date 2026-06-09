



PKG_CONFIG = pkg-config

# Base flags, environment CFLAGS / LDFLAGS can be appended.
BASE_CFLAGS = -std=c11 -Wall -Wextra -O2 -Wpedantic $(shell $(PKG_CONFIG) --cflags libavutil)
BASE_LDFLAGS = $(shell $(PKG_CONFIG) --libs libavutil)


all: av_helper

av_helper: av_helper.c
	$(CC) av_helper.c -o av_helper.o $(BASE_CFLAGS) $(CFLAGS) $(BASE_LDFLAGS) $(LDFLAGS)
