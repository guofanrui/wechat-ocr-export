# Build the two tiny macOS helpers and the Vision OCR wrapper.
# Requires Xcode command line tools (clang, swiftc). No third-party deps.

CC      ?= clang
CFLAGS  ?= -O2 -Wall
FW      := -framework CoreGraphics -framework CoreFoundation -framework ApplicationServices

all: tools/wxwin tools/wxscroll tools/ocrfast

tools/wxwin: tools/wxwin.c
	$(CC) $(CFLAGS) -o $@ $< $(FW)

tools/wxscroll: tools/wxscroll.c
	$(CC) $(CFLAGS) -o $@ $< $(FW)

tools/ocrfast: tools/ocr.swift
	@if ! swiftc -O -o $@ $<; then \
		echo ""; \
		echo "ocrfast 编译失败：多半是本机 Swift 与 SDK 版本不匹配，见 docs/troubleshooting.md"; \
		echo "wxwin / wxscroll 已可用；也可用 WXEXPORT_OCR=/path/to/ocrfast 指定现成的 OCR 程序。"; \
		rm -f $@; exit 1; \
	fi

clean:
	rm -f tools/wxwin tools/wxscroll tools/ocrfast

.PHONY: all clean
