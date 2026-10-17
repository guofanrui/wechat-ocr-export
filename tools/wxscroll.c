/*
 * wxscroll - post scroll-wheel events via CoreGraphics (no third-party deps)
 *
 * Build:
 *   clang -O2 -Wall -o wxscroll wxscroll.c -framework CoreGraphics -framework CoreFoundation
 *
 * Usage:
 *   wxscroll <x> <y> <lines> [--unit pixel]
 *
 *   x, y   point to scroll at (screen coordinates, top-left origin); the cursor
 *          is moved there first so the event lands on the window under it
 *   lines  signed wheel amount: positive = scroll up (toward older messages),
 *          negative = scroll down (toward newer messages)
 *
 * Requires the posting process to have Accessibility permission.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ApplicationServices/ApplicationServices.h>

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--check") == 0) {
        /* permission probe only: moves nothing, posts nothing */
        printf("accessibility_trusted=%d\n", AXIsProcessTrusted() ? 1 : 0);
        return AXIsProcessTrusted() ? 0 : 1;
    }
    if (argc < 4) {
        fprintf(stderr, "usage: %s <x> <y> <lines> [--unit pixel]\n", argv[0]);
        return 2;
    }

    double x = atof(argv[1]);
    double y = atof(argv[2]);
    int lines = atoi(argv[3]);
    CGScrollEventUnit unit = kCGScrollEventUnitLine;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--unit") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "pixel") == 0) unit = kCGScrollEventUnitPixel;
            i++;
        }
    }

    CGPoint pt = CGPointMake(x, y);

    fprintf(stderr, "accessibility_trusted=%d\n", AXIsProcessTrusted() ? 1 : 0);

    CGEventRef move = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, pt, kCGMouseButtonLeft);
    if (!move) {
        fprintf(stderr, "failed to create mouse event\n");
        return 1;
    }
    CGEventPost(kCGHIDEventTap, move);
    CFRelease(move);

    usleep(120000); /* let the move settle so the target window is current */

    int step = lines > 0 ? 1 : -1;
    int remaining = lines > 0 ? lines : -lines;
    int posted = 0;
    int chunk_max = (unit == kCGScrollEventUnitPixel) ? 120 : 3;

    while (remaining > 0) {
        int chunk = remaining > chunk_max ? chunk_max : remaining;
        CGEventRef sc = CGEventCreateScrollWheelEvent(NULL, unit, 1, step * chunk);
        if (!sc) {
            fprintf(stderr, "failed to create scroll event\n");
            return 1;
        }
        CGEventPost(kCGHIDEventTap, sc);
        CFRelease(sc);
        posted += chunk;
        remaining -= chunk;
        usleep(unit == kCGScrollEventUnitPixel ? 12000 : 40000);
    }

    fprintf(stderr, "posted scroll: at (%.0f,%.0f) amount=%d unit=%s\n",
            x, y, posted * step,
            unit == kCGScrollEventUnitPixel ? "pixel" : "line");
    return 0;
}
