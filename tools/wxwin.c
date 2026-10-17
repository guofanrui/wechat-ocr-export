/*
 * wxwin - list on-screen windows (for locating the WeChat conversation window)
 *
 * No third-party dependencies; links only against system frameworks.
 *
 * Build:
 *   clang -O2 -Wall -o wxwin wxwin.c -framework CoreGraphics -framework CoreFoundation
 *
 * Usage:
 *   wxwin            # human readable
 *   wxwin --json     # machine readable
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>

static void cfstr_to_buf(CFStringRef s, char *buf, size_t n) {
    buf[0] = '\0';
    if (!s) return;
    CFStringGetCString(s, buf, (CFIndex)n, kCFStringEncodingUTF8);
}

static void json_escape(const char *in, char *out, size_t n) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 7 < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"' || c == '\\') {
            out[j++] = '\\';
            out[j++] = (char)c;
        } else if (c == '\n') {
            out[j++] = '\\'; out[j++] = 'n';
        } else if (c == '\t') {
            out[j++] = '\\'; out[j++] = 't';
        } else if (c < 0x20) {
            j += (size_t)snprintf(out + j, n - j, "\\u%04x", c);
        } else {
            out[j++] = (char)c;
        }
    }
    out[j] = '\0';
}

int main(int argc, char **argv) {
    int json = (argc > 1 && strcmp(argv[1], "--json") == 0);

    CFArrayRef list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID);
    if (!list) {
        fprintf(stderr, "CGWindowListCopyWindowInfo failed\n");
        return 1;
    }

    CFIndex n = CFArrayGetCount(list);
    if (json) printf("[\n");
    int first = 1;

    for (CFIndex i = 0; i < n; i++) {
        CFDictionaryRef w = CFArrayGetValueAtIndex(list, i);
        if (!w || CFGetTypeID(w) != CFDictionaryGetTypeID()) continue;

        CFStringRef ownerRef = CFDictionaryGetValue(w, kCGWindowOwnerName);
        CFStringRef nameRef = CFDictionaryGetValue(w, kCGWindowName);
        CFNumberRef idRef = CFDictionaryGetValue(w, kCGWindowNumber);
        CFNumberRef layerRef = CFDictionaryGetValue(w, kCGWindowLayer);
        CFDictionaryRef boundsRef = CFDictionaryGetValue(w, kCGWindowBounds);

        char owner[256] = "";
        char name[512] = "";
        cfstr_to_buf(ownerRef, owner, sizeof owner);
        cfstr_to_buf(nameRef, name, sizeof name);

        int wid = 0, layer = 0;
        if (idRef) CFNumberGetValue(idRef, kCFNumberIntType, &wid);
        if (layerRef) CFNumberGetValue(layerRef, kCFNumberIntType, &layer);

        double bx = 0, by = 0, bw = 0, bh = 0;
        if (boundsRef) {
            CGRect r;
            if (CGRectMakeWithDictionaryRepresentation(boundsRef, &r)) {
                bx = r.origin.x; by = r.origin.y;
                bw = r.size.width; bh = r.size.height;
            }
        }

        if (json) {
            char eo[600], en[1200];
            json_escape(owner, eo, sizeof eo);
            json_escape(name, en, sizeof en);
            printf("%s  {\"id\":%d,\"layer\":%d,\"owner\":\"%s\",\"name\":\"%s\","
                   "\"x\":%.1f,\"y\":%.1f,\"w\":%.1f,\"h\":%.1f}",
                   first ? "" : ",\n", wid, layer, eo, en, bx, by, bw, bh);
            first = 0;
        } else {
            if (bw > 200 && bh > 200) {
                printf("id=%-8d layer=%-3d %8.0fx%-8.0f at (%6.0f,%6.0f)  %s  |  %s\n",
                       wid, layer, bw, bh, bx, by, owner, name);
            }
        }
    }

    if (json) printf("\n]\n");
    CFRelease(list);
    return 0;
}
