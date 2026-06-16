#!/bin/bash
# BFS dependency collector for ffmpeg + OpenCV.
# Walks ldd output for every .so in /dist and copies any missing
# transitive dependencies into /dist. Idempotent, runs until no
# new files are added (max 10 rounds to prevent infinite loops).
#
# Used by Dockerfile.distroless stage 3 (deps).
set -e

DIST="${DIST:-/dist}"
ROUND=0
CHANGED=1

echo "Starting BFS dependency collection into $DIST"
mkdir -p "$DIST/usr/bin" "$DIST/usr/lib/x86_64-linux-gnu" "$DIST/lib/x86_64-linux-gnu" "$DIST/etc"

# Copy the entry-point binaries (ffmpeg, ffprobe).
cp /usr/bin/ffmpeg "$DIST/usr/bin/"
cp /usr/bin/ffprobe "$DIST/usr/bin/"

# Iteratively walk ldd output and copy missing libs.
while [ "$CHANGED" = "1" ] && [ $ROUND -lt 10 ]; do
    CHANGED=0
    ROUND=$((ROUND + 1))
    echo "=== Round $ROUND ==="
    for entry in "$DIST/usr/bin/"* "$DIST/usr/lib/x86_64-linux-gnu/"*.so* "$DIST/lib/x86_64-linux-gnu/"*.so*; do
        [ -f "$entry" ] || continue
        # Skip if the original source path doesn't exist (e.g. it's already
        # a path under /dist).
        original="${entry#$DIST}"
        [ -f "$original" ] || continue
        while read -r lib; do
            if [ -f "$lib" ] && [ ! -f "$DIST$lib" ]; then
                dest_dir="$DIST$(dirname "$lib")"
                mkdir -p "$dest_dir" 2>/dev/null || true
                cp -L "$lib" "$DIST$lib" 2>/dev/null || true
                echo "  + $lib"
                CHANGED=1
            fi
        done < <(ldd "$original" 2>/dev/null | awk '/=>/ {print $3}')
    done
done

# Copy SSL certs and NSS config.
cp -r /etc/ssl "$DIST/etc/" 2>/dev/null || true
cp /etc/nsswitch.conf "$DIST/etc/" 2>/dev/null || true

echo "BFS collection finished in $ROUND rounds"
