#!/bin/bash
INPUT="$1"
TARGET_SIZE_MB="$2"
OUTPUT="${INPUT%.*}_compressed.mp4"

# 1. Get video duration
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT")

# 2. Calculate bitrates (Allocating 128k for audio)
AUDIO_BITRATE=128
VIDEO_BITRATE=$(awk "BEGIN {print int(($TARGET_SIZE_MB * 1024 * 1024 * 8 / $DURATION / 1000) - $AUDIO_BITRATE)}")

echo "Target Video Bitrate: ${VIDEO_BITRATE}k"

# 3. Two-pass encoding
ffmpeg -y -i "$INPUT" -c:v libx264 -b:v "${VIDEO_BITRATE}k" -pass 1 -an -f null /dev/null && \
ffmpeg -y -i "$INPUT" -c:v libx264 -b:v "${VIDEO_BITRATE}k" -pass 2 -c:a aac -b:a "${AUDIO_BITRATE}k" "$OUTPUT"

# Clean up log files
rm ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree 2>/dev/null
