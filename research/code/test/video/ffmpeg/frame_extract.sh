#!/bin/bash

VIDEO_FILE="$1"
NUM_FRAMES=50 # The desired fixed number of frames
OUTPUT_PATTERN="frame%04d.png"

if [ -z "$VIDEO_FILE" ]; then
    echo "Usage: $0 <video_file>"
    exit 1


# Use ffprobe to get the total number of frames in the video
TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$VIDEO_FILE" | tr -d '\n')

if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" -eq 0 ]; then
    echo "Could not determine total frames, or video has no frames."
    exit 1


# Calculate the interval (step size) between frames
# Using 'scale=0' ensures integer division after calculation
INTERVAL=$(echo "scale=0; $TOTAL_FRAMES / $NUM_FRAMES" | bc)

if [ "$INTERVAL" -eq 0 ]; then
    INTERVAL=1


echo "Total frames: $TOTAL_FRAMES"
echo "Interval: $INTERVAL"
echo "Extracting $NUM_FRAMES frames with interval $INTERVAL..."

# Use ffmpeg with the select filter to extract frames at the calculated interval
# The select filter 'not(mod(n, INTERVAL))' selects every Nth frame
# -vsync 0 prevents frame duplication
# -vframes limits the total output frames
ffmpeg -i "$VIDEO_FILE" -vf "select='not(mod(n, $INTERVAL))',setpts=N/TB" -vsync 0 -vframes "$NUM_FRAMES" "$OUTPUT_PATTERN"
