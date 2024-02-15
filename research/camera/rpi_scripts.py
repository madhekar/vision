libcamera-vid -o - -t 0 -width 800 -height 600 -fps 12  | cvlc -vvv stream:///dev/stdin --sout '#rtp{sdp=rtsp://:8080/}' :demux=h264



#!/bin/bash
array[0]="none"
array[1]="negative"
array[2]="solarise"
array[3]="sketch"
array[4]="denoise"
array[5]="emboss"
array[6]="oilpant"
array[7]="hatch"
array[8]="gpen"
array[9]="pastel"
array[10]="watercolour"
array[11]="film"
array[12]="blur"
array[13]="saturation"
array[14]="colourswap"
array[15]="washedout"
array[16]="posterise"
array[17]="colourpoint"
array[18]="colourbalance"
array[19]="cartoon"
size=${#array[@]}
index=$(($RANDOM % $size))
echo ${array[$index]}
sleep 1
raspivid -t 0 -w 800 -h 600 -ifx ${array[$index]} -fps 15 -l -o tcp://0.0.0.0:5000


libcamera-vid -t 0 --width 1080 --height 720 -q 100 -n --codec mjpeg --inline --listen -o tcp://zesha.local:8888 -v

libcamera-vid -t 0 --width 1080 --height 720 -q 100 -n --codec mjpeg --inline --listen -o tcp://192.168.68.115:8888 -v



libcamera-vid -t 0 --width 1920 --height 1080 --inline --listen -o tcp://0.0.0.0:8000 websockify 0.0.0.0:8001 0.0.0.0:8000

-----

libcamera-vid -t 0 --inline -n -o - | cvlc stream:///dev/stdin --sout '#rtp{sdp=rtsp://:10554/stream_iot}' :demux=h264 >> rtsp_record.txt 2>&1 &

rtsp://madhekar:Manya1@us@192.168.68.115:10554/stream_iot

----

libcamera-vid -t 0 --inline -o udp://192.168.68.115:3009 --codec h264


----

libcamera-vid -o - -t 0 -fps 25 -g 50 -rot 180 -n -a 12 -b 6000000 | ffmpeg -re -ar 44100 -ac 2 -acodec pcm_s16le -f s16le -ac 2 -i /dev/zero -f h264 -thread_queue_size 256 -i - -vcodec copy -acodec aac -ab 128k -strict experimental -f flv rtmp://a.rtmp.youtube.com/live2/YOUR_KEY_HERE

---

libcamera-vid --framerate 30 --nopreview --inline -t 0 --width 1920 --height 1080 --listen -o - | ffmpeg -i - -profile:v high -pix_fmt yuvj420p -level:v 4.1 -preset ultrafast -tune zerolatency -vcodec libx264 -r 10 -s 1920x1080 -f mpegts -flush_packets 0 udp://192.168.0.3:5000?pkt_size=1316

