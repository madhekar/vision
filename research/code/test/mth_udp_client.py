import threading
import queue
import cv2

# ffmpeg -re -i C:\Users\test\Downloads\out.ts -strict -2 -c:v copy -an -preset slower -tune stillimage -b 11200k -f rawvideo udp://127.0.0.1:5000

q = queue.Queue()

def receive():
    cap = cv2.VideoCapture("udp://@192.168.68.115:5000?buffer_size=65535&pkt_size=65535&fifo_size=65535")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)

def display():
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow("Video", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:  # press "ESC" to quit
            break

tr = threading.Thread(target=receive, daemon=True)
td = threading.Thread(target=display)

tr.start()
td.start()

td.join()
