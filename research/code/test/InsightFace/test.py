import cv2
#from PIL import Image
import inspireface as isf

# Create a session with optional features
opt = isf.HF_ENABLE_NONE
session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)

# Load the image using OpenCV.
image = cv2.imread('/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/IMG_3141.JPG')
#image = Image.open('/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/IMG_3141.JPG')
# Perform face detection on the image.
faces = session.face_detection(image)

for face in faces:
    print(face.location)
    x1, y1, x2, y2 = face.location
    rect = ((x1, y1), (x2, y2), face.roll)
     # Calculate center, size, and angle
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    size = (x2 - x1, y2 - y1)
    angle = face.roll

    # Apply rotation to the bounding box corners
    rect = ((center[0], center[1]), (size[0], size[1]), angle)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    # Draw the rotated bounding box
    cv2.drawContours(image, [box], 0, (100, 180, 29), 2)


cv2.imshow("face detection", image)

#if cv2.waitKey(1) & 0xFF == ord('q'):
cv2.waitKey(10)
cv2.destroyAllWindows()