import cv2
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Initialize InsightFace for face detection and alignment
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Load an image
img_path = "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/e8828925-b35c-4779-a62e-1adcb11a156c.jpg"  # Replace with your image path
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image at {img_path}")
else:
    # Detect faces using InsightFace
    faces = app.get(img)

    for face in faces:
        # Extract the bounding box from the InsightFace detection
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Crop the face from the original image
        face_img = img[y1:y2, x1:x2]

        # Perform emotion analysis on the cropped face using DeepFace
        try:
            analysis = DeepFace.analyze(
                face_img, actions=["emotion"], enforce_detection=False
            )

            # Assuming a single face in the cropped region for simplicity
            if analysis:
                emotion = analysis[0]["dominant_emotion"]
                confidence = analysis[0]["emotion"][emotion]

                print(f"Detected emotion: {emotion} with confidence: {confidence:.2f}%")

                # Draw bounding box and emotion label on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            else:
                print("No emotion detected for this face.")

        except Exception as e:
            print(f"Error during DeepFace analysis: {e}")

    # Display the result
    cv2.imshow("Emotion Analysis", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
