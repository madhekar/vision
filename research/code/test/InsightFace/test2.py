import cv2
from deepface import DeepFace
from insightface.app import FaceAnalysis

"""

DeepFace is a Python library primarily known for its face recognition and facial attribute analysis capabilities, including emotion analysis. 
While DeepFace uses its own internal face detection and landmark detection mechanisms (which can be configured to use various backends like OpenCV, Dlib, RetinaFace, etc.), 
it can be integrated with other libraries like InsightFace for enhanced landmark detection, which can then be used to potentially improve emotion analysis.
Integrating InsightFace Landmarks with DeepFace for Emotion Analysis:
Landmark Detection with InsightFace.

InsightFace is a powerful library for 2D and 3D face analysis, including robust facial landmark detection. You would use InsightFace to detect the facial landmarks on a given image.
Python

    import insightface
    import cv2

    # Load InsightFace model for landmark detection
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 for GPU, -1 for CPU

    # Load image
    img = cv2.imread("path/to/your/image.jpg")

    # Detect faces and landmarks
    faces = model.get(img)

    # 'faces' will contain a list of detected faces, each with a 'landmark_2d_106' attribute (or similar)
    # This attribute holds the coordinates of the detected facial landmarks.

Emotion Analysis with DeepFace.
DeepFace's analyze function performs facial attribute analysis, including emotion detection. 
While DeepFace typically handles landmark detection internally, you can potentially feed the results from 
InsightFace to guide or enhance the emotion analysis.

Python

    from deepface import DeepFace

    # Assuming you have a detected face from InsightFace (e.g., faces[0])
    # You would need to extract the relevant face region or use the landmark information
    # to potentially refine the input for DeepFace's emotion analysis.

    # Option 1: Directly analyze the whole image with DeepFace (DeepFace will do its own detection)
    result = DeepFace.analyze(img_path="path/to/your/image.jpg", actions=['emotion'])

    # Option 2: Pass a cropped face image to DeepFace for analysis (if InsightFace provides a clear bounding box)
    # This might require extracting the face region based on InsightFace's detection.
    # For example, if 'faces[0].bbox' gives the bounding box:
    x1, y1, x2, y2 = faces[0].bbox.astype(int)
    cropped_face = img[y1:y2, x1:x2]
    # Then save or pass the cropped_face as input to DeepFace.analyze
    # result = DeepFace.analyze(img_path=cropped_face, actions=['emotion'])

Note: DeepFace's emotion analysis models are trained on specific facial features and patterns. Directly "feeding" 
InsightFace landmarks into DeepFace's emotion analysis module might not be a straightforward process, as DeepFace's internal models are designed to work with its own detection and feature 
extraction. However, using InsightFace for highly accurate face detection and then providing those detected face regions to DeepFace could lead to more precise emotion analysis by ensuring DeepFace 
focuses on the correct facial area. You could also use the InsightFace landmarks to perform custom feature engineering for a separate emotion classification model if you are building a custom solution.

"""

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
