from deepface import DeepFace
import cv2
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load your image
image_path = "/home/madhekar/temp/Deepface_issues/IMG_8629.PNG" # Replace with your image file
img = cv2.imread(image_path)
# plt.imshow(img)
# plt.show()

# Step 1: Detect all faces
# The result is a list of dictionaries, one for each face detected
detected_faces = DeepFace.extract_faces(
    img_path=image_path,
    detector_backend="retinaface",  # "opencv", # Use a detector backend
    enforce_detection=False,  # Set to False to prevent errors if no faces are found
)

# Step 2: Calculate area and sort faces
if detected_faces:
    # Calculate the pixel area for each face and add it to the dictionary
    for face in detected_faces:
        facial_area = face["facial_area"]
        area = facial_area["w"] * facial_area["h"]
        face["area"] = area

    # Step 3: Sort faces by area in descending order
    sorted_faces = sorted(detected_faces, key=lambda x: x["area"], reverse=True)
    print(sorted_faces)

    # Step 4: Limit to the top N faces (e.g., the 3 largest)
    num_faces_to_keep = 3
    restricted_faces = sorted_faces[:num_faces_to_keep]

    print(f"Detected {len(detected_faces)} faces initially. Kept {len(restricted_faces)} largest faces.")

    # Step 5: Visualize the restricted faces
    for face_obj in restricted_faces:
        x, y, w, h,_,_= face_obj["facial_area"].values()
        print(face_obj["facial_area"].values())
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Displaying top {num_faces_to_keep} largest faces")
    plt.axis('off')    
    plt.show()

    # cv2.imshow('in', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("No faces detected in the image.")