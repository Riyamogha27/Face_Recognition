import face_recognition
import cv2
import os
import numpy as np

# Paths
known_dir = r"/Known_faces"
test_dir = r"/test_faces"
output_dir = r"/CyFuture/results"
os.makedirs(output_dir, exist_ok=True)

# Function to augment known images
def augment_image(image):
    augmented = [image]  # Original

    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)

    # Rotate +15 degrees
    rows, cols = image.shape[:2]
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rotated1 = cv2.warpAffine(image, M1, (cols, rows))
    augmented.append(rotated1)

    # Rotate -15 degrees
    M2 = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    rotated2 = cv2.warpAffine(image, M2, (cols, rows))
    augmented.append(rotated2)

    return augmented

# Step 1: Load and augment known faces
known_encodings = []
known_names = []

for filename in os.listdir(known_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(known_dir, filename)
        image = face_recognition.load_image_file(image_path)

        print(f"Processing known image: {filename}")
        augmented_images = augment_image(image)

        for aug_img in augmented_images:
            encodings = face_recognition.face_encodings(aug_img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"❌ No face found in an augmented version of: {filename}")

print(f"✅ Total known encodings (with augmentation): {len(known_encodings)}")

# Step 2: Process test images
for filename in os.listdir(test_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(test_dir, filename)
        print(f"\n🧪 Now processing test image: {filename}")
        print(f"➡️  Image path: {image_path}")

        test_image = face_recognition.load_image_file(image_path)
        print(f"➡️  Image shape: {test_image.shape}")
        print(f"➡️  Detecting faces using CNN model...")

        face_locations = face_recognition.face_locations(test_image, model="cnn")
        face_encodings = face_recognition.face_encodings(test_image, face_locations)
        print(f"➡️  Faces found: {len(face_encodings)}")

        image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = known_names[matched_idx]

            # Draw rectangle and label
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # Save result
        output_path = os.path.join(output_dir, f"output_{filename}")
        cv2.imwrite(output_path, image_bgr)
        print(f"✅ Saved output image to: {output_path}")
