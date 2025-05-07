import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("👤 Face Recognition with Augmentation")
st.markdown("Upload known images and a test image to identify faces.")

# --- Helper Functions ---
def augment_image(image_np):
    augmented = [image_np]

    flipped = cv2.flip(image_np, 1)
    augmented.append(flipped)

    rows, cols = image_np.shape[:2]
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    M2 = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    augmented.append(cv2.warpAffine(image_np, M1, (cols, rows)))
    augmented.append(cv2.warpAffine(image_np, M2, (cols, rows)))

    return augmented

def get_face_encodings(image, name):
    encodings = []
    names = []
    for aug_img in augment_image(image):
        faces = face_recognition.face_encodings(aug_img)
        if faces:
            encodings.append(faces[0])
            names.append(name)
    return encodings, names

# --- Inputs ---
st.sidebar.header("Upload Known Faces")
known_files = st.sidebar.file_uploader("Known Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.sidebar.header("Upload Test Image")
test_file = st.sidebar.file_uploader("Test Image", type=["jpg", "jpeg", "png"])

if st.sidebar.button("🔍 Run Recognition"):
    if not known_files or not test_file:
        st.warning("Please upload at least one known image and one test image.")
    else:
        known_encodings = []
        known_names = []

        with st.spinner("Encoding known faces..."):
            for file in known_files:
                name = os.path.splitext(file.name)[0]
                image = face_recognition.load_image_file(file)
                encs, names = get_face_encodings(image, name)
                known_encodings.extend(encs)
                known_names.extend(names)

        test_image_np = face_recognition.load_image_file(test_file)
        face_locations = face_recognition.face_locations(test_image_np, model="cnn")
        face_encodings = face_recognition.face_encodings(test_image_np, face_locations)

        st.success(f"✅ Detected {len(face_encodings)} face(s) in test image.")

        image_bgr = cv2.cvtColor(test_image_np, cv2.COLOR_RGB2BGR)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]

            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="🖼️ Output Image", use_column_width=True)
