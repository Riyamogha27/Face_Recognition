# 🧠 Face Recognition with Data Augmentation

This project uses the `face_recognition` library with OpenCV to detect and recognize faces from test images. To improve accuracy, it applies **data augmentation** techniques to known face images such as **flipping** and **rotation**.

---

## 🚀 Features

- Recognizes multiple faces in test images using a **CNN** model.
- Augments known faces to improve recognition (flip, rotate).
- Saves annotated images with names and bounding boxes.

---

## 📁 Folder Structure

Face_Recognition/
├── face_recognition_augmented.py # Main script
├── Known_faces/ # Folder with labeled known faces
├── test_faces/ # Folder with test images
├── results/ # Output folder (auto-created)
├── README.md
└── requirements.txt # Python dependencies


---

## ⚙️ Installation

Install required packages:

```bash
pip install -r requirements.txt
