#import file part
import cv2
import numpy as np
from retinaface import RetinaFace
import tensorflow as  tf

# Set fixed random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#Face detection Part
def detect_faces(image, min_confidence=0.9):
    faces = RetinaFace.detect_faces(image)
    bounding_boxes = []
    for face in faces.values():
        if face['score'] >= min_confidence:
            x1, y1, x2, y2 = face['facial_area']
            bounding_boxes.append((x1, y1, x2, y2))
    return bounding_boxes

#found face is algined to strain ,get the landmark of eye distance
def align_face(image, landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned