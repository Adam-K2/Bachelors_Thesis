# Experiments to get results of each face recognition algorithms to the table, which is located in thesis in 6.7 section
# Goal is to evaluate resiliance against deepfake

import argparse
import numpy as np
import subprocess
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
from PIL import Image
from deepface import DeepFace
from face_ai_kit.FaceRecognition import FaceRecognition
import cv2
import sys
import os
import csv

def deepface_model(reference, image):
    metrics = ["cosine", "euclidean_l2"]  # euclidean_l2 should be best
    models = ["Facenet", "DeepFace", "ArcFace"] 

    try:
        result = DeepFace.verify(img1_path=reference, 
                                    img2_path=image, 
                                    model_name=models[2],
                                    distance_metric=metrics[0])
    except Exception as e:
        print(f"Deepface: Unable to detect face on image: {e}")
        count = 0
        return count
    if result["verified"]:
        count = 1
    else:
        count = 0
    return count

def face_ai_kit_model(reference, image, model_name):
    count = 0
    face_lib = FaceRecognition(recognition=model_name)
    
    frame_1 = cv2.imread(reference)
    frame_2 = cv2.imread(image)
    
    results1 = face_lib.face_detection(frame_1, align='keypoints')
    if not results1:
        print(f"{model_name}: Unable to detect face on reference image.")
        return None
    face_img1 = results1[0]["img"]

    results2 = face_lib.face_detection(frame_2, align='keypoints')
    if not results2:
        print(f"{model_name}: Unable to detect face on target image.")
        return None
    face_img2 = results2[0]["img"]

    distance = face_lib.verify(face_img1, face_img2)
    if distance < 1.1:
        count = 1
    else:
        count = 0
    return count
        
if __name__ == '__main__':
    count_face_1_2 = 0
    count_deep_1_2 = 0
    count_deep_1_fake = 0
    count_deep_2_fake = 0
    for i in range(1, 201):
        reference = f'data/{i}/CFP_1.jpg'
        image = f'data/{i}/CFP_2.jpg'
        deepfake = f'data/{i}/Deep_2.jpg'
        
        # DeepFace model
        # deep_distances_genuine = deepface_model(reference, image)
        # deep_distances_fake1 = deepface_model(reference, deepfake)
        # deep_distances_fake2 = deepface_model(image, deepfake)
        # count_deep_1_2 += deep_distances_genuine
        # count_deep_1_fake += deep_distances_fake1
        # count_deep_2_fake += deep_distances_fake2
        
        # Face-ai-kit models
        model_name = "magface"
        deep_distances_genuine = face_ai_kit_model(reference, image, model_name)
        deep_distances_fake1 = face_ai_kit_model(reference, deepfake, model_name)
        deep_distances_fake2 = face_ai_kit_model(image, deepfake, model_name)
        count_deep_1_2 += deep_distances_genuine
        count_deep_1_fake += deep_distances_fake1
        count_deep_2_fake += deep_distances_fake2
        
        print(i)
    print(count_deep_1_2/200,(200-count_deep_1_fake)/200,(200-count_deep_2_fake)/200)
    # FaceNet L2 - 0.745 1.0 1.0
    # FaceNet Cos - 0.885 0.995 0.985
    # DeepFace L2 - 0.64 0.585 0.245
    # DeepFace Cos - 0.725 0.46 0.185
    # ArcFace L2 - 0.955 0.99 0.94
    # ArcFace Cos - 0.965 0.94 0.88
    # MagFace L2 - 1.0 1.0 1.0
    # Siamese - 0.89 0.565 0.915