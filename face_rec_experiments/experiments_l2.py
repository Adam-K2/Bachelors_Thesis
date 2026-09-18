# Experiments to get results of each face recognition algorithms
# Goal is to evaluate resiliance against deepfake
# This skript is doing experiments with euclidean_l2 metric

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
    models = ["Facenet", "DeepFace"] 
    distances = {}

    for model in models:
        try:
            result = DeepFace.verify(img1_path=reference, 
                                     img2_path=image, 
                                     model_name=model,
                                     distance_metric=metrics[1])
            distances[model] = result['distance']
        except Exception as e:
            print(f"Deepface ({model}): Unable to detect face on image: {e}")
            distances[model] = None
    
    return distances

def face_ai_kit_model(reference, image, model_name):
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
    return distance
        
if __name__ == '__main__':
    results = []
    for i in range(1, 201):
        reference = f'data/{i}/CFP_1.jpg'
        image = f'data/{i}/CFP_2.jpg'
        deepfake = f'data/{i}/Deep_2.jpg'

        row = {'img1': reference, 'img2': image, 'deepfake': deepfake}
        
        # DeepFace model
        deep_distances_genuine = deepface_model(reference, image)
        deep_distances_fake = deepface_model(reference, deepfake)
        for model in ["Facenet", "DeepFace"]:
            row[f'{model}_genuine'] = deep_distances_genuine.get(model)
            row[f'{model}_fake'] = deep_distances_fake.get(model)
        
        # Face-ai-kit models
        for model_name in ["arcface", "magface"]:
            genuine_distance = face_ai_kit_model(reference, image, model_name)
            fake_distance = face_ai_kit_model(reference, deepfake, model_name)
            row[f'{model_name}_genuine'] = genuine_distance
            row[f'{model_name}_fake'] = fake_distance
        print(i)
        results.append(row)
    
    # Create csv file
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['img1', 'img2', 'deepfake', 'Facenet_genuine', 'Facenet_fake', 'DeepFace_genuine', 'DeepFace_fake', 'arcface_genuine', 'arcface_fake', 'magface_genuine', 'magface_fake']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Success.")