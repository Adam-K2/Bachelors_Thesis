# Author: Adam Kucik

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

os.environ['CUDA_VISIBLE_DEVICES']="-1"
# Argument parser function
def argument_parse():
    parser = argparse.ArgumentParser(description='''
                                     Main goal of this application is to generate deepfake images and also test resilience of face 
                                     recognition algorithms against deepfake. The flow of application supports three different models.
                                     The first one is vgg, which needs one input image and on output will send result if the image is
                                     deepfake or not. The second one is siamese, which uses two input images. The first image should be
                                     referencial and the second one potential deepfake. With this model you can also choose to see
                                     result from 4 selected face recognition algorithms. We support DeepFace, ArcFace, Facenet and MagFace.
                                     Last model is similar to the siamese. It is generate, which takes as first image in argument reference
                                     image and the second one face of person that we want to place on reference. After the generation of
                                     deepfake the result is same as for siamese model. The model was trained and tested on jpg images.
                                     The jpg images must be 128x128 size.
                                     
                                     The output of vgg or siamese is: Predicted image is Deepfake or Original.
                                     The output of each face recognition algoritm is whether the images contain same person.
                                     We provide also the vektor distance and treshold. If the distance is larger than thershold it means
                                     that the images do not contain same person. In another words that the image is probably deepfake.
                                     ''')
    parser.add_argument('-m', '--model', choices=['siamese', 'vgg', 'generate'], required=True,
                        help='Choose a model (siamese, vgg or deepface)')
    parser.add_argument('images', metavar='image.jpg', type=str, nargs='+',
                        help='Image files')
    parser.add_argument('-r', '--recognition', choices=['ArcFace', 'DeepFace', 'MagFace','Facenet'], required=False,
                        help='Choose a face recognition algorithm (ArcFace, DeepFace, MagFace, Facenet)')
    parser.add_argument('-v', '--vektor', choices=['l2', 'cos'], required=False,
                        help='Choose a vektor metric (l2 or cos)')
    parser.add_argument('-c', '--csv', action='store_true', required=False,
                        help='Create csv log file')

    args = parser.parse_args()
    
    order = 0
    
    global rec_algo, vektor, create_csv     # Global variables if voluntarily arguments are set
    rec_algo = args.recognition
    vektor = args.vektor
    create_csv = args.csv

    if args.model == 'siamese':     # Without generating of deepfake
        if len(args.images) != 2:
            parser.error("Siamese model requires two image files.")
        order = 2
    elif args.model == 'vgg':       # Without generating of deepfake
        if len(args.images) != 1:
            parser.error("VGG model requires one image file.")
        order = 1
    elif args.model == 'generate':  # With deepfake generation, the rest of program is same as for args.model == 'siamese'
        if len(args.images) != 2:
            parser.error("Application pipeline requires two input images.")
        order = 2
        command = f"python run.py -s {args.images[1]} -t {args.images[0]} -o .\out --execution-providers cpu --headless"
        subprocess.run(command, shell=True)
        files = os.listdir('.\out')
        sorted_files = sorted(files, key=lambda f: os.path.getmtime(os.path.join('.\out', f)), reverse=True) # To make sure that the lates created image is the right one
        args.images[1] = f".\out\{sorted_files[0]}"
    else:
        print("Error occured in argument parsing")
        exit(1)
            
    return args.images, order # Order helps to choose model in model_select function

# Function to select which model will be used
def model_select(count):
    if count == 1:
        name = 'vgg16_50_imagenet'
    elif count == 2:
        name = 'siamese_model_none_cross_16'
    return name

# Function to log results into a csv file if argument -c or --csv is set 
def log_to_csv(reference, image, model_name, prediction, distance=None, threshold=None):
    if not create_csv:
        return
    csv_file = 'app_result.csv'
    file_exist = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exist:
            writer.writerow(['Reference_Image', 'Test_Image', 'Model', 'Prediction', 'Distance', 'Threshold'])
        writer.writerow([reference, image, model_name, prediction, distance, threshold])    

# Vgg model handler
def vgg_model(image):
    img = Image.open(image)
    
    img = np.array(img, dtype='float32') / 255.0

    predict = model.predict(np.expand_dims(img, 0))
    
    if predict >= 0.5:
        prediction = 'Deepfake' 
    else: 
        prediction = 'Original'
    
    print(f'Vgg: Predicted image is {prediction}')
    
    log_to_csv(image, '', 'Vgg', prediction) # If csv argument, then the result is also logged to the csv file 

# Siamese model handler and also calling of face recognition libraries - the main purpose of application according to thesis assignment
def siamese_model(reference, image):
    img1 = Image.open(reference)
    img2 = Image.open(image)
    
    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = model.predict(np.expand_dims(image_pair, axis=0))
    
    evaluation = predictions[0][0]
    
    if evaluation >= 0.5:
        prediction = 'Deepfake' 
    else: 
        prediction = 'Original'

    print(f'Siamese: Predicted image is {prediction}')
    
    if rec_algo == None:
        face_ai_kit_model(reference, image, prediction)
        deepface_model(reference, image, prediction)
    elif rec_algo != "MagFace":
        deepface_model(reference, image, prediction)
    else:
        face_ai_kit_model(reference, image, prediction)

# Calling of Deepface library
# FaceNet, DeepFace, ArcFace
def deepface_model(reference, image, prediction):
    metrics = ["cosine", "euclidean_l2"] # Choose metric
    models = ["Facenet", "DeepFace", "ArcFace"] # Choose model
    threshold_cos = [0.4, 0.23, 0.68] # Tresholds for cos metric
    threshold_l2 = [0.8, 0.64, 1.13] # Tresholds for l2 metric
    
    if vektor == "cos":
        metrcis_select = 0
    else:
        metrcis_select = 1
    
    if rec_algo == None:
        rec_algo_select = "Facenet"
    else:
        rec_algo_select = rec_algo

    index = models.index(rec_algo_select)
    if metrcis_select == 0:
        threshold = threshold_cos[index]
    else:
        threshold = threshold_l2[index]
    
    try:
        result = DeepFace.verify(img1_path = reference, 
            img2_path = image, 
            model_name = rec_algo_select,
            distance_metric = metrics[metrcis_select])

        if result["verified"]:
            print(f"{rec_algo_select}: The images contain same person. Distance: {result['distance']}. Threshold: {threshold}.")
        else:
            print(f"{rec_algo_select}: The images do not contain same person. Distance: {result['distance']}. Threshold: {threshold}.")
        log_to_csv(reference, image, rec_algo_select, prediction, result['distance'], threshold)    
    except Exception as e:
        print(f"{rec_algo}: Unable to detect face on image: {e}")

# Calling of Face-ai-kit library
# ArcFace was used by Face-ai-kit during experiments but it do not have cos metric so in application it is calling only MagFace
# MagFace
def face_ai_kit_model(reference, image, prediction):
    if vektor == "cos":
        print("MagFace only works with euclidean l2 distance. Provided result is for l2 distance:")
    models = ["magface"] 
    face_lib = FaceRecognition(recognition=models[0])
    
    frame_1 = cv2.imread(reference)
    frame_2 = cv2.imread(image)
    
    results1 = face_lib.face_detection(frame_1, align='keypoints')
    face_img1 = results1[0]["img"]

    results2 = face_lib.face_detection(frame_2, align='keypoints')
    face_img2 = results2[0]["img"]

    distance = face_lib.verify(face_img1, face_img2)

    if distance < 1.1:
        print(f"MagFace: The images contain same person. Distance: {distance}. Threshold: 1.1.")
    else:
        print(f"MagFace: The images do not contain same person. Distance: {distance}. Threshold: 1.1.")
    
    log_to_csv(reference, image, 'MagFace', prediction, distance, 1.1)

# "Main" of the program
if __name__ == '__main__':
    images,order = argument_parse()   # Parsing arguments
    
    model_name = model_select(order)  # Selecting the right model according to arguments
    
    optimizer = Adam(learning_rate=1e-4)    # Optimizer initialization with learning rate 1e-4
    
    model = load_model(f'{model_name}.h5', compile=False)   # Recompiling the model to avoid errors with using not same version of tensorflow as it was trained

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  
    
    if len(images) == 1:
        vgg_model(images[0])
            
    elif len(images) == 2:
        siamese_model(images[0], images[1])