# ===================================================================

# Example : apply a specific pre-trained classifier to the test images 
# path to the directory containing images is specified on the command line 
# e.g. python classify.py --data=path_to_data
# path to the pre-trained network weights can be specified on the command line 
# e.g. python classify.py --model=path_to_model
# python classify.py --data=test_images --model=classifier.model

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2022 Amir Atapour Abarghouei

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# ===================================================================

import os
import argparse
import numpy as np
import cv2

# ===================================================================

# parse command line arguments for paths to the data and model

parser = argparse.ArgumentParser(
    description='Perform image classification on test images')

parser.add_argument(
    "--data",
    type=str,
    help="specify path to test images",
    default='test_images')

parser.add_argument(
    "--model",
    type=str,
    help="specify path to model weights",
    default='classifier.model')

args = parser.parse_args()

# ===================================================================

# load model weights:

model =  cv2.dnn.readNetFromONNX(args.model)

# lists to keep filenames, images and identifiers for healthy and sick labels:

names = []
images = []
healthys = []
sicks = []

# the first 20 images are healthy and the next 20 are not:

for i in range(1, 21):
    healthys.append(f'im{str(i).zfill(2)}')

for i in range(21, 41):
    sicks.append(f'im{str(i).zfill(2)}')

# read all the images from the directory

for file in os.listdir(args.data):
    names.append(file)
names.sort()

# remove any extra files Mac might have put in there:

if ".DS_Store" in names:
    names.remove(".DS_Store")

# keeping track of the number of correct predictions for accuracy:
correct = 0

# main loop:
for filename in names:

    # read image:
    img = cv2.imread(os.path.join(args.data, filename))

    if img is not None:

        # pass the image through the neural network:
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (256, 256),(0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward()

        # identify what the predicted label is:
        if(output > 0.5):
            print(f'{filename}: sick')
            if(filename.startswith(tuple(sicks))):
                correct += 1
        else:
            print(f'{filename}: healthy')
            if(filename.startswith(tuple(healthys))):
                correct += 1

# print final accuracy:
print(f'Accuracy is {correct/len(names)}')
        
# ===================================================================