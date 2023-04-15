'''
Your program must contain an argument parser in the main script that allows a directory containing images to be specified. I should be able to run your program in the following way: 
python main.py image-processing-files/test_images/ 
from which it will cycle through the images in the specified directory, perform all the processing and save the images without
 changing the filenames in a directory called “results”. 
 
The contents of this directory will also be a part of your submission.
'''

import argparse
import os
import cv2
import numpy as np


def edge_sharpener(img,kernel_size,kernel_size2, k):    
    gaussian_img = cv2.GaussianBlur(img,(kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    # apply a laplacian mask to the gaussian blurred image
    laplacian_img = cv2.Laplacian(gaussian_img,-1,ksize=kernel_size2)
    # subtract the laplacian from the original image
    scaled_lap = laplacian_img*k

    scaled_lap=scaled_lap.astype('uint8') # have to convert back to int 
    sharp_img = cv2.subtract(gaussian_img,scaled_lap)
    return sharp_img


def main():
    print("hello world")
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the directory containing the images to be processed")
    args = parser.parse_args()

    # Get list of files in the directory
    files = os.listdir(args.path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")


    # Create a directory to store the results
    # if it exists, overwrite it
    if os.path.exists("results"):
        # recursively remove the directory called results
        for file in os.listdir("results"):
            os.remove("results/" + file)
        os.rmdir("results")

    os.makedirs("results")

    # Process each image
    for file in files:
        # Read image
        try:
            # if the file ends in .png, .jpg, or .jpeg, then process it
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                img = cv2.imread(args.path + "/" + file)



                sigma_r = 250
                sigma_s = 600

                # apply billateral filtering
                #img = cv2.bilateralFilter(img, 15, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)


                # apply non local means denoising
                #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) 

                # remove salt and pepper noise
                img = cv2.medianBlur(img,3)
                
                # sharpen the edges
                #img = edge_sharpener(img,5,3,0.8)


                cv2.imwrite("results/" + file, img)

        
        # print exception and continue
        except Exception as e:
            print("Exception: ", e)
            continue



if __name__ == "__main__":
    main()