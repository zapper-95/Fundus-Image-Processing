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
from matplotlib import pyplot as plt

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
        print("Processing image: ", file)
        try:
            # if the file ends in .png, .jpg, or .jpeg, then process it
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                img = cv2.imread(args.path + "/" + file)

                #show_image(img)

                img = remove_noise(img)
                #img = alter_contrast_brightness(img)
                img = fix_perspective(img)
                #plot_histogram(img)
                
                #show_image(img)
                
                # sharpen the edges
                img = edge_sharpener(img,3,3,2)

                img = remove_noise(img)
                



                cv2.imwrite("Results/" + file, img)

        
        # print exception and continue
        except Exception as e:
            print("Exception: ", e)
            continue


def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)


def plot_histogram(img):
     plt.hist(img.ravel(),256,[0,256])
     plt.show()


def fix_perspective(img):
    # ellipse = cv2.ellipse(img, (130,125), (105,145), -7, 0, 360, (255,255,255), -1)
    #circle = cv2.circle(img, (130,125), 120, (255,255,255), -1)
    
    rows, cols = img.shape[:2]

    src_points = np.float32([[25,125], [235,125], [130,0], [130,250]])
    
    #draw the src points on the image. Make the colour green

    for point in src_points:
        # get the x and y coordinates of the point
        x = int(point[0])
        y = int(point[1])
        #img = cv2.circle(img, (x,y), radius=3, color=(0, 255, 0), thickness=-1,)
    
    dst_points = np.float32([[0,125], [250,125], [130,-20], [130,270]]) 
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, projective_matrix, (cols,rows))



def remove_noise(img):
    sigma_r = 400
    sigma_s = 400
        # apply billateral filtering
    #img = cv2.bilateralFilter(img, 3, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)


    # apply non local means denoising
    #img = cv2.fastNlMeansDenoisingColored(img, 8, 8, 21, 7) 
    
    
    #remove salt and pepper noise
    img = cv2.medianBlur(img,3)
    return img


def alter_contrast_brightness(img):
    # credit to https://www.etutorialspoint.com/index.php/311-python-opencv-histogram-equalization

    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

    # apply histogram equalization 
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


    return img





if __name__ == "__main__":
    main()