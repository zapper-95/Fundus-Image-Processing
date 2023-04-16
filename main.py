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

def add_in_painting(img):
    mask = cv2.imread("inpaint_mask.jpg", cv2.IMREAD_GRAYSCALE)

    img = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    return img

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
    if os.path.exists("Results"):
        # recursively remove the directory called results
        for file in os.listdir("Results"):
            os.remove("Results/" + file)
        os.rmdir("Results")

    os.makedirs("Results")

    # Process each image
    for file in files:
        # Read image
        print("Processing image: ", file)
        try:
            # if the file ends in .png, .jpg, or .jpeg, then process it
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                img = cv2.imread(args.path + "/" + file)

                #show_image(img)


                img = add_in_painting(img)
                img = fix_perspective(img)
                img = CLACHE(img)

                img = remove_noise(img)

                #show_image(img)

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

    src_points = np.float32([[25,125], [233,110], [125,9], [143,235]])

    #draw the src points on the image. Make the colour green

    for point in src_points:
        # get the x and y coordinates of the point
        x = int(point[0])
        y = int(point[1])
        #img = cv2.circle(img, (x,y), radius=3, color=(0, 255, 0), thickness=-1,)
    
    dst_points = np.float32([[0,125], [250,125], [130,0], [130,255]]) 
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, projective_matrix, (cols,rows))


def colour_correction(img):
    # display all histograms for each channel
    for i in range(3):
        plt.hist(img[:,:,i].ravel(),256,[0,256])
    plt.show()




    return img


def remove_noise(img):
    sigma_r = 200
    sigma_s = 400
        # apply billateral filtering
    #img = cv2.bilateralFilter(img, 3, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)


    # apply non local means denoising
    #img = cv2.fastNlMeansDenoisingColored(img,None,5,0,7,21)
    
    
    #remove salt and pepper noise
    img = cv2.medianBlur(img,3)
    #img  = cv2.bilateralFilter(img,9,75,75)
    return img


def alter_contrast_brightness(img):
    # credit to https://www.etutorialspoint.com/index.php/311-python-opencv-histogram-equalization
    # https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


    return img

def CLACHE(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return img


def alter_contrast_brightness2(img):

    return img



if __name__ == "__main__":
    main()