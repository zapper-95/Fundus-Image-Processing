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
from skimage.restoration import inpaint
import math


def add_in_painting(img):
    mask = cv2.imread("inpaint_mask.jpg", cv2.IMREAD_GRAYSCALE)
    
    img = img.astype(np.uint8)
    mask = mask.astype(np.uint8)

    img_result = inpaint.inpaint_biharmonic(img, mask, channel_axis=-1)
    # scale the image to 0-255
    img_result = (img_result * 255).astype(np.uint8)
    # convert img_result to uint8

    img = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

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
                orig_img = cv2.imread(args.path + "/" + file)
                img = add_in_painting(orig_img)
                cv2.imwrite("ns.png", img)
                '''
                img = gamma_correction(img, 0.8)


                img = remove_salt_pepper_noise(img)
                

                img = clahe(img, 1, (3,3))

                #show_image(img , "img1")

                img = remove_noise(img)
                
                '''
                #img = fix_perspective(orig_img)
                
            


                show_image(img, "final image")
                
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                cv2.imwrite("Results/" + file, img)

        
        # print exception and continue
        except Exception as e:
            print("Exception: ", e)
            continue



def brightness_correction(img):
    return

def remove_salt_pepper_noise(img):
    eye_mask = cv2.imread("eye_mask.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_and(img, img, mask=eye_mask)

    #show_image(img, "original image")

    # convert to ycrcb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    #display the brightness channel
    #show_image(ycrcb_img[:,:,0], "brightness channel")
    # for all brightness values less than 100, make a mask that includes them
    # but also also is in the eye_mask
    mask = cv2.inRange(ycrcb_img[:,:,0], 0, 25)

    
    mask = cv2.bitwise_and(mask, eye_mask)
    # display the mask to make sure it is correct
    #show_image(mask, "mask")

    # paint anything in that mask using inpainting
    img = cv2.inpaint(img,mask,10,cv2.INPAINT_NS)
    #img = inpaint.inpaint_biharmonic(img, mask, channel_axis=-1)
    # scale the image to 0-255
    #img = (img * 255).astype(np.uint8)
    return img

def show_image(img, name="image"):
    cv2.imshow(name, img)



def plot_histogram(img, file_name, gray_scale = False):
    #https://docs.opencv.org/3.4/d1/db7/tutorial_py_histogram_begins.html
    plt.rcParams.update({'font.size': 15})

    if (gray_scale):
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        plt.plot(hist)
    else:
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(hist,color = col)
    
    plt.xlim([0,256])
    plt.ylabel("Count")
    plt.xlabel("Pixel Value")
    plt.title("Histogram of " + file_name)
    # increase font size
    plt.show()

    return


def gamma_correction(img, gamma=0.7):
    img = ((np.power(img/255, gamma))*255).astype('uint8')

    return img

def fix_perspective(img):
    # ellipse = cv2.ellipse(img, (130,125), (105,145), -7, 0, 360, (255,255,255), -1)
    #circle = cv2.circle(img, (130,125), 120, (255,255,255), -1)
    
    rows, cols = img.shape[:2]

    src_points = np.float32([[25,125], [235,115], [125,9], [143,235]])

    #draw the src points on the image. Make the colour green

    for point in src_points:
        # get the x and y coordinates of the point
        x = int(point[0])
        y = int(point[1])

        #img = cv2.circle(img, (x,y), radius=3, color=(0, 255, 0), thickness=-1,)

    #cv2.imshow("image", img)


    #cv2.waitKey(0)
    dst_points = np.float32([[10,125], [240,125], [130,23], [130,232]]) 
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, projective_matrix, (cols,rows))


def remove_noise(img):
    sigma_r = 10
    sigma_s = 30
        # apply billateral filtering
    #img = cv2.bilateralFilter(img, 9, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)


    # apply non local means denoising
    #img = cv2.fastNlMeansDenoisingColored(img,None,4,4,7,21)
    
    
    #remove salt and pepper noise
    img = cv2.medianBlur(img,3)
    #img  = cv2.bilateralFilter(img,9,75,75)
    return img

def equalize_histogram(img):
    # credit to https://www.etutorialspoint.com/index.php/311-python-opencv-histogram-equalization
    # https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


    return img

def clahe(img, clipLimit=1, tileGridSize=(8, 8)):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return img


if __name__ == "__main__":
    main()