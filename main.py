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
import math

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
                
                dft(img)
                #show_image(img)

                #show_image(img, "Original Image")

                img = add_in_painting(img)
                img = fix_perspective(img)

                img = remove_noise(img)
                img = CLACHE(img)

                #show_image(img)
                #cv2.waitKey(0)
                cv2.imwrite("Results/" + file, img)

        
        # print exception and continue
        except Exception as e:
            print("Exception: ", e)
            continue


def show_image(img, name="image"):
    cv2.imshow(name, img)



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
    sigma_r = 10
    sigma_s = 30
        # apply billateral filtering
    #img = cv2.bilateralFilter(img, 9, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)


    # apply non local means denoising
    #img = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)
    
    
    #remove salt and pepper noise
    #img = cv2.medianBlur(img,3)
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return img


def alter_contrast_brightness2(img):

    return img

def dft(img, radius=50, order=5):
    
    # convert to ycrb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = int(gray_frame.shape[1])
    height = int(gray_frame.shape[0])
    dim = (width, height)

    # set up optimized DFT settings

    nheight = cv2.getOptimalDFTSize(height)
    nwidth = cv2.getOptimalDFTSize(width)

    # Performance of DFT calculation, via the FFT, is better for array
    # sizes of power of two. Arrays whose size is a product of
    # 2's, 3's, and 5's are also processed quite efficiently.
    # Hence we modify the size of the array to the optimal size (by padding
    # zeros) before finding DFT.

    pad_right = nwidth - width
    pad_bottom = nheight - height
    nframe = cv2.copyMakeBorder(
        gray_frame,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0)

        # perform the DFT and get complex output

    dft = cv2.dft(np.float32(nframe), flags=cv2.DFT_COMPLEX_OUTPUT)

    # shift it so that we the zero-frequency, F(0,0), DC component to the
    # center of the spectrum.

    dft_shifted = np.fft.fftshift(dft)



    lp_filter = create_butterworth_low_pass_filter(nwidth, nheight, radius, order)
    lo_dft_filtered = cv2.mulSpectrums(dft_shifted, lp_filter, flags=0)

    # shift back to original quaderant ordering

    lo_dft = np.fft.fftshift(lo_dft_filtered)

    # recover the original image via the inverse DFT

    lo_filtered_img = cv2.dft(lo_dft, flags=cv2.DFT_INVERSE)

    # normalized the filtered image into 0 -> 255 (8-bit grayscale) 
    # so we can see the output

    # low pass filter output

    lo_min_val, lo_max_val, lo_min_loc, lo_max_loc = \
        cv2.minMaxLoc(lo_filtered_img[:, :, 0])
    lo_filtered_img_normalised = lo_filtered_img[:, :, 0] * (
        1.0 / (lo_max_val - lo_min_val)) + ((-lo_min_val) / (lo_max_val - lo_min_val))
    lo_filtered_img_normalised = np.uint8(lo_filtered_img_normalised * 255)

    # calculate the magnitude spectrum and log transform + scale for visualization


    lo_magnitude_spectrum = np.log(cv2.magnitude(
        lo_dft_filtered[:, :, 0], lo_dft_filtered[:, :, 1]))

    magnitude_spectrum = np.log(cv2.magnitude(
        dft_shifted[:, :, 0], dft_shifted[:, :, 1]))

    # create 8-bit images to put the magnitude spectrum into

    magnitude_spectrum_normalised = np.zeros((nheight, nwidth, 1), np.uint8)

    # normalized the magnitude spectrum into 0 -> 255 (8-bit grayscale) so
    # we can see the output

    cv2.normalize(
        np.uint8(magnitude_spectrum),
        magnitude_spectrum_normalised,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX)

    # convert back to colour for visualisation

    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    magnitude_spectrum_normalised = cv2.cvtColor(magnitude_spectrum_normalised, cv2.COLOR_GRAY2BGR)
    lo_filtered_img_normalised = cv2.cvtColor(lo_filtered_img_normalised, cv2.COLOR_GRAY2BGR)
    lp_filter_vis = cv2.cvtColor(np.uint8(lp_filter[:, :, 0] * 255), cv2.COLOR_GRAY2BGR)

    cv2.imshow('original', gray_frame)
    cv2.imshow('magnitude spectrum', magnitude_spectrum_normalised)
    cv2.imshow('low pass filtered image', lo_filtered_img_normalised)
    cv2.imshow('low pass filter', lp_filter_vis)

    cv2.waitKey(0)
    

def create_low_pass_filter(width, height, radius):
    lp_filter = np.zeros((height, width, 2), np.float32)
    cv2.circle(lp_filter, (int(width / 2), int(height / 2)),
               radius, (1, 1, 1), thickness=-1)
    return lp_filter

def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, lp_filter.shape[1]):  # image width
        for j in range(0, lp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            lp_filter[j, i] = 1 / (1 + math.pow((radius / d), (2 * n)))
    return lp_filter

if __name__ == "__main__":
    main()