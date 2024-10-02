import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--path",
    type=str,
    help="Specify path to the directory containing the images to be processed",
    default="test_images")

    parser.add_argument(
    "--show",
    type=str,
    help="Show each processed image",
    default="n")


    args = parser.parse_args()

    # get list of files in the directory
    files = os.listdir(args.path)
    
    # remove any extra files MAC might have put in there
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
        print("Processing image: ", file)
        try:
            orig_img = cv2.imread(args.path + "/" + file)
            #show_image(orig_img, "original image")

            img = add_in_painting(orig_img)

            img = gamma_correction(img)

            # experimental method I made to try and remove dark patches of noise
            img = dark_noise_replacement(img)
            
            img = clahe(img)

            img = median_filter(img)
            
            img = fix_perspective(img)


            if args.show == "y":
                show_image(orig_img, "unprocessed image")  
                show_image(img, "processed image")     
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
            
            cv2.imwrite("results/" + file, img)
        
        # print exception and continue
        except Exception as e:
            print("Exception: ", e)
            continue


def show_image(img, name="image"):
    cv2.imshow(name, img)
    return

def add_in_painting(img):
    # load mask of the hole to fill
    mask = cv2.imread("masks/inpaint_mask.jpg", cv2.IMREAD_GRAYSCALE)  
    img = img.astype(np.uint8)
    mask = mask.astype(np.uint8)

    # paint the hole using inpainting
    img = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

    return img

def dark_noise_replacement(img):
    eye_mask = cv2.imread("masks/eye_mask.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_and(img, img, mask=eye_mask)

    # convert to ycrcb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # for all brightness values less than 25, make a mask that includes them
    mask = cv2.inRange(ycrcb_img[:,:,0], 0, 25)
    # now apply a mask to the image to ensure that it is only pixels on the eye
    mask = cv2.bitwise_and(mask, eye_mask)

    # display the mask to make sure it is correct
    #show_image(mask, "mask")

    # paint anything in that mask using inpainting
    img = cv2.inpaint(img,mask,10,cv2.INPAINT_NS)
    return img


def plot_histogram(img, file_name, gray_scale = False):
    #https://docs.opencv.org/3.4/d1/db7/tutorial_py_histogram_begins.html
    # used for plotting the histogram

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
    plt.show()
    return


# taken from https://github.com/atapour/ip-python-opencv/blob/main/gamma_correction.py
def gamma_correction(img, gamma=0.8):
    img = ((np.power(img/255, gamma))*255).astype('uint8')

    return img


# help from https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
# and https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143
def fix_perspective(img):
    rows, cols = img.shape[:2]

    src_points = np.float32([[25,125], [235,115], [125,9], [143,235]])

    '''
    CODE TO DRAW THE POINTS ON THE IMAGE
    for point in src_points:
        get the x and y coordinates of the point
        x = int(point[0])
        y = int(point[1])
        img = cv2.circle(img, (x,y), radius=3, color=(0, 255, 0), thickness=-1,)  
    
    
    '''
    dst_points = np.float32([[10,125], [240,125], [130,23], [130,232]]) 
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # return the image with the perspective transform applied
    return cv2.warpPerspective(img, projective_matrix, (cols,rows))


def median_filter(img):
    img = cv2.medianBlur(img,3)
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

def clahe(img, clipLimit=1, tileGridSize=(3, 3)):
    
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # apply CLAHE to the Y channel
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return img


if __name__ == "__main__":
    main()