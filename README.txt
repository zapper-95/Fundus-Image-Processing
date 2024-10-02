Image Processing Pipeline:
Inpaint -> Gamma Correction -> Dark Noise Replacement -> CLAHE -> Median Filter -> Fix Perspective 

Note: There are some commented out parts that may be useful to uncomment. For example, I have commented out adding the source points when I apply a projective transformation. This might be informative to look at. Additionally, I have commented out where I show the original and processed image. Also, you should note that as the code is running, it outputs the current image it is processing.

Accuracy of 0.8 on test images