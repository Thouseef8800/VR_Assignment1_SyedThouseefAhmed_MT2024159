# VR_Assignment1_SyedThouseefAhmed_MT2024159
============================================================================================
## Part 1: Use computer vision techniques to Detect, segment, and count coins from an image containing scattered Indian coins.

### Steps Implemented:
- **Detection:**
  - Used Canny edge detection to identify coin edges by converting the image to gray scale as canny's work best with single channeled image and also smoothened the image with gaussian blur.
  - morphological closing is used after detecting edges with the Canny edge detector to fill small gaps in the edges of the coins. This helps ensure that each coin’s outline is continuous and closed
  - Extracted the outmost contour as it the outline of the coin and marked the outline on the coins.
- **Segmentation:**
   -An empty black image of the same size as the original is created to serve as a mask
   -For each detected coin, the code finds the minimum enclosing circle and draws it on the mask in green
   -The mask is blended with the original image using transparency (0.7 original, 0.3 mask
   -For each detected coin, the bounding rectangle is found, and the coin is cropped from the original image and saved as a separate image
- **Counting:**
  -The total number of detected coins is calculated from the contours list
  -For each detected contour, the centroid is found from the minimum enclosing circle, and a number is drawn on the original image
  -The total number of coins is also displayed on the blended image
  ### How to Run:
 ### How to Run:
1. Ensure you have OpenCV installed:  
   ```sh
   pip install opencv-python numpy
2. Make sure to update the input image path in the code as per its location.
3. Open the terminal, navigate to the script location, and run:
   ```sh
   python coin_segmentation.py
### Input and Output Files(images folder):
    * Input Image: WhatsApp Image 2025-02-14 at 23.37.29_1086d849
    * Original Image with Numbers: original_image_with_numbers.png
    * Only Coin Edges: only_coin_edges.png
    * Segmented Coins: green_filled_coins.png
    * Coin Count with Labels: blended_output.png
    * each individual segmented coins:- coin_1.jpg, coin_2.jpg, coin_3.jpg, coin_4.jpg, coin_5.jpg, coin_6.jpg, coin_7.jpg, coin_8.jpg, coin_9.jpg, coin_10.jpg 
-------------------------------------------------------------------------------------------------
## Part 2: Create a stitched panorama from multiple overlapping images.

### Steps Implemented:
- **Key points:**
   - Naming the images in the order from left to right and converting into gray scale
   - Using SIFT algorithm to detect key points and descriptors of the image.
- **Image Stitching:**
   -Detected keypoints and descriptors using the SIFT algorithm for both images. Matched keypoints using the FLANN-based matcher.
   -Applied Lowe’s ratio test to select the best matches and eliminate false positives.
   -Computed the homography matrix using RANSAC, which handles outliers and aligns the images accurately.
   - Used the computed homography to warp the second image to the first image’s perspective.
   - Blended the warped image and the original image into a single panoramic output, adjusting the canvas size to fit both images.
   - Saved keypoints, matched features, and the final stitched image to the output directory.
 ### How to Run:
1. Ensure you have OpenCV installed:  
   ```sh
   pip install opencv-python numpy
2. Make sure to update the input image path in the code as per its location.
3. Open the terminal, navigate to the script location, and run:
   ```sh
   python panoroma.py
### Input and Output Files(images folder):
  #### Example 1:-
    * Input Image1: image1.jpg
    * Input Image2: image2.jpg
    * SIFT on image1:- Keypoints_image1.jpg
    * SIFT on image2:- Keypoints_image2.jpg
    * final result:- stitched_image.jpg
