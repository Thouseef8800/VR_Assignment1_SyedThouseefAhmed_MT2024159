# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:21:37 2025

@author: thous
"""


import cv2
import numpy as np
import os

# Load images
image1 = cv2.imread("D:/iiitb/sem2/vr/vr assignment 1/4.jpg")
image2 = cv2.imread("D:/iiitb/sem2/vr/vr assignment 1/5.jpg")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#  Feature Detection & Description using SIFT
sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Draw keypoints on images
keypoints_image1 = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoints_image2 = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#  Feature Matching using FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe’s Ratio Test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matched keypoints (Optional: visualize matching)
matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matched Features", matched_img)
cv2.imwrite("D:/iiitb/sem2/vr/vr assignment 1/stitched_output/Matched Features.png", matched_img)

# Create output directory
output_dir = "stitched_output"
os.makedirs(output_dir, exist_ok=True)

# Save keypoints images
cv2.imwrite(os.path.join(output_dir, "keypoints_image1.jpg"), keypoints_image1)
cv2.imwrite(os.path.join(output_dir, "keypoints_image2.jpg"), keypoints_image2)

#  Extract Keypoints
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    #  Compute Homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    #  Warp Image2 to Image1's perspective
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    panorama_width = width1 + width2

    warped_image2 = cv2.warpPerspective(image2, H, (panorama_width, height1))

    #  Stitch Images Together
    stitched_image = warped_image2.copy()
    stitched_image[0:height1, 0:width1] = image1

    # Save stitched image
    stitched_image_path = os.path.join(output_dir, "stitched_image.jpg")
    cv2.imwrite(stitched_image_path, stitched_image)
  
    #  Save & Display Output
    cv2.imshow("Stitched Image", stitched_image)
    print(f"Stitched image saved at: {stitched_image_path}")
    print(f"Keypoints images saved at: {output_dir}")

else:
    print("Not enough matches found!")

cv2.waitKey(0)
cv2.destroyAllWindows()
