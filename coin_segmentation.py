# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:33:47 2025

@author: thous
"""

import cv2
import numpy as np
import os

# Load the image
image_path = "D:/iiitb/sem2/vr/vr assignment 1/WhatsApp Image 2025-02-14 at 23.37.29_1086d849.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

# Detect edges using Canny
edges = cv2.Canny(blurred, 50, 150)

# Apply Morphological Closing to remove small gaps
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours (only external)
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create output directory for segmented coins
output_dir = "segmented_coins"
os.makedirs(output_dir, exist_ok=True)

# Draw only the outer edge of coins
output = np.zeros_like(image)
cv2.drawContours(output, contours, -1, (255, 255, 255), 2)  # Draw in white

# Create a blank image of the same size as the original
mask = np.zeros_like(image)

# Loop through each detected contour and extract individual coins
total_coins = len(contours)
for i, contour in enumerate(contours):
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract and save segmented coin
    segmented_coin = image[y:y+h, x:x+w]
    coin_path = os.path.join(output_dir, f"coin_{i+1}.png")
    cv2.imwrite(coin_path, segmented_coin)
    
    # Create an individual image for each coin
    coin_output = np.zeros_like(image)
    coin_output[y:y+h, x:x+w] = segmented_coin
    cv2.imwrite(os.path.join(output_dir, f"coin_output_{i+1}.png"), coin_output)
    
    # Get the minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    center = (int(cx), int(cy))
    radius = int(radius)

    # Draw a filled green circle on the mask
    cv2.circle(mask, center, radius, (0, 255, 0), -1)  # -1 fills the circle completely
    
    # Draw number on the coin
    cv2.putText(image, str(i+1), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

# Blend the green mask with the original image
blended = cv2.addWeighted(image, 0.7, mask, 0.3, 0)  # 70% original, 30% green overlay

# Display the total count
print(f"Total Number of Coins Detected: {total_coins}")

# Show the count on the image
cv2.putText(blended, f"Total Coins: {total_coins}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 0, 255), 2, cv2.LINE_AA)

# Loop through each detected contour and extract individual coins
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
    coin = image[y:y+h, x:x+w]  # Crop coin region
    
    # Save the segmented coin
    coin_path = os.path.join(output_dir, f"coin_{i+1}.jpg")
    cv2.imwrite(coin_path, coin)

    # Show each segmented coin separately
    cv2.imshow(f"Coin {i+1}", coin)
    cv2.waitKey(500)  # Display each coin for 500ms
    
# Resize images for display
image_resized = cv2.resize(image, (400, 400))
output_resized = cv2.resize(output, (400, 400))
blended_resized = cv2.resize(blended, (400, 400))
mask_resized = cv2.resize(mask, (400, 400))

cv2.imwrite(os.path.join(output_dir, "segmented_coins_with_bounding_boxes.png"), image)
cv2.imwrite(os.path.join(output_dir, "original_image_with_numbers.png"), image_resized)
cv2.imwrite(os.path.join(output_dir, "only_coin_edges.png"), output_resized)
cv2.imwrite(os.path.join(output_dir, "green_filled_coins.png"), mask_resized)
cv2.imwrite(os.path.join(output_dir, "blended_output.png"), blended_resized)

# Display results
cv2.imshow("Segmented Coins with Bounding Boxes", image)
cv2.imshow("Original Image with Numbers", image_resized)
cv2.imshow("Only Coin Edges", output_resized)
cv2.imshow("Green Filled Coins", mask_resized)
cv2.imshow("Blended Output", blended_resized)  # Blended image with green overlay

# cv2.imwrite()

cv2.waitKey(0)
cv2.destroyAllWindows()
