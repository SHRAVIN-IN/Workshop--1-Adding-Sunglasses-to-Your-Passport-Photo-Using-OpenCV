# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Developed by:

NAME : JANA SHRAVIN S
REG NO : 212224243003

## Program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read face image
faceImage = cv2.imread("shravinws11.jpeg")

# Read sunglasses PNG with alpha channel
glassPNG = cv2.imread("sung.png", cv2.IMREAD_UNCHANGED)

# Check images
if faceImage is None:
    print("Face image not found")
elif glassPNG is None:
    print("Sunglasses image not found")
else:

    # Position of sunglasses
    x = 330
    y = 260
    w = 400
    h = 170

    # Copy original image
    faceWithGlassesArithmetic = faceImage.copy()

    # Get image size
    imageHeight, imageWidth = faceImage.shape[:2]

    # Make sure ROI does not go outside the image
    actualW = min(w, imageWidth - x)
    actualH = min(h, imageHeight - y)

    # Extract actual face region
    eyeROI = faceWithGlassesArithmetic[y:y+actualH, x:x+actualW]

    # Resize sunglasses to EXACTLY match eyeROI
    glassPNG = cv2.resize(glassPNG, (actualW, actualH))

    # Separate BGR and Alpha channels
    glassBGR = glassPNG[:, :, :3]
    glassMask1 = glassPNG[:, :, 3]

    # Create 3-channel mask
    glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

    # Convert everything to float
    eyeROI = eyeROI.astype(np.float32)
    glassBGR = glassBGR.astype(np.float32)
    glassMask = glassMask.astype(np.float32) / 255.0

    # Blend sunglasses with face
    maskedEye = eyeROI * (1 - glassMask)
    maskedGlass = glassBGR * glassMask

    eyeRoiFinal = maskedEye + maskedGlass

    # Convert back to uint8
    maskedEye = np.uint8(maskedEye)
    maskedGlass = np.uint8(maskedGlass)
    eyeRoiFinal = np.uint8(eyeRoiFinal)

    # Display intermediate results
    plt.figure(figsize=(20, 7))

    plt.subplot(131)
    plt.imshow(maskedEye[:, :, ::-1])
    plt.title("Masked Eye Region")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(maskedGlass[:, :, ::-1])
    plt.title("Masked Sunglass Region")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(eyeRoiFinal[:, :, ::-1])
    plt.title("Augmented Eye and Sunglass")
    plt.axis("off")

    plt.show()

    # Put sunglasses back into image
    faceWithGlassesArithmetic[y:y+actualH, x:x+actualW] = eyeRoiFinal

    # Display final result
    plt.figure(figsize=(16, 9))

    plt.subplot(121)
    plt.imshow(faceImage[:, :, ::-1])
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(faceWithGlassesArithmetic[:, :, ::-1])
    plt.title("With Sunglasses")
    plt.axis("off")

    plt.show()

    # Save output
    cv2.imwrite("output_with_glasses.jpg", faceWithGlassesArithmetic)

    print("Saved as output_with_glasses.jpg")
```
## output
<img width="1260" height="463" alt="download" src="https://github.com/user-attachments/assets/b3b99c26-6685-4b3e-926b-d173fd958e52" />
