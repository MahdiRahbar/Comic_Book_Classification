import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math

from classification import training, getLabel

SIGNS = ["SPIDERMAN",
        "BATMAN",
        "NEITHER"]

### Clean all previous file
def clean_images():
	file_list = os.listdir('./')
	for file_name in file_list:
		if '.png' in file_name:
			os.remove(file_name)

### Preprocess image
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_hist_equalized)
    y = cv2.equalizeHist(y)
    img_hist_equalized = cv2.merge((y, cr, cb))
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image

def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

### Find Comics
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids =\
            cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE    )
    cnts = cnts[0]
    return cnts

def contourIsComic(perimeter, centroid, threshold):
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]

def cropComic(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    return image[top:bottom,left:right]

def findLargestComic(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_comic, distance = contourIsComic(c, [cX, cY], 1-threshold)
        if is_comic and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            comic = cropComic(image,coordinate)
    return comic, coordinate

def findComics(image, contours, threshold, distance_theshold):
    comics = []
    coordinates = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_comic, max_distance = contourIsComic(c, [cX, cY], 1-threshold)
        if is_comic and max_distance > distance_theshold:
            comic = cropContour(image, [cX, cY], max_distance)
            comics.append(comic)
            coordinate = np.reshape(c, [-1,2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinates.append([(top-2,left-2),(right+1,bottom+1)])
    return comics, coordinates

def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)
    binary_image = removeSmallComponents(binary_image, min_size_components)
    binary_image = cv2.bitwise_and(binary_image,binary_image,\
            mask=remove_other_color(image))
    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)
    comic, coordinate = findLargestComic(original_image, contours,\
            similitary_contour_with_circle, 15)
    text = ""
    comic_type = -1
    i = 0

    if comic is not None:
        comic_type = getLabel(model, comic)
        comic_type = comic_type if comic_type <= 8 else 8
        text = COMICS[comic_type]
        cv2.imwrite(str(count)+'_'+text+'.png', comic)

    if comic_type > 0 and comic_type != current_comic_type:
        cv2.rectangle(original_image, coordinate[0],coordinate[1],\
                (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image,\
                text,(coordinate[0][0], coordinate[0][1] -15),font,\
                1,(0,0,255),2,cv2.LINE_4)
    return coordinate, original_image, comic_type, text

def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),2)
    return cv2.bitwise_and(img, img, mask=mask)

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    return mask


### Main function
def main(args):

    #Clean previous image
    clean_images()

    #Training phase
    model = training()

    success = True
    similitary_contour_with_circle = 0.65   # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w")
    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        frame = cv2.resize(frame, (640,480))

        print("Frame:{}".format(count))
        coordinate, image, sign_type, text = localization(frame, args.min_size_components, args.similitary_contour_with_circle, model, count, current_sign)
        if coordinate is not None:
            cv2.rectangle(image, coordinate[0],coordinate[1], (255, 255, 255), 1)
        print("Sign:{}".format(sign_type))
        if sign_type > 0 and (not current_sign or sign_type != current_sign):
            current_sign = sign_type
            current_text = text
            top = int(coordinate[0][1]*1.05)
            left = int(coordinate[0][0]*1.05)
            bottom = int(coordinate[1][1]*0.95)
            right = int(coordinate[1][0]*0.95)

            position = [count, sign_type if sign_type <= 8 else 8, coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]]
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NLP Assignment Command Line")

    parser.add_argument(
      '--min_size_components',
      type = int,
      default= 300,
      help= "Min size component to be reserved"
      )

    parser.add_argument(
      '--similitary_contour_with_circle',
      type = float,
      default= 0.65,
      help= "Similitary to a circle"
      )

    args = parser.parse_args()
    main(args)
