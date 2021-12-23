import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage import measure
import sys

def ploter(image,si=[12,12]):
    fig, ax = plt.subplots(figsize=si);ax.imshow(image,cmap='gray')
    ax.get_xaxis().set_visible(False);ax.get_yaxis().set_visible(False)
    plt.show()

def arrange_th(labels):   
    
    if labels < 4000:
        return 5000, 999999
    
    elif labels > 4000 and labels < 6000:
        return 7000, 999999
    
    elif labels > 6000 and labels < 8000:
        return 8000, 999999
    
    else:
        return 9000, 999999

def conn_comp_label(src_img):
    # perform connected components analysis on the thresholded images and initialize the
    # mask to hold only the "large" components we are interested in
    labels = measure.label(src_img, neighbors=4, background=0)
    mask = np.zeros(src_img.shape, dtype="uint8")
    print("[INFO] found {} blobs".format(len(np.unique(labels))))
    lower_th, upper_th = arrange_th(len(np.unique(labels)))
    
 
    # loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # if this is the background label, ignore it
        if label == 0:
            #print("[INFO] label: 0 (background)")
            continue
 
        # otherwise, construct the label mask to display only connected components for
        # the current label
        #print("[INFO] label: {} (foreground)".format(i))
        labelMask = np.zeros(src_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv.countNonZero(labelMask)
 
        # if the number of pixels in the component is sufficiently large, add it to our
        # mask of "large" blobs
        if numPixels > lower_th and numPixels < upper_th:
            mask = cv.add(mask, labelMask)

    return mask
    
def find_biggest_label(src_img):
    labels = measure.label(src_img, neighbors=4, background=0)
    biggest_label = np.zeros(src_img.shape, dtype="uint8")
    list = []
    for (i, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue
        
        labelMask = np.zeros(src_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv.countNonZero(labelMask)
        # save the pixel size to list
        list.append(numPixels)
    
    #find the biggest object in the image
    max_value = max(list)
    label_index = list.index(max_value)
    
    labelMask = np.zeros(src_img.shape, dtype="uint8")
    labelMask[labels == label_index + 1] = 255
    biggest_label = cv.add(biggest_label, labelMask)
    
    return biggest_label

def line_counter(src_img):
    canny_img = cv.Canny(src_img, 50, 200, None, 3)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1  # minimum number of pixels making up a line
    max_line_gap = 30 # maximum gap in pixels between connectable line segments
    #line_image = np.copy(img) * 0  # creating a blank to draw lines on
    line_image = np.zeros(src_img.shape, dtype="uint8")

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(canny_img, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    if lines is not None:
        return len(lines)
    else:
        return 0

def find_rectangle(src_img):
    # Scan the labels of the entire image, save the ones that has line values between 5 from 15.
    labels = measure.label(src_img, neighbors=4, background=0)
    rect_img = np.zeros(src_img.shape, dtype="uint8")
    
    for (i, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue
        
        labelMask = np.zeros(src_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        line_cnt = line_counter(labelMask)
        
        if line_cnt >=5 and line_cnt <= 30:
            # add to the image
            rect_img = cv.add(rect_img, labelMask)
    
    return rect_img 




def sign(src_img):
    print("Image: ", src_img)
    img = cv.imread(src_img, 0)
    
    # Normalize the image
    img = cv.equalizeHist(img)
    print("NORMALIZED IMAGE")
    ploter(img)
    
    # Sharpen the image
    kernel = np.array([[0,-1,0],
                  [-1,5,-1],
                  [0,-1,0]])
    img = cv.filter2D(img,-1,kernel)
    print("SHARPENED IMAGE")
    ploter(img)

    # Threshold the image
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
    cv.THRESH_BINARY_INV, 139, 3)
    print("THRESHOLDED IMAGE")
    ploter(thresh)
    
    kernel_open = np.ones((2,2),np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_open,iterations = 2)
    print("OPENED IMAGE")
    ploter(opening)
    
    #median = cv.medianBlur(thresh,9)
    #print("MEDIAN IMAGE")
    #ploter(median)

    mask = conn_comp_label(opening)
    print("LABELED IMAGE")
    ploter(mask)
    
    kernel_ero = np.ones((3,3),np.uint8)
    dilate = cv.dilate(mask,kernel_ero,iterations = 1)
    print("DILADED IMAGE")
    ploter(dilate)
    
    # Use closing operator
    kernel_close = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel_close,iterations = 6)
    print("CLOSED IMAGE")
    ploter(closing)
    
    kernel_ero = np.ones((4,4),np.uint8)
    erosion = cv.erode(closing,kernel_ero,iterations = 2)
    print("ERODED IMAGE")
    ploter(erosion)
    

    # Find the rectangles in the image
    rectangle_img = find_rectangle(erosion)
    numPixels = cv.countNonZero(rectangle_img)
    if numPixels == 0:
        kernel_close = np.ones((8,8),np.uint8)
        closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel_close,iterations = 10)
        rectangle_img = find_rectangle(closing)
    print("RECTANGLE IMAGE")
    ploter(rectangle_img)

    # If the image has one object, then use the closing with 10 iterations.
    #closing = cv.morphologyEx(rectangle_img, cv.MORPH_CLOSE, kernel,iterations = 10)
    #print("CLOSED IMAGE AFTER RECTANGLED")
    #ploter(closing)

    # Detect the edges in the rectangle
    canny_img = cv.Canny(rectangle_img, 50, 200, None, 3)
    print("CANNY IMAGE")
    ploter(canny_img)
    
    ######## Show the signboards location in the image ########
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1  # minimum number of pixels making up a line
    max_line_gap = 30 # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(canny_img, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
    lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)
    print("HOUGH LINED IMAGE")
    ploter(lines_edges)