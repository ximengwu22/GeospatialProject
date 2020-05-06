import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import imutils


# importing all the images in a folder
cv_img = [];
avg_img = np.zeros((500, 500, 3), np.float);

imgset = []


cnt = 1
test_img = '393408653*' #
path = 'sample_drive/cam_5/'
for img in glob.glob(path + test_img):
    if cnt > 55:
        break
    a = cv2.imread(img);
    # Redraw the img propotionaly
    b = imutils.resize(a, width=500, height=500);
    print("Image " + img + " imported and resized");
    cv_img.append(b);
    cnt += 1
    imgset.append(img);
    # cv2.imshow('image',b);
    # cv2.waitKey(0);

for img in cv_img:
    img1 = cv2.GaussianBlur(img, (3, 3), 0); # Gaussian Kernel Size:[height width]. eliminate noise
    imarr = np.array(img1, dtype=np.float);
    # avg_img = avg_img+(imarr)/len(cv_img);
    avg_img = avg_img + imarr;

# avg_img = avg_img/len(cv_img)

avg_img = np.array(np.round(avg_img), dtype=np.uint8); # Even number, unsigned integer (0 to 255)
avg = 1
cv2.imwrite("Average"+test_img+".jpg", avg_img);
# displaying the average image
if avg == 1:
    cv2.imshow("Average", avg_img);
    cv2.waitKey(0);

avg_img_grey = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY);

# Binary Threshold - changable
# Adaptive Threshold : [src,dst, ADAPTIVE_THRESH_MEAN_C/ADAPTIVE_THRESH_GAUSSIAN_C,
#                       THRESH_BINARY/THRESH_BINARY_INV,neighbor blockSize, deviation size]
thresh_img = cv2.adaptiveThreshold(avg_img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 69, 0);#11, 2);

# thresh_img = cv2.threshold(avg_img_grey, 101, 255, cv2.THRESH_BINARY_INV)[1];

if avg == 1:
    cv2.imshow("Threshold Gaussian", thresh_img);
    cv2.imwrite("ThresholdImg"+test_img+".jpg", thresh_img);
    cv2.waitKey(0);

# Edge Detection, [src, thre1_min, thre2_max]
edge = cv2.Canny(thresh_img, 100, 220);#75, 100);

if avg == 1:
    cv2.imshow("Edge Detection", edge);
    cv2.imwrite("EdgesDetected"+test_img+".jpg", edge);
    cv2.waitKey(0);

# [contours, hiararchy] = [src, mode=CV_RETR_EXTERNAL/CV_RETR_LIST/CV_RETR_CCOMP/CV_RETR_TREE,
#                           method=CV_CHAIN_APPROX_NONE/CV_CHAIN_APPROX_SIMPLE/CV_CHAIN_APPROX_TC89_L1/CV_CHAIN_APPROX_TC89_KCOS]
cnts, hier = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
cv2.drawContours(avg_img, cnts, -1, (0, 255, 255), 2);
if avg == 1:
    cv2.imshow("Cont", avg_img);
    cv2.imwrite("AllContours"+test_img+".jpg", avg_img);
    cv2.waitKey(0);

# cv2.drawContours(avg_img,cnts,-1,(0,255,0),3);
list1 = [];
mask_img = np.zeros((500, 500, 1), np.float);
avg = 1;    #mask

oimg = cv2.imread(imgset[-1]);#"Average"+test_img+".jpg"); #
print("Read image:", imgset[-1])
oimg = imutils.resize(oimg, width=500, height=500);


for c in cnts:
    # Contour perimeter, [curve, curveClosed?]
    p = cv2.arcLength(c, True);
    # output: curve = [curve, distance=epsilon, curveClosed?]
    approx = cv2.approxPolyDP(c, 0.1 * p, True);
    # Output: [center=[x,y], radius]
    (x, y), radius = cv2.minEnclosingCircle(c);
    radius = int(radius);
    if y >= 387 and y <= 432 and x >=400 and x <= 436:
        print(x,y,radius)
        area = cv2.contourArea(c)
        print(area)
        # print(c)
    # compare smear area,
    if cv2.contourArea(c) > 200 and abs(cv2.contourArea(c) - (3.1415 * (radius ** 2))) < 200:
        cv2.drawContours(oimg, [approx], -1, (255, 255, 0), 2);
        cv2.drawContours(mask_img, [approx], -1, (255, 255, 255), -1);
        list1.append(c);
if avg == 1:
    cv2.imshow("FinalResult", oimg); #oimg);
    cv2.waitKey(0);
    cv2.imshow("Mask", mask_img);
    cv2.waitKey(0);
    cv2.imwrite("FinalResult"+test_img+str(cnt)+".jpg", oimg);
    cv2.imwrite("Mask"+test_img+str(cnt)+".jpg", mask_img);
cv2.destroyAllWindows();