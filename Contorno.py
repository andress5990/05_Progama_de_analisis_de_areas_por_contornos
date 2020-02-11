import numpy as np  
import cv2 
import sys 


img = cv2.imread('02.jpg', 0)
cv2.imshow('image', img)
cv2.waitKey(0)