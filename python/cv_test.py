import cv2

img = cv2.imread('lena.bmp')
cv2.namedWindow('test')
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyWindow('test')
