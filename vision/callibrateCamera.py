import numpy as np
import cv2 as cv
import glob
from time import sleep
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('callibration/*.png')
# print(len(images))
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(f"Processing {fname}")
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        # cv.imshow('img', img)
        
        # cv.waitKey(500)
    

    
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# newcameramtx = np.float32([[906.02659967,   0.        , 327.11629144],
#        [  0.        , 673.70604432, 231.46136242],
#        [  0.        ,   0.        ,   1.        ]])
# mtx = np.float32([[548.99241182,   0.        , 361.8435228 ],
#        [  0.        , 537.82571362, 249.76303655],
#        [  0.        ,   0.        ,   1.        ]])
# dist = np.float32([[-6.06898780e-01,  6.59210727e-01, -9.38203699e-04,
#          2.18868872e-02,  4.95129596e+01]])
# roi = (284, 210, 113, 45)

img = cv.imread('callibration/145.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(repr(newcameramtx))
print(repr(mtx))
print(repr(dist))
print(repr(roi))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
cv.destroyAllWindows()

