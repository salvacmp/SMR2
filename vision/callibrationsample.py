import cv2
from time import sleep
# Open the default camera
cam = cv2.VideoCapture(2)

for i in range(250):
    result, image = cam.read()

    if result:
        
        # show the image
        cv2.imshow("sample", image)
        print(f"Taking Picture {i}")
        # sleep(1)
        # save the image
        cv2.imwrite(f"callibration/{i}.png", image)

    else:
        print("No image detected. Please! try again") 

