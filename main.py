#############
# imports
#############

import cv2
import mediapipe as mp # contains gesture recognition ml model
import time

# load the video from the webcam
capture = cv2.VideoCapture(0)

#### setting parameters for the webcam capture
capture.set(3, 640) # the width of the window
capture.set(4, 480) # height of the window
#capture.set(10, 100)

# loop through every frame of the video and show it (stops when you reach the end of the video)
while capture.isOpened():
    # read the next frame of the video. "isLoaded" is a boolean representing whether the frame was loaded or not
    isLoaded, img = capture.read()

    if isLoaded == True:
        # show the current frame in a window titled "Lab vid"
        cv2.imshow("Lab vid", img)
        # if you press the q key or click on the exit button, then close the video window (& is bitwise "and")
        # This also makes it wait 25ms between each frame
        # Note: "getWindowProperty"
        if cv2.waitKey(25) & 0xFF == ord('q') or cv2.getWindowProperty("Lab vid", 0) < 0:
            break

    # if an image is not loaded, leave the loop
    else:
        break

# When everything done, release the video capture object
capture.release()
   
# Closes all the frames
cv2.destroyAllWindows()




