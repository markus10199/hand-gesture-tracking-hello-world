#############
# imports
#############

import enum
import cv2
import mediapipe as mp # contains gesture recognition ml model
import pygame

# load the video from the webcam
capture = cv2.VideoCapture(0)

#### setting parameters for the webcam capture
capture.set(3, 640) # the width of the window
capture.set(4, 480) # height of the window

# getting hand identification models
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# get built in drawing tools for drawing hand detection
mpDraw = mp.solutions.drawing_utils

# clock object to help track framerate
clock = pygame.time.Clock()

# loop through every frame of the video and show it (stops when you reach the end of the video)
while capture.isOpened():
    # read the next frame of the video. "isLoaded" is a boolean representing whether the frame was loaded or not
    isLoaded, img = capture.read()

    if isLoaded == True:
        # the hand predictive model was trained on RGB images
        # so we need to convert our image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # results of hand model on our image
        results = hands.process(imgRGB)

        #print(results.multi_hand_landmarks)

        # record and display the framerate of the video
        timePassed = clock.tick()
        fps = 1/(timePassed/1000)


        # draw the framerate as text
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,0))

        # if there are any hands detected on the image (via their landmarks)
        if results.multi_hand_landmarks:
            # for every set of landmarks (1 set per hand) in the image
            for landmarkSet in results.multi_hand_landmarks:
                # for every landmark in the current set of landmarks
                for id, landmark in enumerate(landmarkSet.landmark):
                    # get the width height and channels of the image
                    height, width, channels = img.shape
                    # convert the position of the landmarks from a fraction to a location on the img
                    xPos, yPos = int(landmark.x*width), int(landmark.y*height)
                    # print out the results
                    print(id, xPos, yPos)


                # draw landmarks on the hand
                mpDraw.draw_landmarks(img, landmarkSet, mpHands.HAND_CONNECTIONS)

        # show the current frame in a window titled "Webcam feed"
        cv2.imshow("Webcam feed", img)
        # if you press the q key or click on the exit button, then close the video window (& is bitwise "and")
        # This also makes it wait 1ms between each frame
        # Note: "getWindowProperty"
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Webcam feed", 0) < 0:
            break

    # if an image is not loaded, leave the loop
    else:
        break

# When everything done, release the video capture object
capture.release()
   
# Closes all the frames
cv2.destroyAllWindows()




