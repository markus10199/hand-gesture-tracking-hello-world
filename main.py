#############
# imports
#############

import cv2
import pygame

from HandDetector import HandDetector

#########
# main function
#########

def main():
    handDetector = HandDetector()

    while handDetector.shouldClose == False:
        handDetector.update()
        if len(handDetector.landmarkDictionary) > 0: 
            print(handDetector.landmarkDictionary[0])

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



