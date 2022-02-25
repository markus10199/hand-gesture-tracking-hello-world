# # # # # # #
# imports
# # # # # # #

import cv2
import pygame

from HandDetector import HandDetector

# # # # # # #
# Global Vars
# # # # # # #

# Define the size of the game window
WIDTH = 1200
HEIGHT = 800
# make the game window object
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
# name the game window
pygame.display.set_caption("Best game")

# # # # # # #
# Functions
# # # # # # #

def mapToNewRange(val, inputMin, inputMax, outputMin, outputMax):
    return outputMin + ((outputMax - outputMin) / (inputMax - inputMin)) * (val - inputMin)



#########
# main function
#########

def main():
    # make a hand detector
    handDetector = HandDetector()
    # keeps track if pygame should keep running
    gameIsRunning = True

    #circle vars
    circleX=0
    circleY=0
    circleZ=0

    # while the opencv window is running
    while not handDetector.shouldClose and gameIsRunning:

        # update the webcam feed and hand tracker calculations
        handDetector.update()

        #fill background
        WINDOW.fill((0,0,0))

        # if there is at least one hand seen, then
        # print out the landmark positions
        if len(handDetector.landmarkDictionary) > 0: 
            circleX = handDetector.landmarkDictionary[0][9][0]
            circleY = handDetector.landmarkDictionary[0][9][1]
            circleZ = handDetector.landmarkDictionary[0][9][2]
            circleZ = abs(circleZ)*5

            # draw circle at point
            pygame.draw.circle(WINDOW, (255,255,0), (circleX, circleY), circleZ)


        # for all the game events
        for event in pygame.event.get():
             # if the game is exited out of, then stop running the game
            if event.type == pygame.QUIT:
                gameIsRunning = False


    # Closes all the frames
    cv2.destroyAllWindows()

    # put code here that should be run every frame
    # of your game             
    pygame.display.update()




if __name__ == "__main__":
    main()



