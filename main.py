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





#########
# main function
#########

def main():
    # make a hand detector
    handDetector = HandDetector()
    # keeps track if pygame should keep running
    gameIsRunning = True

    # while the opencv window is running
    while not handDetector.shouldClose and gameIsRunning:

        # update the webcam feed and hand tracker calculations
        handDetector.update()

        # if there is at least one hand seen, then
        # print out the landmark positions
        if len(handDetector.landmarkDictionary) > 0: 
            print(handDetector.landmarkDictionary[0])


        # for all the game events
        for event in pygame.event.get():
             # if the game is exited out of, then stop running the game
            if event.type == pygame.QUIT:
                gameIsRunning = False


    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



