#!/usr/bin/env python

import cv2
import os
import pygame

def addImage(imagePath, structure, ingredient):
    # If it is time for another validation point...
    if numTraining[structure][ingredient] / 4 > numValidation[structure][ingredient]:
        destination = 'sandwiches/validation/{}/{:04}.jpg'.format(typeNames[structure][ingredient], numValidation[structure][ingredient])
        numValidation[structure][ingredient] += 1
    else:
        destination = 'sandwiches/training/{}/{:04}.jpg'.format(typeNames[structure][ingredient], numTraining[structure][ingredient])
        numTraining[structure][ingredient] += 1

    print(imagePath)
    print(destination)

    img = cv2.imread(imagePath)

    print(img)
    cv2.imshow("picture",img)
    img = cv2.resize(img, (200, 200))
    cv2.imwrite(destination, img)





screenWidth = 800
screenHeight = 600

os.environ['SDL_VIDEO_CENTERED'] = '1'

pygame.init()
imageDisplay = pygame.display.set_mode((screenWidth,screenHeight))
clock = pygame.time.Clock()

pictures = os.listdir('downloads/ - thumbnail/')
currentPicture = 'downloads/ - thumbnail/' + pictures.pop(0)
imageTexture = pygame.image.load(currentPicture)
imageDisplay.blit(imageTexture, ((screenWidth - imageTexture.get_width()) / 2, (screenHeight - imageTexture.get_height()) / 2))
pygame.display.update()

done = False

# Array for how many pictures have been sorted per category
numTraining = [ [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]  ]

numValidation =[[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]  ]

typeNames = [   ['lawful_good', 'neutral_good', 'chaotic_good'],
                ['lawful_neutral', 'true_neutral', 'chaotic_neutral'],
                ['lawful_evil', 'neutral_evil', 'chaotic_evil'] ]
while not done:

    newImage = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        
        elif event.type == pygame.KEYDOWN:

            # Backspace to indicate that the image is not a sandwich
            if event.key == pygame.K_BACKSPACE:
                print("Not a sandwich")
                newImage = True

            elif event.key == pygame.K_ESCAPE:
                done = True

            # Detect sandwich types

            # lawful good is Q
            elif event.key == pygame.K_q:
                addImage(currentPicture, 0, 0)
                newImage = True

            # lawful neutral as A
            elif event.key == pygame.K_a:
                addImage(currentPicture, 1, 0)
                newImage = True

            # lawful evil is Z
            elif event.key == pygame.K_z:
                addImage(currentPicture, 2, 0)
                newImage = True

            # neutral good is W
            elif event.key == pygame.K_w:
                addImage(currentPicture, 0, 1)
                newImage = True

            # true neutral is S
            elif event.key == pygame.K_s:
                addImage(currentPicture, 1, 1)
                newImage = True

            # neutral evil is X
            elif event.key == pygame.K_x:
                addImage(currentPicture, 2, 1)
                newImage = True

            # chaotic good is E
            elif event.key == pygame.K_e:
                addImage(currentPicture, 0, 2)
                newImage = True

            # chaotic neatral is D
            elif event.key == pygame.K_d:
                addImage(currentPicture, 1, 2)
                newImage = True

            # chaotic evil is C
            elif event.key == pygame.K_c:
                addImage(currentPicture, 2, 2)
                newImage = True


    if newImage:
        imageDisplay.fill((0, 0, 0))
        currentPicture = 'downloads/ - thumbnail/' + pictures.pop(0)
        imageTexture = pygame.image.load(currentPicture)
        imageDisplay.blit(imageTexture, ((screenWidth - imageTexture.get_width()) / 2, (screenHeight - imageTexture.get_height()) / 2))
        pygame.display.update()
    clock.tick(20)

pygame.quit()
quit()