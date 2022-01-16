#####################
#fonctionne
#####################
from mélanôme import *

def color (A,B,n,contours):
# A is an image (2D array in binary), B is an image aswell (but in shape of grey), contours is the list of
# contours of the original image and n is the location of our shape's contour in that list
# --> This function returns an evaluation of color's changes.
    mask_end = mask_to_be_improved(A,n,contours)
    D = multiply (shape_of_grey(B),invertion(mask_end))
    return stand_dev(D)


################################################################################
#Sinbad the sailor
################################################################################