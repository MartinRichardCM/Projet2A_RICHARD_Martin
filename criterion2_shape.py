#####################
#fonctionne
#####################
from mélanôme import *


def shape_assessment(A,contours,n):
# A is the inversion of the main shape's mask (the shape in white over a black background) which is an image # (2D array) contours is the list of contours of the original image and n is the location of our shape's
# contour in that list --> This function returns an evaluation of the shape of the object of interest.

    ellipse = cv.fitEllipse(contours[n])
    mask2 = np.zeros(A.shape,np.uint8)
    img = shape_of_grey(cv.ellipse(mask2, ellipse, (0, 255, 0), 1))
    matplotlib.pyplot.imsave('{jojo#5}.png',img,cmap='gray')
    shape3 = cv.imread("/Users/richardmartin/Documents/melanome/{jojo#5}.png")
    gray_img = cv.cvtColor(shape3, cv.COLOR_BGR2GRAY)
    ret1, thresh1 = cv.threshold(gray_img, 127, 255, 0)
    contours1, hierarchy1 = cv.findContours(gray_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    m = main_shape(contours1)
    cv.drawContours(img, contours1, m, (0,255,0), 1)
    mask3 = mask_to_be_improved(A,m,contours1)
    B = invertion(shape_of_grey(mask3))
    D,d = summation_bis(shape_of_grey(A)/255,B)
    return d


################################################################################
#Sinbad the sailor
################################################################################