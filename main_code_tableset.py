#####################
#fonctionne
#####################
from mélanôme import *
from criterion1_color import *
from criterion2_shape import *
from criterion3_regularity import *
from criterion4_curves import *
from criterion5_symmetry import *
import csv
import os
import pandas as pd
from PIL import Image
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import time
import copy



def table(A):
# A is the list of image-list.
    L,M = [],[]
    w = 0
    for i in range (len(A)):
        time_start_loop = time.time()
        # Our code was designed to process .png images.
        im1 = imread('/Users/richardmartin/Documents/melanome/ISIC-images/BCN_2020_Challenge/'+str(A[i][1])+'.jpg')
        im2 = copy.deepcopy(im1)
        image = shape_of_grey(im1/255)
        filt_real, filt_imag = gabor(image, frequency=0.6)
        filtered_img = gaussian(filt_real, sigma=15, multichannel=True)
        shape1 = shape_modif(image,filt_real)
        shape2 = invertion(shape1)
        matplotlib.pyplot.imsave('{jojo#1}.png',shape2,cmap='gray')
        # jojo#1 is a filtered version of our original image --> we got rid of hairs.
        img1 = cv.imread("/Users/richardmartin/Documents/melanome/{jojo#1}.png")
        gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        # noise reduction
        ret, thresh = cv.threshold(gray_img, 127, 255, 0)
        # Looking for contours
        contours, hierarchy = cv.findContours(gray_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        n=main_shape(contours)
        cv.drawContours(im1, contours, n, (0,255,0), 1)
        # The original image's mask.
        mask_end = mask_to_be_improved(img1,n,contours)
        matplotlib.pyplot.imsave('{jojo#2}.png',mask_end,cmap='gray')
        a = color (img1,image,n,contours)
        b = shape_assessment(img1,contours,n)
        c = regularity(contours,n)
        d = curves(contours,n)
        e = A[i][1]
        if A[i][15] == True:
            f = 1
        else:
            f = 0
        L.append([im1,im2,a,b,c,d,e,f])
        M.append([a,b,c,d,e,f])
        time_end_loop = time.time()
        number = len(A)
        delta = time_end_loop - time_start_loop
        w = w+delta
        average_time=(w)/(i+1)
        print('Image number {} ,this image took {} seconds to process. While an image  takes {} seconds to process on average, note that there are {} images in total.'.format(i+1, delta,average_time,number))
    return L,M

df = pd.read_csv("/Users/richardmartin/Documents/melanome/ISIC-images/metadata.csv")
df = df.replace(r'^\s*$', np.nan, regex=True)
dfbis = df.to_numpy()
a,b = table(dfbis)
#print(table(dfbis)[1])
#np.savetxt('test1.txt', Ea, fmt='%d')
print('hello world')


################################################################################
#Sinbad the sailor
################################################################################

