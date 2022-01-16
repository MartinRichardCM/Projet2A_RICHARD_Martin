from skimage.filters import gabor
from skimage.filters import gaussian
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib
from PIL import Image, ImageFilter, ImageDraw

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()



def shape_of_grey(im):
# im is an image (format matplotlib) --> the function returns a 2D arraythat has the same
# size as the original image and is the transformation of im into shape of grey.
    dim = np.shape(im)

    if len(dim) == 2:
        return(im)
    else :
        tab = 0.299 * im[:,:,0] + 0.587 * im[:,:,1] + 0.114 * im[:,:,2]
    return(tab)

def mean (A):
# The function return an integer which is the mean of the values of a 2D array.
    u,v = 0,0
    for i in range (len(A)):
        for j in range (len(A[0])):
            u = u+1
            v = v+A[i][j]
    return (v/u)

def stand_dev(A):
# The function return an integer whih is the tandard variation of the values of a 2D array.
    u,v,w = 0,0,mean(A)
    for i in range (len(A)):
        for j in range (len(A[0])):
            u = u+1
            v = v+(A[i][j]-w)**2
    return (v/u)**(0.5)

def max_list(A):
# A is a list of integer --> this function returns the maximum value of the list
    a=0
    for k in range (len(A)):
        if A[k]>a:
            a = A[k]
        else:
            pass
    return a

def shape(A,B):
# A is the original image (2D array) and B is A after being filtered (2D array)
# --> this funtion returns a 2D array (same size as A) which shows the shape of
# the object of interest in an image already filtered.
    C=np.ones((len(A),len(A[0])))
    a,b = mean(B),stand_dev(B)
    for i in range (len(A)):
        for j in range (len(A[0])):
            if B[i][j] < (a-b):
                C[i][j] = A[i][j]
            else:
                pass
    return C


def main_shape(contours):
    L,m,n=[],0,0
    for k in range (len(contours)):
        L.append([len(contours[k]),k])
    for i in range (len(L)):
        if L[i][0]>m:
            m,n=L[i][0],L[i][1]
        else:
            pass
    return n

def invertion(M):
    # M is an image inshape of gray (from png format) --> This function returns
    #the image's negative

    (a,b) = (len(M),len(M[0]))
    N = np.zeros((a,b))

    for i in range(a):
        for j in range(b):
            N[i][j] = 1-M[i][j]
            if N[i][j]<0.0001 :
                N[i][j] = 0
    return(N)

def invertion_bis (M):
    # M is a binary image (from jpeg format) --> This function returns
    #the image's negative
    (a,b) = (len(M),len(M[0]))
    N = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            N[i][j] = 255-M[i][j]
            if N[i][j] == 255:
                N[i][j] = 1
            else:
                pass
    return N



def shape_modif(A,B):
# A is the original image (2D array) and B is A after being filtered (2D array)
# --> this funtion returns a 2D array (same size as A) which shows the shape of
# the object of interest in an image already filtered.
    C=np.ones((len(A),len(A[0])))
    a,b = mean(B),stand_dev(B)
    for i in range (len(A)):
        for j in range (len(A[0])):
            if B[i][j] < (a-b):
                C[i][j] = 0
            else:
                pass
    return C

def mask_to_be_improved (img,n,contours):
    mask = np.zeros(img.shape,np.uint8)
    mask.fill(255)
    cv.drawContours(mask, contours, n, (0,0,0), 1)
    im = Image.fromarray(mask)
    width, height = im.size
    center = (int(0.5 * width), int(0.5 * height))
    black = (0, 0, 0, 255)
    ImageDraw.floodfill(im, xy=center, value=black)
    mask_end = shape_of_grey(np.array(im)/255)
    return mask_end

def multiply (A,B):
# Both A and B are 2D array of the same size --> This functionsthe 2D array AB
# (AB being a term by term multiplication).
    C=np.ones((len(A),len(A[0])))
    for i in range (len(A)):
        for j in range (len(A[0])):
            C[i][j] = A[i][j] * B[i][j]
    return C

def summation(A,B):
# Both A and B are 2D array of the same size --> This functionsthe 2D array A+B
# (AB being a term by term sum).
    C=np.ones((len(A),len(A[0])))
    b,c = 0,0
    for i in range (len(A)):
        for j in range (len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
            b = b +B[i][j]
            if C[i][j] != 0 and C[i][j] != 1:
                c = c+1

    return C,(c/b)

def summation_bis (A,B):
# Both A and B are 2D array of the same size --> This functionsthe 2D array A+B
# (AB being a term by term sum) and 1 integer.
    a,b,c = 0,0,0
    C=np.zeros((len(A),len(A[0])))
    for i in range (len(A)):
        for j in range (len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
            a = a+A[i][j]
            b = b+B[i][j]
            if C[i][j] != 0 and C[i][j] != 1:
                c = c+1
    return C,(c/(b+a))


def find_outline(A):
# A is an image --> This function returns a blacked out image with only the outline remaining.
    C=np.zeros((len(A),len(A[0])))
    for i in range (len(A)):
        for j in range (len(A[0])):
            if A[i][j] != 0 and A[i][j] != 1:
                C[i][j] = 1
            else:
                pass
    return C

###########################

# og_image = imread('/Users/richardmartin/Desktop/Screenshot 2021-11-17 at 11.21.01.png')
# image = shape_of_grey(imread('/Users/richardmartin/Desktop/Screenshot 2021-11-17 at 11.21.01.png'))
# filt_real, filt_imag = gabor(image, frequency=0.6)
# filtered_img = gaussian(filt_real, sigma=15, multichannel=True)
# # plt.imshow(filtered_img,cmap='gray')
# # plt.show()
# shape = shape(image,filtered_img)
# shape1 = shape_modif(image,filt_real)
# shape2 = invertion(shape1)
# # outlinebis = rid_im_outline(shape2)
# matplotlib.pyplot.imsave('{jojo#1}.png',shape2,cmap='gray')
# img = cv.imread("/Users/richardmartin/Documents/melanome/{jojo#1}.png")
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # noise reduction
# ret, thresh = cv.threshold(gray_img, 127, 255, 0)
# # Looking for contours
# contours, hierarchy = cv.findContours(gray_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# n=main_shape(contours)
# cv.drawContours(shape2, contours, n, (0,255,0), 1)
# # outline_end=cv.drawContours(img, contours,n, (0,255,0), 1)
# # mask = np.zeros(img.shape,np.uint8)
# # mask.fill(255)
# # mask_end = cv.fillPoly(mask,contours[n],(0,0,0),1)
# #mask_end = cv.fillConvexPoly(mask,contours[n],(0,0,0),1)
# mask_end = mask_to_be_improved(img,n,contours)
# matplotlib.pyplot.imsave('{jojo#2}.png',mask_end,cmap='gray')
# ###########################
# # show_image_list(list_images=[og_image,img],
# #                 list_titles=['original','the contour belonging to the shape of interest'],
# #                 num_cols=2,
# #                 figsize=(20, 10),
# #                 grid=False,
# #                 title_fontsize=20)
# ###########################







################################################################################
#Sinbad the sailor
################################################################################
