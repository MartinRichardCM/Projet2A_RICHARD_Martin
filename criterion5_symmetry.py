#####################
#fonctionne
#####################
from mélanôme import *

def symmetry(A,n):
# A is a list containing the coutours our our shapes (strange format) and n is the coordonate
# of the outline we're interested in --> This function returns the angle formed between the
# x axis and the shape's principal axis.
    a,b,c,d,e,f,g,h=3000,0,0,0,3000,0,0,0
    # The maxima of x' values.
    for i in range (len(A[n])):
        if A[n][i][0][0] < a:
            a = A[n][i][0][0]
            c = i
        elif A[n][i][0][0] > b:
            b = A[n][i][0][0]
            d = i
        elif A[n][i][0][1] < e:
            e = A[n][i][0][1]
            g = i
        elif A[n][i][0][1] > f:
            f = A[n][i][0][1]
            h = i
    w = []
    if abs(A[n][c][0][0]-A[n][d][0][0]) < abs(A[n][g][0][1]-A[n][h][0][1]):
        w.append(g)
        w.append(h)
    else:
        w.append(c)
        w.append(d)
    x_1 = A[n][w[0]][0][0]
    x_2 = A[n][w[1]][0][0]
    y_1 = A[n][w[0]][0][1]
    y_2 = A[n][w[1]][0][1]
    a = (y_1 - y_2)/(x_2 - x_1)
    b = y_2 - (y_1 - y_2)/(x_2 - x_1)*x_1
    return (((np.pi)/2 - np.arctan((abs(a**2 + 2*a*b))**(1/2))) * (180/np.pi),a,b)

def symmetry_assessment(A,B,n):
# A is an image (2D array), B is the list of coutours for this image (strange format)
# and n is the coordinate of the contour of interest in that list. --> This algorithm
# returns an evaluation of the shape's symmetry
    w,x,y = symmetry(B,n)
    z = x * len(A)/2 + y
    alpha = len(A) - z
    C = Image.fromarray(A)
    D = C.rotate(-w)
    E = shape_of_grey(np.asarray(D))
    M = np.float32([
        [1, 0, 0],
        [0, 1, int(alpha)]])
    F = cv.warpAffine(E, M, (len(A[0]), len(A)))
    G = cv.flip(F, 0)
    N = np.float32([
        [1, 0, 0],
        [0, 1, z]])
    H = cv.warpAffine(G, N, (len(F[0]), len(F)))
    I,J = H/255,E/255
    K,a = summation_bis(I,J)
    return a




################################################################################
#Sinbad the sailor
################################################################################