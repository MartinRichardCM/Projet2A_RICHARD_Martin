#####################
#fonctionne
#####################
from mélanôme import *

def curves (L,n):
# L is there the list of the points forming the outline we're interested in and it is in a strange format
# and n is the shape of interest--> This function returns the standard deviation of the curve' values.
    M=[]

    for k in range (len(L[n])-2):
        if (L[n][k+1][0][0] - L[n][k][0][0]) != 0:
            h = L[n][k+1][0][0] - L[n][k][0][0]
            M.append((L[n][k+2][0][1]+L[n][k][0][1]-L[n][k+1][0][1])/h)
        else:
            M.append(0)
    u,v = 0,0
    for i in range (len(M)):
        u = u+1
        v = v+M[i]
    w = (v/u)
    x = 0
    for i in range (len(M)):
        x = x+(M[i]-w)**2
    return (x/u)**(0.5)





################################################################################
#Sinbad the sailor
################################################################################
