#####################
#fonctionne
#####################
from mélanôme import *

def regularity (L,n):
# L is there the list of the points forming the outline we're interested in and it is in a strange format
# and n is the shape of interest--> This function returns how many times does the local derivative's symbol
# changes.
    M=[]
    # print(L[n])
    # print(L[n][50][0][0])
    # print()
    for k in range (len(L[n])-1):
        #print(k)
        #print(L[n][k+1][0][0])
        if (L[n][k+1][0][0] - L[n][k][0][0]) != 0:
            M.append((L[n][k+1][0][1]-L[n][k][0][1])/(L[n][k+1][0][0]-L[n][k][0][0]))
        else:
            pass
    u,v = 0,1
    for k in range (len(M)-1):
        #print(M[k+1]*M[k])

        if v*M[k]<0:
            u = u+1
            v = M[k]
        else:
            pass
    return u





################################################################################
#Sinbad the sailor
################################################################################