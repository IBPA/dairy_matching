import numpy as np

def row_corr(A,B):
    #number of columns in A or B
    N = B.shape[1]

    # Store row-wise sums of A and B, as they would be used at few places
    sA = A.sum(1)
    sB = B.sum(1)

    # Compute the four terms in pcc matrix-wise
    p1 = N*np.einsum('ik,jk->ij',A,B)
    p2 = sB*sA[:,None]
    p3 = N*((B**2).sum(1)) - (sB**2)
    p4 = N*((A**2).sum(1)) - (sA**2)

    # compute pcc as 2D array 
    pcorr = ((p1 - p2)/np.sqrt(p3*p4[:,None]))
    return pcorr
