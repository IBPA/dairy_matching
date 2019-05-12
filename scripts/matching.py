import numpy as np
import pandas as pd
from scipy.stats import variation

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


def row_corr_weighted(A,B,weights):
# converted to python from here: https://stackoverflow.com/questions/9460664/weighted-pearsons-correlation
    w = weights/sum(weights)
    
    A = A - (A*w).sum(1)[:,None]
    B = B - (B*w).sum(1)[:,None]
    
    pcorr = np.matmul(A,((B*w).T))/np.sqrt(np.matmul(((A**2)*w).sum(1)[:,None],(((B**2)*w).sum(1)[:,None]).T))
    return pcorr


def match(similarity_matrix,asa_24,ncc,N):
    TOP = N # set the number of matches to return

    # Get indices of the top matches from the combined similarity matrix
    indices_top = np.argsort(-similarity_matrix,axis=1,)[:,:TOP]
    pcc_top = np.sort(-similarity_matrix,axis=1,)[:,:TOP]
    
    iters = [np.arange(0,asa_24.shape[0],1).tolist(),np.arange(0,TOP,1).tolist()]
    
    # Construct dataframe to store top results
    results_cols = asa_24.columns.values.tolist() + ['similarity'] + ncc.columns.values.tolist()

    mi = pd.MultiIndex.from_product(iters, names=['asa_index', 'match_index'])
    
    results_top = pd.DataFrame(index=mi,columns=results_cols)
    
    # Copy ASA24 values to left side
    results_top.loc[results_top.eval('match_index==0'), asa_24.columns] = asa_24.values

    results_top.loc[:,ncc.columns] = ncc.iloc[indices_top.flatten(),:].values

    results_top.loc[:,'similarity'] = -pcc_top.flatten()
    
    # calculate variation across matches
    variations = pd.DataFrame(results_top['Lactose (g)'].groupby("asa_index").apply(variation))
    
    for index in results_top.index.get_level_values(0).unique():
        results_top.loc[index,'variation'] = variations.loc[index,:].values[0]
        
    return results_top
