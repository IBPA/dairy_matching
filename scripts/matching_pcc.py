#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from scipy.stats import variation
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.metrics import r2_score, mean_absolute_error
rcParams['figure.figsize'] = 10,10


# ### Load the NCC Data

# In[27]:


ncc = pd.read_csv('../data/NCC_2018_nutrients_per_100g_originalcolnames.txt',sep='\t')


# Set Food ID as the index

# In[3]:


# ncc['Food ID'].is_unique


# In[4]:


# ncc = ncc.set_index('Food ID')


# In[30]:


ncc = ncc.rename(columns={'Food ID':'NCC Food ID'})


# ### Load the data from the ASA24 Recalls

# In[57]:


asa_24 = pd.read_csv('../data/training_for_GS_122118.csv')


# Set FoodCode as the index 

# In[6]:


# asa_24 = asa_24.set_index('FoodCode')


# ### Load the file listing matching columns between the datasets

# In[7]:


matching = pd.read_csv('../data/matching_ncc_fndds_columns.txt',sep='\t')


# Get the list of columns for each dataset

# In[8]:


ncc_cols = matching['NCC.Term'].values.tolist()
asa_24_cols = matching['FNDDS.Term'].values.tolist()
asa_24_cols = [val.replace(" ","") for val in asa_24_cols]


# ### Calculate the pairwise correlations
# Define a function to calculate the pairwise PCC matrix between two matrices A and B

# In[9]:


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


# Get the columns provided by the `matching` file

# In[10]:


A = asa_24.loc[:,asa_24_cols].values
B = ncc.loc[:,ncc_cols].values


# In[11]:


corr = row_corr(A,B)


# ### Calculate weighted pairwise correlations

# Load the weights from the Lasso model

# In[12]:


lasso_coef = pd.read_csv('lasso_coef.csv')
weights = lasso_coef.loc[:,'coef'].values[:-1] #omit 'year'


# Define a function to calculate weighted row-wise PCC

# In[13]:


def row_corr_weighted(A,B,weights):
# converted to python from here: https://stackoverflow.com/questions/9460664/weighted-pearsons-correlation
    w = weights/sum(weights)
    
    A = A - (A*w).sum(1)[:,None]
    B = B - (B*w).sum(1)[:,None]
    
    pcorr = np.matmul(A,((B*w).T))/np.sqrt(np.matmul(((A**2)*w).sum(1)[:,None],(((B**2)*w).sum(1)[:,None]).T))
    return pcorr


# In[14]:


corr_weighted = row_corr_weighted(A,B,weights)


# ### Get results (unweighted)

# Get indices of the top matches from the correlation matrix

# In[58]:


TOP = 5 # set the number of matches to return

indices_top = np.argsort(-corr,axis=1,)[:,:TOP]
pcc_top = np.sort(-corr,axis=1,)[:,:TOP]


# Construct dataframe to store top results

# In[59]:


iters = [np.arange(0,asa_24.shape[0],1).tolist(),np.arange(0,TOP,1).tolist()]

results_cols = asa_24.columns.values.tolist() + ['PCC'] + ncc.columns.values.tolist()

mi = pd.MultiIndex.from_product(iters, names=['asa_index', 'match_index'])

results_top = pd.DataFrame(index=mi,columns=results_cols)


# Copy ASA24 values to left side

# In[60]:


results_top.loc[results_top.eval('match_index==0'), asa_24.columns] = asa_24.values

results_top.loc[:,ncc.columns] = ncc.iloc[indices_top.flatten(),:].values

results_top.loc[:,'PCC'] = -pcc_top.flatten()


# In[61]:


variations = pd.DataFrame(results_top['Lactose (g)'].groupby("asa_index").apply(variation))


# In[62]:


for index in results_top.index.get_level_values(0).unique():
    results_top.loc[index,'variation'] = variations.loc[index,:].values[0]


# ### Save results to CSV

# In[63]:


name = 'pcc_matching_results_top_{}.tsv'.format(TOP)
path = '../data/' + name
results_top.to_csv(path,sep='\t')


# In[64]:


desc_only_cols = ['FoodCode','Food_Description','year','PCC','NCC Food ID','Keylist','Food Description','Short Food Description','Food Type','Lactose (g)','variation']


# In[65]:


results_top_desc_only = results_top[desc_only_cols]


# In[66]:


name = 'pcc_matching_results_top_{}_desc_only.tsv'.format(TOP)
path = '../data/' + name
results_top_desc_only.to_csv(path,sep='\t')


# ### Get results (weighted)

# Get indices of the top matches from the correlation matrix

# In[67]:


TOP = 5 # set the number of matches to return

indices_top = np.argsort(-corr_weighted,axis=1,)[:,:TOP]
pcc_top = np.sort(-corr_weighted,axis=1,)[:,:TOP]


# Construct dataframe to store top results

# In[68]:


iters = [np.arange(0,asa_24.shape[0],1).tolist(),np.arange(0,TOP,1).tolist()]

results_cols = asa_24.columns.values.tolist() + ['Weighted PCC'] + ncc.columns.values.tolist()

mi = pd.MultiIndex.from_product(iters, names=['asa_index', 'match_index'])

results_top = pd.DataFrame(index=mi,columns=results_cols)


# Copy ASA24 values to left side

# In[69]:


results_top.loc[results_top.eval('match_index==0'), asa_24.columns] = asa_24.values

results_top.loc[:,ncc.columns] = ncc.iloc[indices_top.flatten(),:].values

results_top.loc[:,'Weighted PCC'] = -pcc_top.flatten()


# In[70]:


variations = pd.DataFrame(results_top['Lactose (g)'].groupby("asa_index").apply(variation))


# In[71]:


for index in results_top.index.get_level_values(0).unique():
    results_top.loc[index,'variation'] = variations.loc[index,:].values[0]


# ### Save results to CSV

# In[72]:


name = 'pcc_matching_results_top_{}_weighted.tsv'.format(TOP)
path = '../data/' + name
results_top.to_csv(path,sep='\t')


# In[73]:


results_top_desc_only = results_top[['Weighted PCC' if col == 'PCC' else col for col in desc_only_cols]]


# In[74]:


name = 'pcc_matching_results_top_{}_weighted_desc_only.tsv'.format(TOP)
path = '../data/' + name
results_top_desc_only.to_csv(path,sep='\t')


# ### Load the data back in
# (Check that it was saved, further analysis, etc.)

# In[2]:


results_top_desc_only_w =  pd.read_csv('../data/pcc_matching_results_top_5_weighted_desc_only.tsv',sep='\t')


# In[76]:


results_top_w = pd.read_csv('../data/pcc_matching_results_top_5_weighted.tsv',sep='\t')
results_top = pd.read_csv('../data/pcc_matching_results_top_5.tsv',sep='\t')


# In[3]:


results_top_desc_only =  pd.read_csv('../data/pcc_matching_results_top_5_desc_only.tsv',sep='\t')


# Calculate MAE between dietitian labels and our lookup results

# In[153]:


labeled_lactose = results_top.loc[:,'lac.per.100g']
labeled_lactose = labeled_lactose[~np.isnan(labeled_lactose)].values.flatten()

lookup_lactose = pd.DataFrame(results_top['Lactose (g)'].groupby("asa_index").apply(np.mean)).values.flatten()
labeled_lactose_nonzero = labeled_lactose[labeled_lactose != 0]
lookup_lactose_nonzero = lookup_lactose[labeled_lactose != 0]


# In[145]:


def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[144]:


np.mean(lookup_lactose_nonzero)


# In[147]:


print('MAE: {}'.format(mean_absolute_error(labeled_lactose,lookup_lactose)))
print('MAPE: {}'.format(mean_absolute_percentage_error(labeled_lactose_nonzero,lookup_lactose_nonzero)))
print('R2: {}'.format(r2_score(labeled_lactose,lookup_lactose)))


# In[141]:


plt.scatter(x=labeled_lactose[labeled_lactose != 0],y=lookup_lactose[labeled_lactose != 0],s=3)
plt.xlabel('Dietitian-selected value')
plt.ylabel('Top 5 PCC Match Average value')
plt.title('Dietitian selected vs Top 5 PCC Match Average Values\n lactose g/100g')


# In[ ]:




