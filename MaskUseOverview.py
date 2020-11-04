#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('./covid-19-data/mask-use/mask-use-by-county.csv')
df.index = df['COUNTYFP']
df.head()


# In[3]:


df_geocodes = pd.read_excel('./uscensus/all-geocodes-v2018.xlsx', skiprows=4)


# In[4]:


fipscode_u = df_geocodes[['State Code (FIPS)','County Code (FIPS)']].to_numpy()
fipscodes = np.zeros(fipscode_u.shape[0], dtype=np.int)
for idx in range(fipscode_u.shape[0]):
    fipscodes[idx] = int("%d%03d" % (fipscode_u[idx,0],fipscode_u[idx,1]))
    
df_geocodes.index = fipscodes
    


# In[5]:


df = df.join(df_geocodes, how='left')


# In[6]:


la_df = df[df['State Code (FIPS)'] == 22]
la_df['parish'] = la_df['Area Name (including legal/statistical area description)'].str.replace('Parish','')
la_df.tail()


# In[22]:


fig,axs = plt.subplots(3,figsize=(10,12))
pergraph = int(la_df.shape[0] / 3) + 1

metrics = ['NEVER','RARELY', 'SOMETIMES','FREQUENTLY','ALWAYS']

for idx,ax in zip(range(0,pergraph * len(axs),pergraph),axs):
    plots = []
    lasttop = np.zeros( la_df.shape[0], dtype=np.float )
    for metric, color in zip(metrics, ['r','orange','yellow','lime','darkgreen']):
        
        plots.append (ax.bar(la_df.iloc[idx:idx+pergraph]['parish'],            la_df.iloc[idx:idx+pergraph][metric] ,bottom=lasttop[idx:idx+pergraph],color=color))
        
        lasttop += la_df[metric]
    
    ax.tick_params(axis='x',rotation=90)
    fig.legend(reversed(plots),reversed(metrics))
    
    ax.set_yticks(np.arange(0,1.2,0.2))
    ax.set_yticklabels(["%d%%" % x for x in range(0,101,20)])
    ax.grid(axis='y')
    
fig.suptitle('"How Often Do You Wear a Mask in Public?" by Parish  (July)',fontsize='xx-large')
fig.tight_layout()
fig.savefig("fig10.jpg")


# In[ ]:




