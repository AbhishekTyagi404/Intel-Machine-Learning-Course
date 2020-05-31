#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
data_path = ['Wholesale_Customers_Data']
print (data_path)


# In[2]:


import numpy as np
import pandas as pd

filepath = os.sep.join(data_path + ['Wholesale_Customers_Data.csv'])
print(filepath)
data = pd.read_csv('Wholesale_Customers_Data.csv')
data.head()


# In[3]:


# Number of rows
print(data.shape[0])

# Column names
print(data.columns.tolist())

# Data types
print(data.dtypes)


# In[5]:


#data_type
idx = pd.value_counts(['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
print(idx)
data.mean(axis=0)


# In[6]:


# The mean calculation
data.groupby('Channel').mean()
Region = data['Region'].mean()
Fresh = data['Fresh'].mean()
Milk = data['Milk'].mean()
Grocery = data['Grocery'].mean()
Frozen = data['Frozen'].mean()
Detergents_Paper = data['Detergents_Paper'].mean()
Delicassen = data['Delicassen'].mean()
#'Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'

print("Region", Region ,'\n', "Fresh", Fresh ,'\n',"Milk", Milk,'\n',"Grocery", Grocery,'\n',"Frozen", Frozen,'\n',"Detergents_Paper", Detergents_Paper,'\n',"Delicassen", Delicassen)


# In[7]:


# The median calculation
# The mean calculation
data.groupby('Channel').median()
Region = data['Region'].median()
Fresh = data['Fresh'].median()
Milk = data['Milk'].median()
Grocery = data['Grocery'].median()
Frozen = data['Frozen'].median()
Detergents_Paper = data['Detergents_Paper'].median()
Delicassen = data['Delicassen'].median()
#'Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'

print("Region", Region ,'\n', "Fresh", Fresh ,'\n',"Milk", Milk,'\n',"Grocery", Grocery,'\n',"Frozen", Frozen,'\n',"Detergents_Paper", Detergents_Paper,'\n',"Delicassen", Delicassen)


# In[8]:


# applying multiple functions at once - 2 methods

#data.groupby('Channel').agg(['mean', 'median'])  # passing a list of recognized strings
data.groupby('Channel').agg([np.mean, np.median])  # passing a list of explicit aggregation functions


# In[9]:


# If certain fields need to be aggregated differently, we can do:
from pprint import pprint

agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'Channel'}
#agg_dict['Region'] = 'max'
pprint(agg_dict)
data.groupby('Channel').agg(agg_dict)


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# A simple scatter plot with Matplotlib
ax = plt.axes()

ax.scatter(data.Milk , data.Frozen)

# Label the axes
ax.set(xlabel='Milk',
       ylabel='Frozen',
       title='Milk vs Frozen');


# In[11]:


plt.style.use('ggplot')
plt.hist(data.Milk, bins=25)
plt.xlabel('Quantity')
plt.ylabel('Frequency')
  
plt.title('Channel   Milk     \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[12]:


plt.hist(data.Frozen, bins=25)
plt.xlabel('Quantity')
plt.ylabel('Frequency')
  
plt.title('Channel Frozen \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[13]:


#'Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'
plt.hist(data.Fresh, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Channel Fresh \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[14]:


plt.hist(data.Grocery, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Channnel Grocery \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[15]:


plt.hist(data.Detergents_Paper, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Detergents_Paper \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[16]:


plt.hist(data.Delicassen, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Delicassen \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[18]:


import seaborn as sns

sns.set_context('notebook')

# This uses the `.plot.hist` method
ax = data.plot.hist(bins=25, alpha=0.5)
ax.set_xlabel('Quantity');


# In[19]:


# To create four separate plots, use Pandas `.hist` method
axList = data.hist(bins=25)

# Add some x- and y- labels to first column and last row
for ax in axList.flatten():
    if ax.is_last_row():
        ax.set_xlabel('Quantity)')
        
    if ax.is_first_col():
        ax.set_ylabel('Frequency')


# In[20]:


(data
.groupby('Channel')
.mean()
.plot(color=['red' ,
             'blue' ,
             'black' ,
             'orange',
             'green'],fontsize=10.0, figsize=(4,4)))


# In[21]:


# First we have to reshape the data so there is 
# only a single measurement in each column

plot_data = (data
             .set_index('Channel')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'})
            )

plot_data.head()


# In[22]:


# Now plot the dataframe from above using Seaborn

sns.set_style('white')
sns.set_context('notebook')
sns.set_palette('dark')

f = plt.figure(figsize=(6,4))
sns.boxplot(x='measurement', y='size', 
            hue='Channel', data=plot_data);

