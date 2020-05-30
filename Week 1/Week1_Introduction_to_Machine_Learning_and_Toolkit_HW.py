#!/usr/bin/env python
# coding: utf-8

# # Introduction to Machine Learning and Toolkit Exercises 

# ## Introduction
# 
# We will be using the iris data set for this tutorial. This is a well-known data set containing iris species and sepal and petal measurements. The data we will use are in a file called `Iris_Data.csv` found in the [data](../../data) directory.

# In[85]:


from __future__ import print_function
import os
data_path = ['Iris_Data']
print (data_path)


# ## Question 1
# 
# Load the data from the file using the techniques learned today. Examine it.
# 
# Determine the following:
# 
# * The number of data points (rows). (*Hint:* check out the dataframe `.shape` attribute.)
# * The column names. (*Hint:* check out the dataframe `.columns` attribute.)
# * The data types for each column. (*Hint:* check out the dataframe `.dtypes` attribute.)

# In[86]:


import numpy as np
import pandas as pd

filepath = os.sep.join(data_path + ['Iris_Data.csv'])
print(filepath)
data = pd.read_csv('Iris_Data.csv')
data.head()


# In[87]:


# Number of rows
print(data.shape[0])

# Column names
print(data.columns.tolist())

# Data types
print(data.dtypes)


# ## Question 2
# 
# Examine the species names and note that they all begin with 'Iris-'. Remove this portion of the name so the species name is shorter. 
# 
# *Hint:* there are multiple ways to do this, but you could use either the [string processing methods](http://pandas.pydata.org/pandas-docs/stable/text.html) or the [apply method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html).

# In[88]:


# The str method maps the following function to each entry as a string
#data['species'] = data.species.str.replace('Iris-', '')
# alternatively
data['species'] = data.species.apply(lambda r: r.replace('Iris-', ''))

data.head()


# ## Question 3
# 
# Determine the following:  
# * The number of each species present. (*Hint:* check out the series `.value_counts` method.)
# * The mean, median, and quantiles and ranges (max-min) for each petal and sepal measurement.
# 
# *Hint:* for the last question, the `.describe` method does have median, but it's not called median. It's the *50%* quantile. `.describe` does not have range though, and in order to get the range, you will need to create a new entry in the `.describe` table, which is `max - min`.

# In[89]:


#Student writes code here
idx = pd.value_counts(['sepal_length' , 'sepal_width' , 'petal_length' , 'petal_width' , 'species'])
print(idx)
data.mean(axis=0)


# In[ ]:





# ## Question 4
# 
# Calculate the following **for each species** in a separate dataframe:
# 
# * The mean of each measurement (sepal_length, sepal_width, petal_length, and petal_width).
# * The median of each of these measurements.
# 
# *Hint:* you may want to use Pandas [`groupby` method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) to group by species before calculating the statistic.
# 
# If you finish both of these, try calculating both statistics (mean and median) in a single table (i.e. with a single groupby call). See the section of the Pandas documentation on [applying multiple functions at once](http://pandas.pydata.org/pandas-docs/stable/groupby.html#applying-multiple-functions-at-once) for a hint.

# In[90]:


# The mean calculation
data.groupby('species').mean()
sepal_length = data['sepal_length'].mean()
print("sepal_length", sepal_length)
sepal_width = data['sepal_width'].mean()
print("sepal_width", sepal_width)
petal_length = data['petal_length'].mean()
print("petal_length", petal_length)
petal_width = data['petal_width'].mean()
print("petal_width", petal_width)


# In[91]:


# The median calculation
data.groupby('species').median()
sepal_length = data['sepal_length'].median()
print("sepal_length", sepal_length)
sepal_width = data['sepal_width'].median()
print("sepal_width", sepal_width)
petal_length = data['petal_length'].median()
print("petal_length", petal_length)
petal_width = data['petal_width'].median()
print("petal_width", petal_width)


# In[92]:


# applying multiple functions at once - 2 methods

#data.groupby('species').agg(['mean', 'median'])  # passing a list of recognized strings
data.groupby('species').agg([np.mean, np.median])  # passing a list of explicit aggregation functions


# In[93]:


# If certain fields need to be aggregated differently, we can do:
from pprint import pprint

agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'species'}
agg_dict['petal_length'] = 'max'
pprint(agg_dict)
data.groupby('species').agg(agg_dict)


# ## Question 5
# 
# Make a scatter plot of `sepal_length` vs `sepal_width` using Matplotlib. Label the axes and give the plot a title.

# In[94]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[95]:


# A simple scatter plot with Matplotlib
ax = plt.axes()

ax.scatter(data.sepal_length, data.sepal_width)

# Label the axes
ax.set(xlabel='Sepal Length (cm)',
       ylabel='Sepal Width (cm)',
       title='Sepal Length vs Width');


# ## Question 6
# 
# Make a histogram of any one of the four features. Label axes and title it as appropriate. 

# In[149]:


#Student writes code here


# In[175]:



plt.style.use('ggplot')
plt.hist(data.sepal_length, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Species   sepal_length     \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[176]:


plt.hist(data.sepal_width, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Species sepal_width \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[177]:


plt.hist(data.petal_width, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Species petal_width \n\n', 
          fontweight ="bold") 
  
plt.show() 


# In[178]:


plt.hist(data.petal_length, bins=25)
plt.xlabel('Size')
plt.ylabel('Frequency')
  
plt.title('Species petal_length \n\n', 
          fontweight ="bold") 
  
plt.show() 


# ## Question 7
# 
# Now create a single plot with histograms for each feature (`petal_width`, `petal_length`, `sepal_width`, `sepal_length`) overlayed. If you have time, next try to create four individual histogram plots in a single figure, where each plot contains one feature.
# 
# For some hints on how to do this with Pandas plotting methods, check out the [visualization guide](http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html) for Pandas.

# In[179]:


import seaborn as sns

sns.set_context('notebook')

# This uses the `.plot.hist` method
ax = data.plot.hist(bins=25, alpha=0.5)
ax.set_xlabel('Size (cm)');


# In[182]:


# To create four separate plots, use Pandas `.hist` method
axList = data.hist(bins=20)

# Add some x- and y- labels to first column and last row
for ax in axList.flatten():
    if ax.is_last_row():
        ax.set_xlabel('Size (cm)')
        
    if ax.is_first_col():
        ax.set_ylabel('Frequency')


# ## Question 8
# 
# Using Pandas, make a boxplot of each petal and sepal measurement. Here is the documentation for [Pandas boxplot method](http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#visualization-box).

# In[212]:


(data
.groupby('species')
.mean()
.plot(color=['red' ,
             'blue' ,
             'black' ,
             'green'],fontsize=10.0, figsize=(4,4)))


# ## Question 9
# 
# Now make a single boxplot where the features are separated in the x-axis and species are colored with different hues. 
# 
# *Hint:* you may want to check the documentation for [Seaborn boxplots](http://seaborn.pydata.org/generated/seaborn.boxplot.html). 
# 
# Also note that Seaborn is very picky about data format--for this plot to work, the input dataframe will need to be manipulated so that each row contains a single data point (a species, a measurement type, and the measurement value). Check out Pandas [stack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.stack.html) method as a starting place.
# 
# Here is an example of a data format that will work:
# 
# |   | species | measurement  | size |
# | - | ------- | ------------ | ---- |
# | 0	| setosa  | sepal_length | 5.1  |
# | 1	| setosa  | sepal_width  | 3.5  |

# In[213]:


# First we have to reshape the data so there is 
# only a single measurement in each column

plot_data = (data
             .set_index('species')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'})
            )

plot_data.head()


# In[214]:


# Now plot the dataframe from above using Seaborn

sns.set_style('white')
sns.set_context('notebook')
sns.set_palette('dark')

f = plt.figure(figsize=(6,4))
sns.boxplot(x='measurement', y='size', 
            hue='species', data=plot_data);


# ## Question 10
# 
# Make a [pairplot](http://seaborn.pydata.org/generated/seaborn.pairplot.html) with Seaborn to examine the correlation between each of the measurements.
# 
# *Hint:* this plot may look complicated, but it is actually only a single line of code. This is the power of Seaborn and dataframe-aware plotting! See the lecture notes for reference.
