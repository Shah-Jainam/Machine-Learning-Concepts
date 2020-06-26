#!/usr/bin/env python
# coding: utf-8

# ## Min/Max Scaler in `sklearn`

# In[1]:


import sys
sys.path.append("/home/cit5/Downloads/ud120-projects-master/tools/")
sys.path.append('/home/cit5/Downloads/ud120-projects-master/choose_your_own')
sys.path.append('/home/cit5/Downloads/ud120-projects-master/datasets_questions')

import os
os.chdir('/home/cit5/Downloads/ud120-projects-master/outliers')


import numpy as np
from sklearn.preprocessing import MinMaxScaler

weights = np.array([[115.], [140.], [175.]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)


# ## Computing Rescaled Features

# In[3]:


import pickle
import numpy
#import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

# the features to be used
features_list = ['poi', 'salary', 'exercised_stock_options']

data = featureFormat(data_dict, features_list)

_, salary, stock = zip(*data)

# put the features into 2-D numpy arrays
salary = np.array(salary).reshape((len(salary),1))
stock = np.array(stock).reshape((len(stock),1))

# rescale
scaler = MinMaxScaler()
salary = scaler.fit_transform(salary)
print ('$200,000 becomes {0}'.format(scaler.transform([[200000.]])[0][0]))

stock = scaler.fit_transform(stock)
print ('$1,000,000 becomes {0}'.format(scaler.transform([[1000000.]])[0][0]))

