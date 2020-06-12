#!/usr/bin/env python
# coding: utf-8

# ## Size of the Enron Dataset

# In[3]:


import sys
sys.path.append("/home/cit5/Downloads/ud120-projects-master/tools/")
sys.path.append('/home/cit5/Downloads/ud120-projects-master/choose_your_own')
sys.path.append('/home/cit5/Downloads/ud120-projects-master/datasets_questions')

import os
os.chdir('/home/cit5/Downloads/ud120-projects-master/datasets_questions')

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
print('Number of people in the Enron dataset: {0}'.format(len(enron_data)))


# ## Features in the Enron Dataset

# In[15]:


print ('Number of features for each person in the Enron dataset: {0}'.format(len(list(enron_data.values())[0])))


# ## Finding POI's in the Enron Data

# In[16]:


pois = [x for x, y in enron_data.items() if y['poi']]
print ('Number of POI\'s: {0}'.format(len(pois)))
#enron_data.items()[0]


# ## Query the Dataset 1

# In[17]:


# DELETE ME
enron_data['PRENTICE JAMES']


# In[18]:


enron_data['PRENTICE JAMES']['total_stock_value']


# ## Query the Dataset 2

# In[19]:


enron_data['COLWELL WESLEY']['from_this_person_to_poi']


# ## Query the Dataset 3

# In[20]:


enron_data['SKILLING JEFFREY K']['exercised_stock_options']


# ## Follow the Money

# In[21]:


names = ['SKILLING JEFFREY K', 'FASTOW ANDREW S', 'LAY KENNETH L']
names_payments = {name:enron_data[name]['total_payments'] for name in names}
print (sorted(names_payments.items(), key=lambda x: x[1], reverse=True))


# ## Dealing with Unfilled Features

# In[22]:


import pandas as pd

df = pd.DataFrame(enron_data)
print ('Has salary data: {0}'.format(sum(df.loc['salary',:] != 'NaN')))
print ('Has email: {0}'.format(sum(df.loc['email_address',:] != 'NaN')))


# ## Missing POI's 1

# In[23]:


# How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments? 
# What percentage of people in the dataset as a whole is this?

isnan = sum(df.loc['total_payments',:]=='NaN')
_,cols = df.shape
print ('total_payments == \'NaN\': {0} people = {1:.2f}%'.format(isnan, 100.*isnan/cols))


# ## Missing POI's 2

# In[24]:


isnan = sum(df.loc['total_payments',pois]=='NaN')
print ('POI total_payments == \'NaN\': {0} people = {1:.2f}%'.format(isnan, 100.*isnan/len(pois)))

