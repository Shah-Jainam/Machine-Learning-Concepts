#!/usr/bin/env python
# coding: utf-8

# ## Support Vector Machine in SKlearn

# In[48]:


import sys
sys.path.append("/home/cit5/Downloads/ud120-projects-master/tools/")
sys.path.append('/home/cit5/Downloads/ud120-projects-master/choose_your_own')
sys.path.append('/home/cit5/Downloads/ud120-projects-master/svm')

import os
os.chdir('/home/cit5/Downloads/ud120-projects-master/svm')


from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

from sklearn.metrics import accuracy_score


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## Support Vector Machine #################################
from sklearn.svm import SVC

# Accuracy Function
def submitAccuracy():
    return accuracy_score(pred, labels_test)

# Create classifier
clf = SVC(kernel="linear")

# Fit the classifier on the training features and labels
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print(accuracy_score(pred, labels_test))


# ## Kernel and Gamma

# In[49]:


clf = SVC(kernel="linear", gamma=1.0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

get_ipython().run_line_magic('matplotlib', 'inline')
prettyPicture(clf, features_test, labels_test)


# ## SVM `C` Parameter

# In[51]:


clf = SVC(kernel="rbf", C=10**5, gamma="auto")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

get_ipython().run_line_magic('matplotlib', 'inline')
prettyPicture(clf, features_test, labels_test)


# ## SVM `gamma` Parameter

# In[42]:


clf = SVC(kernel="rbf", gamma=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

get_ipython().run_line_magic('matplotlib', 'inline')
prettyPicture(clf, features_test, labels_test)


# ## SVM Author ID Accuracy & Timing

# In[52]:


""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
from sklearn.metrics import accuracy_score

from time import time
from email_preprocess_data import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


def my_svm(features_train, features_test, labels_train, labels_test, kernel='linear', C=1.0):
    # the classifier
    clf = SVC(kernel=kernel, C=C, gamma="auto")

    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print("\ntraining time:", round(time()-t0, 3), "s")

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print("predicting time:", round(time()-t0, 3), "s")
    
    # accuracy
    accuracy = accuracy_score(pred, labels_test)

    print('\naccuracy = {0}'.format(accuracy))
    return pred

pred = my_svm(features_train, features_test, labels_train, labels_test)


# ## A Smaller Training Set

# In[53]:


features_train2 = features_train[:int(len(features_train)/100)]
labels_train2 = labels_train[:int(len(labels_train)/100)]

pred = my_svm(features_train2, features_test, labels_train2, labels_test)


# ## Deploy an RBF Kernel

# In[54]:


pred = my_svm(features_train2, features_test, labels_train2, labels_test, 'rbf')


# ## Optimize `C` Parameter

# In[55]:


for C in [10, 100, 1000, 10000]:
    print('C =',C)
    pred = my_svm(features_train2, features_test, labels_train2, labels_test, kernel='rbf', C=C)
    print('\n\n')


# ## Optimized RBF vs. Linear SVM: Accuracy

# In[56]:


pred = my_svm(features_train, features_test, labels_train, labels_test, kernel='rbf', C=10000)


# ## Extracting Predictions from an SVM

# In[57]:


print(pred[10])
print(pred[26])
print(pred[50])


# ## How many Chris emails predicted?

# In[58]:


print(sum(pred))

