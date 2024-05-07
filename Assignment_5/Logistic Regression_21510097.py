#!/usr/bin/env python
# coding: utf-8

# In[36]:


pip install nose


# In[37]:


#Perform logistic regression on diabetes dataset


# In[38]:


#Load diabetes dataset here and return shape of dataset
from cz_utility import dataset
# diabetes dataset
df = dataset.load('63ea33b29ac5f900139b8d07')
def returnShape(df):
    return df.shape
    raise NotImplementedError


# In[39]:


from nose.tools import assert_equal
assert_equal(returnShape(df), (768, 9))


# In[40]:


#separate input and output from dataset and remove unwanted data columns
#take x for input and y for output
def sepAndRemove(df):
    inp = df.drop('Outcome', axis = 1)
    out = df['Outcome']
    inp.drop('SkinThickness', axis=1, inplace = True)

    return [inp, out]
    raise NotImplementedError


# In[41]:


trainAndTestData = sepAndRemove(df)
assert_equal(trainAndTestData[0].shape, (768, 7))
assert_equal(trainAndTestData[1].shape, (768,))


# In[42]:


#separate training and testing dataset with test_size = 0.3
from sklearn.model_selection import train_test_split
def trainTestSplit(trainAndTestData):
    X_train, X_test, y_train, y_test = train_test_split(trainAndTestData[0], trainAndTestData[1], test_size=0.3, random_state=42)
    return [X_train, X_test, y_train, y_test]
    raise NotImplementedError


# In[44]:


trainAndTestData=trainTestSplit(sepAndRemove(df))
assert_equal(trainAndTestData[0].shape,(537, 7))
assert_equal(trainAndTestData[1].shape,(231, 7))
assert_equal(trainAndTestData[2].shape,(537,))
assert_equal(trainAndTestData[3].shape,(231,))


# In[46]:


#use MinMaxScaler to scale the data to a fixed range
from sklearn.preprocessing import MinMaxScaler
def scalar(trainAndTestData):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(trainAndTestData[0])
    X_test_scaled = scaler.transform(trainAndTestData[1])
    return [X_train_scaled, X_test_scaled]
    raise NotImplementedError


# In[47]:


trainAndTestData=scalar(trainTestSplit(sepAndRemove(df)))
assert_equal(trainAndTestData[0].shape, (537, 7))
assert_equal(trainAndTestData[1].shape, (231, 7))


# In[48]:


#perform logistic regression here and return accuracy between predected output for testx and testy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def logisticReg(trainAndTestData, trainy, testy):
    model = LogisticRegression()
    model.fit(trainAndTestData[0], trainy)
    predicted_output = model.predict(trainAndTestData[1])
    accuracy = accuracy_score(testy, predicted_output)
    return accuracy
    raise NotImplementedError


# In[49]:


#there might be difference between student answer and faculty answer
trainAndTestData1 = trainTestSplit(sepAndRemove(df))
accuracyScore = logisticReg(scalar(trainTestSplit(sepAndRemove(df))), trainAndTestData1[2], trainAndTestData1[3])
assert_equal(accuracyScore>0.60, True)


# In[51]:


df
