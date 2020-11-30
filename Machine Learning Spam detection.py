#!/usr/bin/env python
# coding: utf-8

# In[43]:


# All the Imported Module Here.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn import metrics


# In[3]:


# First i have One file that contain text and label (ham or spam). let's call it.
file = pd.read_csv(r'C:\Users\win 10\Downloads\Spam detection\sms.txt',header = None,names=['Label', 'Message'], sep='\t')
sms = pd.DataFrame(file)
sms.head()


# In[4]:


#let's count no of ham and spam
sms.Label.value_counts()


# In[5]:


X = sms.Message
Y = sms.Label


# In[6]:


# if you don't specify test_size default is 0.25
Xtrain,Xtest,Ytrain, Ytest = train_test_split(X,Y, test_size=0.30)


# In[7]:


#Let's Prepare Our Model
vect = CountVectorizer()

#let's Train data
vect.fit(Xtrain)# making Dictionary
Xtrain_matrix = vect.transform(Xtrain)


# In[27]:


print(
vect.fit(Xtrain)
)
print(vect.transform(Xtrain))


# In[8]:


# for testing Data Make dictionary and matrix
Xtest_matrix = vect.transform(Xtest)


# In[9]:


# make model, train Model
MNB = MultinomialNB()
MNB.fit(Xtrain_matrix, Ytrain)


# In[10]:


# now we have model,
# Let's Test Our Data
Ypredict = MNB.predict(Xtest_matrix)


# In[15]:


MNB.score(Xtest_matrix,Ytest)


# In[ ]:


# Print message for False Positive(Actually it,ham but prediction is spam)
Xtest[(Ypredict == 'spam') &(Ytest == 'ham')]


# In[24]:


# Print message for False Positive(Actually it,spam but prediction is ham)
Xtest[(Ypredict == 'ham') &(Ytest == 'spam')]


# In[16]:


ResultDict = {
    'Actual':Ytest,
    'Predicted':Ypredict
}
Result = pd.DataFrame(ResultDict)


# In[17]:


confusion_matrix(Ytest,Ypredict)


# In[18]:


plot_confusion_matrix(MNB, Xtest_matrix, Ytest)
plt.show()


# In[19]:


# Now, we try Different algoritham, LogisticRegression


# In[21]:


# import it
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(Xtrain_matrix, Ytrain)
LR.predict(Xtest_matrix)
LR.score(Xtest_matrix,Ytest)


# In[31]:


# now Fine tune our model and improve model accuracy
# remove stop words like the, have, has, was,a etc
# you see that matrix size is reduce
vect1 = CountVectorizer(stop_words='english')
Xtrain1 = vect1.fit_transform(Xtrain)
Xtrain1


# In[33]:


#now take another parameter
#ngram_range
# you see that matrix size is very large

vect2 = CountVectorizer(ngram_range=(1,2))
Xtrain2 = vect2.fit_transform(Xtrain)
Xtrain2
df = pd.DataFrame(Xtrain2.toarray(), columns=vect2.get_feature_names())
df


# In[36]:


#now take another parameter
#max_df, min_df
# remove all the word that appear in more than 50%
vect3 = CountVectorizer(max_df=0.50)
Xtrain3 = vect3.fit_transform(Xtrain)
print(Xtrain3.shape)

# onle keep those word in dataframe min 2 times
vect4 = CountVectorizer(min_df=2)
Xtrain4 = vect4.fit_transform(Xtrain)
Xtrain4


# In[39]:


#now it's time to apply all the terms at a once
#for training data
vect_combined = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.5)
XtrainC = vect_combined.fit_transform(Xtrain)
XtrainC


# In[40]:


# for Testing data
XtestC = vect_combined.transform(Xtest)


# In[41]:


#now applying MultinomialNB algoritham
nb = MultinomialNB()
nb.fit(XtrainC, Ytrain)


# In[42]:


YpredictC = nb.predict(XtestC)


# In[48]:


# our Old score is 0.9856459330143541
metrics.accuracy_score(Ytest, YpredictC)


# In[ ]:


# our Old score is 0.9856459330143541
# our new score is 0.9874401913875598
# little Improvement

