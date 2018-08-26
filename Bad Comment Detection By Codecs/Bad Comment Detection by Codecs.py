
# coding: utf-8

# In[54]:


import sys
import spacy
import numpy
import pandas
import matplotlib
import seaborn
from keras.models import model_from_json


# In[55]:


print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Spacy: {}'.format(spacy.__version__))


# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy as sp


# In[51]:


data=pd.read_csv('train.csv')


# In[18]:


print(df.columns)


# In[19]:


df=pd.read_csv('train.csv')
train=df
train.insert(2,"clean",train['toxic']^1)
train_comments=train['comment_text']
train_toxic_comments=df.loc[train['toxic'] -- 1]
df=pd.read_csv('test.csv')
test=df
train_comments=df['comment_text']


# In[20]:


print(df)


# In[21]:


df=pd.read_csv('train.csv')
train=df
train.insert(2,"clean",train['toxic']^1)
train_comments=train['comment_text']
train_toxic_comments=df.loc[train['toxic'] -- 1]
df=pd.read_csv('test.csv')
test=df
train_comments=df['comment_text']


# In[22]:


print(df.describe())


# In[23]:


print(df.shape)


# In[24]:


print(df.columns)


# In[47]:


df=df.sample(frac=0.1,random_state=2)
print(df.shape)
print(df.describe())


# In[52]:


data.hist(figsize=(15,15))
plt.show()


# In[53]:


print(data.hist)


# In[60]:


nlp=spacy.load("data")


# In[65]:


data2=pd.read_csv('sample_submission.csv')


# In[67]:


data.hist(figsize=(5,5))
plt.show()


# In[68]:


data3=pd.read_csv('Book1.csv')


# In[69]:


data3.hist(figsize=(10,10))
plt.show()


# In[72]:


print(data3.columns)


# In[73]:


print(data3.shape)


# In[74]:


print(data3.describe())

