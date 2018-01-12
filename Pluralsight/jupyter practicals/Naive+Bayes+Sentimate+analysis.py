
# coding: utf-8

# In[4]:

with open("/Users/473703/Documents/ML/Pluralsight/sentiment labelled sentences/imdb_labelled.txt","r") as text_file:
    lines=text_file.read().split('\n')


# In[5]:

lines


# In[20]:

lines = [line.split("\t") for line in lines if len(line.split("\t"))==2 and line.split("\t")[1]!='']


# In[21]:

lines


# In[44]:

train_documents= [line[0] for line in lines]
train_documents

train_label= [int(line[1]) for line in lines]


# In[45]:


train_label
from sklearn.feature_extraction.text import CountVectorizer 
count_vectorizer=CountVectorizer(binary='true')
train_documents=count_vectorizer.fit_transform(train_documents)
train_documents


# In[60]:

#training Phase
from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB().fit(train_documents,train_label)

# Test Phase
classifier.predict(count_vectorizer.transform(['this is best movies']))

