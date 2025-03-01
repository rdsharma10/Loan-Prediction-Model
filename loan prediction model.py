#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset=pd.read_csv("train.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset.info()


# In[7]:


pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True)


# In[8]:


dataset.boxplot(column='ApplicantIncome')


# In[9]:


dataset['ApplicantIncome'].hist(bins=20)


# In[10]:


plt.show()


# In[11]:


dataset.boxplot(column='LoanAmount')


# In[12]:


plt.show()


# In[13]:


dataset['LoanAmount'].hist(bins=20)


# In[14]:


plt.show()


# In[15]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[16]:


import numpy as np


# In[17]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[18]:


plt.show()


# In[19]:


dataset.isnull().sum()


# In[20]:


dataset.fillna({'Gender': dataset['Gender'].mode()[0]}, inplace=True)


# In[21]:


dataset.fillna({'Married': dataset['Married'].mode()[0]}, inplace=True)


# In[28]:


dataset.fillna({'Dependents': dataset['Dependents'].mode()[0]}, inplace=True)


# In[29]:


dataset.fillna({'Self_Employed': dataset['Self_Employed'].mode()[0]}, inplace=True)


# In[30]:


dataset.LoanAmount=dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log=dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[31]:


dataset.fillna({'Loan_Amount_Term': dataset['Loan_Amount_Term'].mode()[0]}, inplace=True)


# In[32]:


dataset.fillna({'Credit_History': dataset['Credit_History'].mode()[0]}, inplace=True)


# In[33]:


dataset.isnull().sum()


# In[35]:


dataset['TotalIncome']=dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])


# In[36]:


dataset['TotalIncome_log'].hist(bins=20)


# In[37]:


plt.show()


# In[38]:


dataset.head()


# In[40]:


x=dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y=dataset.iloc[:,12].values


# In[41]:


x


# In[42]:


y


# In[44]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[45]:


print(x_train)


# In[46]:


from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()


# In[48]:


for i in range(0,5):
    x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])


# In[49]:


x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])


# In[50]:


x_train


# In[51]:


labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train)


# In[52]:


y_train


# In[53]:


for i in range(0,5):
    x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])


# In[54]:


x_test[:,7]=labelencoder_x.fit_transform(x_test[:,7])


# In[55]:


labelencoder_y=LabelEncoder()
y_test=labelencoder_y.fit_transform(y_test)


# In[56]:


x_test


# In[57]:


y_test


# In[59]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[61]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(x_train,y_train)


# In[62]:


y_pred=DTClassifier.predict(x_test)
y_pred


# In[64]:


from sklearn import metrics
print('The accuracy of decision tree is :',metrics.accuracy_score(y_pred,y_test))


# In[65]:


from sklearn.naive_bayes import GaussianNB
NBClassifier=GaussianNB()
NBClassifier.fit(x_train,y_train)


# In[66]:


y_pred=NBClassifier.predict(x_test)


# In[67]:


y_pred


# In[68]:


print('The accuracy of Naive Bayes is :',metrics.accuracy_score(y_pred,y_test))


# In[69]:


testdata=pd.read_csv('test.csv')


# In[70]:


testdata.head()


# In[71]:


testdata.isnull().sum()


# In[72]:


testdata.fillna({'Gender': dataset['Gender'].mode()[0]}, inplace=True)


# In[74]:


testdata.fillna({'Dependents': dataset['Dependents'].mode()[0]}, inplace=True)
testdata.fillna({'Self_Employed': dataset['Self_Employed'].mode()[0]}, inplace=True)
testdata.fillna({'Loan_Amount_Term': dataset['Loan_Amount_Term'].mode()[0]}, inplace=True)
testdata.fillna({'Credit_History': dataset['Credit_History'].mode()[0]}, inplace=True)




# In[75]:


testdata.isnull().sum()


# In[78]:


testdata.boxplot(column='LoanAmount')


# In[79]:


plt.show()


# In[80]:


testdata.boxplot(column='ApplicantIncome')


# In[81]:


testdata.LoanAmount=testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[82]:


testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[83]:


testdata.isnull().sum()


# In[85]:


testdata['TotalIncome']=testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[86]:


testdata.head()


# In[87]:


test=testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[88]:


for i in range(0,5):
    test[:,i]=labelencoder_x.fit_transform(test[:,i])


# In[90]:


test[:,7]=labelencoder_x.fit_transform(test[:,7])


# In[91]:


test


# In[92]:


test=ss.fit_transform(test)


# In[93]:


pred=NBClassifier.predict(test)


# In[94]:


pred


# In[ ]:




