#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("happiness_score_dataset.csv")
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# We are checking missing value
df.isnull().sum()


# In[7]:


sns.scatterplot(x="Economy (GDP per Capita)", y="Happiness Score", data=df)


# In[8]:


sns.scatterplot(x="Family", y="Happiness Score", data=df)


# In[9]:


sns.scatterplot(x="Health (Life Expectancy)", y="Happiness Score", data=df)


# In[10]:


sns.scatterplot(x="Freedom", y="Happiness Score", data=df)


# In[11]:


sns.scatterplot(x="Trust (Government Corruption)", y="Happiness Score", data=df)


# In[12]:


sns.scatterplot(x="Generosity", y="Happiness Score", data=df)


# In[13]:


six_factor=df[['Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity']]
six_factor


# In[14]:


six_factor.corr()


# In[15]:


six_factor.corr()['Happiness Score'].sort_values()


# In[16]:


#Economy (GDP per Capita) contributes most positively to the Happiness Score compared to other factors


# Now we will show Correlation using heatmap

# In[17]:


plt.figure(figsize=(10,8))
sns.heatmap(six_factor.corr(), annot=True, linewidths = 0.5, linecolor = 'black', fmt='.2f')


# All the columns of the datasets is positively correlated with the target column
# 
# 1. Economy (GDP per Capita) has 78 percent correlation with the target column which can be considered as a strong bond
# 2. Family has 74 percent correlation with the target column which can be considered as a strong bond
# 3. Health (Life Expectancy) has 72 percent correlation with the target column which can be considered as a strong bond
# 4. Freedom has 57 percent correlation with the target column which can be considered as a strong bond
# 5. Trust (Government Corruption) has 40 percent correlation with the target column which can be considered as a strong bond
# 6. Generosity has 18 percent correlation with the target column which can be considered as a good bond

# In[18]:


six_factor.describe()


# In[19]:


plt.figure(figsize=(15,12))
sns.heatmap(round(six_factor.describe()[1:].transpose(),2), linewidth=2, annot=True, fmt='f')
plt.xticks(fontsize=18)
plt.yticks(fontsize=12)
plt.title("Variables Summary")
plt.show()


# In[20]:


six_factor.info()


# In[21]:


#checking outliers
import warnings
warnings.filterwarnings('ignore')


# In[22]:


collist=six_factor.columns.values
ncol=30
nrows=14
plt.figure(figsize=(ncol,3*ncol))
for i in range (0, len(collist)):
    plt.subplot(nrows, ncol, i+1)
    sns.boxplot(data = six_factor[collist[i]], color='green', orient='v')
    plt.tight_layout()


# In[23]:


six_factor.skew()


# In[24]:


#skewness threshold is taken +/-0.1


# In[25]:


sns.distplot(six_factor["Economy (GDP per Capita)"])


# In[26]:


#Building block is out of normalised curve. 


# In[ ]:





# In[27]:


sns.distplot(six_factor["Family"])


# In[28]:


#Building block is out of normalised curve. 


# In[ ]:





# In[29]:


sns.distplot(six_factor["Health (Life Expectancy)"])


# In[30]:


#Building block is out of normalised curve. 


# In[ ]:





# In[31]:


sns.distplot(six_factor["Freedom"])


# In[32]:


#Building block is out of normalised curve. 


# In[ ]:





# In[33]:


sns.distplot(six_factor["Trust (Government Corruption)"])


# In[34]:


#Building block is out of normalised curve. 


# In[ ]:





# In[35]:


sns.distplot(six_factor["Generosity"])


# In[36]:


#Building block is out of normalised curve. 


# In[37]:


#Normal distribution shows that the datas (Above) are skeewed


# In[ ]:





# In[38]:


six_factor.corr()['Happiness Score']


# In[55]:


np.abs(zscore(six_factor))


# In[54]:


from scipy.stats import zscore
import numpy as np
z=np.abs(zscore(six_factor))
z.shape


# In[56]:


type(z)


# In[40]:


threshold=3
print(np.where(z>3))


# In[59]:


df.drop([27, 128, 147, 153, 157], axis=0)


# In[136]:


len(np.where(z>1)[0])


# In[137]:


six_factor_new=six_factor[(z<3).all(axis=1)]
print('Old DataFrame', six_factor.shape)
print('New DataFrame', six_factor_new.shape)
print('Total Dropped rows', six_factor.shape[0] - six_factor_new.shape[0])


# In[138]:


loss_percent=(158-153)/158*100
print(loss_percent, '%')


# In[139]:


six_factor_new


# In[140]:


x=six_factor_new.iloc[1:,:].values


# In[141]:


y=six_factor_new.iloc[1,:].values


# In[142]:


from sklearn import preprocessing
from sklearn import utils

x=x.transpose()
x.shape


# In[143]:


y.shape


# In[144]:


from sklearn.preprocessing import power_transform
x=power_transform(x,method='yeo-johnson')


# In[145]:


x


# In[146]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[ ]:





# In[ ]:




