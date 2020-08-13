
# coding: utf-8

# In[1]:


import os ,sys 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from operator import itemgetter
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report, confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.svm import SVC, SVR
from scipy.stats import zscore
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_predict , cross_val_score ,KFold
import math
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
#get_ipython().magic(u'matplotlib inline')


# In[29]:


df = pd.read_csv("SubCh1_TrainingData.csv")


# In[30]:


# df.head()


# In[31]:


# df.shape


# In[32]:


# df.info()


# In[33]:


# df.describe()


# In[34]:


# df.columns.values


# In[35]:


df.set_index("Sample_Name",inplace=True)


# In[37]:


# df.head()


# In[39]:


y = df["DHA_IC50"]


# In[41]:


X = df.drop("DHA_IC50",axis=1)


# In[43]:


print (df["Isolate"].unique())


# In[44]:


print (df["Timepoint"].unique())


# In[45]:


print (df["Treatment"].unique())


# In[46]:


df_test = pd.read_csv("SubCh1_TestData.csv")
df_test.set_index("Sample_Name",inplace=True)
X_test = df_test.drop("DHA_IC50",axis=1)
y_test = df_test["DHA_IC50"]


# In[48]:


# for n,m in zip(df.columns , df_test.columns):
#     print (n,m)


# In[54]:


# df[ (df["Isolate"] == "isolate_01") & (df["Timepoint"] == "24HR") & (df["Treatment"] == 'DHA') ]


# In[ ]:


# my_df = df[ (df["Isolate"] == "isolate_01") & (df["Timepoint"] == "24HR") & (df["Treatment"] == 'DHA') ].iloc[:, 2:17]


# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


rfc = RandomForestRegressor(n_jobs = -1)


# In[ ]:


cv = KFold(5, shuffle=True)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rfc, 
                               param_distributions = random_grid, 
                               n_iter = 100, cv = cv, verbose=2, random_state=101, n_jobs = -1)


# In[ ]:


rf_random.fit(X,y )

print (rf_random.best_params_)
# In[ ]:


y_pred=rf_random.predict(X_test)


# In[ ]:


print (classification_report(y_test, y_pred))


# In[ ]:


print ("MCC %f"%(matthews_corrcoef(y_test, y_pred)))
print ("Accuracy %f"%(accuracy_score(y_test, y_pred)))

