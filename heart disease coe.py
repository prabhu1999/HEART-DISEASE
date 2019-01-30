#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


heart_df=pd.read_csv('framingham.csv')


# In[4]:


heart_df.drop(['education'],axis=1,inplace=True)


# In[5]:


heart_df.head()


# In[6]:


heart_df.rename(columns={'male':'Sex_male'},inplace=True)


# In[7]:


heart_df.isnull().sum()


# In[8]:


count=0
for i in heart_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ',count)
print('since it is only',round((count/len(heart_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# In[9]:


heart_df.dropna(axis=0,inplace=True)


# In[19]:


def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()
    plt.show()
draw_histograms(heart_df,heart_df.columns,6,3)


# In[14]:


heart_df.TenYearCHD.value_counts()


# In[21]:


sn.countplot(x='TenYearCHD',data=heart_df)


# In[35]:


sn.pairplot(data=heart_df)


# In[36]:


heart_df.describe()


# In[38]:


from statsmodels.tools import add_constant as add_constant 
heart_df_constant = add_constant(heart_df)
heart_df_constant.head()


# In[40]:


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df) 
cols=heart_df_constant.columns[:-1]
model=sm.Logit(heart_df.TenYearCHD,heart_df_constant[cols])
result=model.fit()
result.summary()


# In[41]:


def back_feature_elem (data_frame,dep_var,col_list):
    while len(col_list)>0:
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)
result=back_feature_elem(heart_df_constant,heart_df.TenYearCHD,cols)


# In[42]:


result.summary()


# In[44]:


params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue 
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))


# In[50]:


import sklearn
new_features=heart_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn import model_selection #import train_test_split
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=.20,random_state=5)


# In[55]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[56]:


sklearn.metrics.accuracy_score(y_test,y_pred)


# In[58]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap='YlGnBu')


# In[59]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# In[60]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',  
      'The Missclassification = 1 - Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',
      'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
      'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',
      'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',
      'Negative Predictive value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',
      'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',
      'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# In[61]:


y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of no Heart Disease (1)'])
y_pred_prob_df.head()


# In[64]:


from sklearn.preprocessing import binarize
for i in range(1,5):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With' ,i/10,'thershold the Confusion Matrix is ','n',cm2,'\n',  
           'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors(False Negatives)','\n\n',
            'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')


# In[65]:


from sklearn.metrics import roc_curve
fpr, tpr, thershold = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specitivity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[66]:


sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])


# In[ ]:




