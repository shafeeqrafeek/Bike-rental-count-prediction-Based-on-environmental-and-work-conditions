
# coding: utf-8

# In[141]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sr
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from fancyimpute import KNN


# In[142]:


os.chdir("F:\Data Analytics\Edwisor Project\Bike Rental Project\Python")
os.getcwd()


# In[143]:


bk_data=pd.read_csv("bike_rental.csv")


# In[144]:


bk_factors=['season','yr','mnth','holiday','weekday','workingday','weathersit']
bk_numeric=['instant','temp','atemp','hum','windspeed','casual','registered','cnt']
for i in bk_factors:
        bk_data[i]=bk_data[i].astype('category')


# # Exploratory Data Analysis
# - the relationship between humidity and rental count seems to be flat at first. But when we introduce season in to this relationship, there appears a descreasing trend in count with increasing humidity as the season moves from spring towards winter
# - a scatterplot to find out how rental count varies with season shows that the count at the start of spring and it starts to increase as the season changes. it reaches its maximum in fall and then starts to drop again when winter arrives. As the 'M' shaped scatterplot depicts, this is clearly the case for both 2011 and 2012  
# - as per the average number of bikes rented displayed by the bargraph, people are more likely to rent a bike when the weather is clear as compared to situations when there is a high chance of rain

# In[145]:


sns.regplot('windspeed','cnt',data=bk_data)
plt.title("Humidity vs Rental count")
#plt.savefig("humidity_vs_count.jpeg")


# In[146]:


fct=sns.FacetGrid(bk_data,col='season')
fct=fct.map(sns.regplot,'hum','cnt')
#plt.savefig("humidity_vs_count_per_season.jpeg")


# In[147]:


sns.lmplot('instant','cnt',data=bk_data,hue='season')
plt.title("seasonal variation of count")
#plt.savefig("seasonal_variation_of_count.jpeg")


# In[148]:


sns.barplot('weathersit','cnt',data=bk_data)
plt.xticks([0,1,2],['clear','misty','light rain'])
plt.title('weather vs count')
#plt.savefig("weather_vs_count.jpg")


# In[149]:


# boxplot visualisation 
fig,axes=plt.subplots(2,4,figsize=(8,6))
axes=axes.reshape(1,8)
axes=axes.flatten()
for i,j in enumerate(bk_numeric):
    axes[i]=axes[i].boxplot(bk_data.loc[:,j],labels=[j])
fig.suptitle("Box Plot Visualisation",y=1.05)
fig.tight_layout()
#plt.savefig("boxplot_n.jpg")


# In[150]:


#outlier removal
for i in bk_numeric:
    q25,q75=np.percentile(bk_data.loc[:,i],[25,75])
    iqr=q75-q25
    max=q75+(1.5*iqr)
    min=q25-(1.5*iqr)
    bk_data=bk_data.drop(bk_data.loc[bk_data.loc[:,i]>max,:].index)
    bk_data=bk_data.drop(bk_data.loc[bk_data.loc[:,i]<min,:].index)                   


# In[151]:


#Feature Engineering
bk_corr=bk_data.loc[:,bk_numeric]
bk_corr=bk_corr.corr()
pl=sns.diverging_palette(10,220,as_cmap=True)
sns.heatmap(bk_corr,cmap=pl,square=True,vmin=-0.6)
#plt.savefig("correlation_heatmap.jpg")


# ## Conclusion:
# -  temp and atemp are highly correlated so atemp has to be removed
# -  dteday can be removed as there are seperate columns for month and year
# -  instance can be removed as it merely is for numbering the observations

# In[152]:


col_rem = ['instant','dteday','atemp']
bk_data=bk_data.drop(col_rem,axis=1)


# ## Model Phase

# In[153]:


bk_data.to_csv("bk_data_model_phase.csv",index=False)
bk_mdata=pd.read_csv("bk_data_model_phase.csv")
train,test=train_test_split(bk_mdata,test_size=0.2)
#error metric
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# #### Linear Regression

# In[154]:


#linear regression
bk_model=sr.OLS(train.iloc[:,10:13],train.iloc[:,0:10]).fit()
predict_lin=bk_model.predict(test.iloc[:,0:10])
predict_lin.columns=['casual','registered','count']
predict_lin['actual']=test.iloc[:,12]


# In[155]:


#MAPE - linear regression
print("casual:",MAPE(test['casual'],predict_lin['casual']))
print("registered:",MAPE(test['registered'],predict_lin['registered']))
print("cnt:",MAPE(test['cnt'],predict_lin['count']))


# #### Decision Tree

# In[156]:


#Decision Tree
bk_model_dt=DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:10],train.iloc[:,10:13])
predict_dt=pd.DataFrame(bk_model_dt.predict(test.iloc[:,0:10]),columns=['casual','registered','cnt'])
test1=test.reset_index(drop=True)
predict_dt['actual']=test1.iloc[:,12]


# In[157]:


#MAPE - Decision Tree
print("casual:",MAPE(test1['casual'],predict_dt['casual']))
print("registered:",MAPE(test1['registered'],predict_dt['registered']))
print("cnt:",MAPE(test1['cnt'],predict_dt['cnt']))


# #### Random Forest

# In[158]:


#Random Forest
bk_model_rf=RandomForestRegressor(n_estimators=10).fit(train.iloc[:,0:10],train.iloc[:,10:13])
predict_rf=pd.DataFrame(bk_model_rf.predict(test.iloc[:,0:10]),columns=['casual','registered','cnt'])
test1=test.reset_index(drop=True)
predict_rf['actual']=test1.iloc[:,12]


# In[159]:


#MAPE - Random Forest
print("casual:",MAPE(test1['casual'],predict_rf['casual']))
print("registered:",MAPE(test1['registered'],predict_rf['registered']))
print("cnt:",MAPE(test1['cnt'],predict_rf['cnt']))


# #### KNN

# In[160]:


#KNN
bk_KNN=bk_mdata.copy()
bk_KNN.loc[test.index,['casual','registered','cnt']]=np.nan
bk_KNN=pd.DataFrame(KNN(k=3).complete(bk_KNN),columns=bk_KNN.columns)
predict_KNN=bk_KNN.loc[test.index,['casual','registered','cnt']]
predict_KNN['actual']=test['cnt']


# In[161]:


#KNN
print("casual:",MAPE(test['casual'],predict_KNN['casual']))
print("registered:",MAPE(test['registered'],predict_KNN['registered']))
print("cnt:",MAPE(test['cnt'],predict_KNN['cnt']))

