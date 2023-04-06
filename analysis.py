#!/usr/bin/env python
# coding: utf-8

# ### Approach to problem
# * First load data and perform basic cleaning
# * Make a baseline dirty model and check performance 
# * Perform in depth EDA, Pre processing, Feature engineering
# * Develop an improved model. See how this compares to our baseline model
# * Once we have our model score, train a decision tree based on these scores and necessary features to find rules that can be recommended to bank managers

# ### Libraries import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn import tree



import xgboost
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


from statsmodels.stats.outliers_influence import variance_inflation_factor


# ### Dataset import

# In[2]:


data = pd.read_csv('sp23_datachallenge.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# ### A million rows and 32 columns - 31 independent features

# In[5]:


data.fraud_bool.value_counts()


# In[6]:


100 * data.fraud_bool.value_counts()[0]/(data.fraud_bool.value_counts()[0] + data.fraud_bool.value_counts()[1])
# 98.89 % rows have no fraud


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# ### No null values present in any column. 

# In[9]:


data.describe().transpose()


# ### List  of categorical columns

# In[10]:


categorical_columns = list(data.select_dtypes(['object']))
categorical_columns


# ### One hot encoding

# In[11]:


data_encoded = pd.get_dummies(data, categorical_columns,drop_first=True)
len(data_encoded.columns)


# #### 48 columns

# ### Train test splitting

# In[12]:


X = data_encoded.iloc[:,1:]
y = data_encoded.iloc[:,0]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 42)


# In[14]:


print(X_train.shape, X_test.shape)


# ### Baseline model - XGBoost

# In[15]:


xgb_baseline = XGBClassifier()
xgb_baseline.fit(X_train,y_train)
ypred_baseline = xgb_baseline.predict(X_test)


# In[16]:


cm_baseline = confusion_matrix(ypred_baseline,y_test)


# In[17]:


cm_baseline


# ### ROC Curve/ROC AUC score

# In[18]:


roc_auc_baseline = roc_auc_score(y_test, xgb_baseline.predict_proba(X_test)[:, 1])


# In[19]:


roc_auc_baseline


# ## <div class="alert alert-block alert-warning"> We get an ROC AUC score of 0.89 which is good. We will try to improve on it

# In[20]:


fpr_baseline, tpr_baseline, thresholds_baseline = metrics.roc_curve(y_test, xgb_baseline.predict_proba(X_test)[:, 1])
plt.plot(fpr_baseline, tpr_baseline)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# ## <div class="alert alert-block alert-warning">  We got a good baseline auc score and a good ROC Curve as well.

# ### Now we will do multiple cleaning, eda, visualization, feature engineering steps and will try to improve our AP score and better our precision recall curve

# #### Heatmap

# In[21]:


#plt.figure(figsize = (12,12))
sns.heatmap(data.corr(),cmap = 'YlGnBu',annot=True,cbar=True, fmt='.2f')
plt.gcf().set_size_inches(15, 12)
plt.show()


# In[22]:


data.describe().transpose()


# In[23]:


### There are some columns where there are -1 or negative values. Lets see which columns are these
# previous address count - This doesnt have zero
# current address count - This has 0 as well
# intended_balcon_amount - have negative values
# velocity_6h - has negative values
# credit risk score - negatived
# session_length_in_minutes - negative values
# device_distinct_emails_8w - negative values


# In[24]:


len(data[data['current_address_months_count'] == 0])  #9609 
len(data[data['current_address_months_count'] == -1]) #4254


# ### Handling negative values in each column

# * 1. prev_address_months_count : -1 converted to 0
# * 2. current_address_months_count : -1 converted to 0
# 
# #### Since people who have recently moved in have marked 0, -1 means that data is missing. Since there is a large number of data missing, we will let it be -1. 
# #### I also plan to create a feature that adds total months of stay in current as well as old address. So having a -1 doesnt make sense in summation, we will add +1 to avoid summation with -1 then

# In[4]:


copy = data.copy()


# In[26]:


data = copy.copy()


# In[ ]:





# * 3. intended_balcon_amount -  No change
# 
# #### Negative values here might be for the cases where banks give line of credit/loans to new users on the basis of their credit history. So the first transaction/opening balance can be negative also. But highest value is 113. It seems like this feature has been scaled down and should be treated accordingly

# * velocity_6h : Multiply by -1

# In[5]:


#checking negative velocity rows - 44 rows
data[data['velocity_6h']<0]['velocity_6h']
# There is a high chance that its a data entry error and signs have been inverted
data['velocity_6h'].iloc[data[data['velocity_6h']<0].index] = data['velocity_6h'].iloc[data[data['velocity_6h']<0].index] * -1


# * credit risk score - no change
# 
# #### Generally credit scores are in range 0-1000. But this is some internal scores where both lower and upper bounds are different. So I am not changing it

# * bank_months_count - no change
# #### -1 means missing here. We dont have info about large proportion of people if they had an old bank account or not. We can keep it -1/missing = yes anything. It serves the same purpose to model

# * session_length_in_minutes - make it zero
# ### this should generally not be negative

# In[6]:


data['session_length_in_minutes'].loc[data[data['session_length_in_minutes'] < 0].index] = 0


# * device_distinct_emails_8w - change to zero
# #### device_distinct_emails_8w - no of emails cant be negative. The range given in excel is also starting from 0

# In[7]:


data['device_distinct_emails_8w'].loc[data[data['device_distinct_emails_8w'] < 0].index] = 0


# # <div class="alert alert-block alert-success"> Univariate Analysis, Multivariate Analysis - EDA

# In[8]:


fraud_data = data[data['fraud_bool'] == 1]
nonfraud_data = data[data['fraud_bool'] == 0]


# ### <div class="alert alert-block alert-info"> Income analysis

# In[9]:


plt.figure(figsize = (4,4))
sns.boxplot(data = data,y ='income',x='fraud_bool')
plt.xlabel("Fraud or Not Fraud")
plt.ylabel("Income")


# ### <div class="alert alert-block alert-warning"> The average income of fraudulent applications are higher than non fraudulent applications

# ### Derived Feature - income_category
# * Bucket1 = 0th-25th quantile
# * Bucket2 = 25th-75th quantile
# * Bucket3 = 75th-100th quantile

# In[10]:


data['income_category'] = pd.qcut(data['income'], q=[0, .25, .75, 1], labels=['low', 'medium', 'high'])


# #### Now lets see proportion of fraud across different income groups

# In[33]:


sns.countplot(data[data['fraud_bool'] == 1], x= 'fraud_bool',hue = 'income_category')


# ### <div class="alert alert-block alert-warning"> There are more fraud cases occuring in medium and high income groups rather than low income groups 

# ### <div class="alert alert-block alert-info"> Address count analysis

# In[34]:


#sns.boxplot(data, y='current_address_months_count',x='fraud_bool')


# ###  derived feature - address counts : if a person has lived for long in their current home or they lived long in their last home and have recently moved in, it shows stability. We can segment them into category accordingly

# In[11]:


data['address_counts']  = data['prev_address_months_count'] + data['current_address_months_count'] +2 # to address -1 issue


# ### derived feature - address count category
# * Bucket1 = 0th-25th quantile
# * Bucket2 = 25th-75th quantile
# * Bucket3 = 75th-100th quantile

# In[13]:


data['address_stability'] = pd.qcut(data['address_counts'], q=[0, .25, .75, 1], labels=['low', 'medium', 'high'])


# In[14]:


data


# ### Now lets see proportion of fraud across different address count category groups¶

# In[15]:


sns.countplot(data[data['fraud_bool'] == 1], x= 'fraud_bool',hue = 'address_stability')


# ### <div class="alert alert-block alert-warning"> Fraudsters tend to show atleast some stable housing history. This shows they have taken necessary steps to conceal their fraudulent activities and create a false sense of trust or legitimacy. 

# <br>

# ### <div class="alert alert-block alert-info">  Age analysis

# #### No. of fraud cases across different age groups

# In[20]:


age_fraud = data.groupby('customer_age')['fraud_bool'].sum().reset_index()
age_fraud
age_fraud['age_bucket'] = ['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99']

plt.figure(figsize = (5,5))
sns.barplot(data =age_fraud , y = 'fraud_bool',x = 'age_bucket')
plt.title("Fraud count across different age buckets")


# ### <div class="alert alert-block alert-warning"> Looks like people aged 20-59 have most fraud occurences. But the number of rows for these age groups are also high. So is this picture real ?
# #### Lets find out proportion between fraud and non fraud applications 

# #### Lets find out fraud proportion for different age groups (for each age bucket, calculate no of fraud applcns/total no of applications)

# In[40]:


age_fraud = 100*(data.groupby('customer_age')['fraud_bool'].sum()/data.groupby('customer_age')['fraud_bool'].count()).reset_index()
age_fraud['customer_age'] = age_fraud['customer_age']/100
age_fraud['age_bucket'] = ['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99']
age_fraud.drop('customer_age',axis=1)

plt.figure(figsize = (4,4))
plt.pie(age_fraud['fraud_bool'],labels = age_fraud['age_bucket'])
plt.legend(loc = 'center')
plt.title("Fraud rate across different age buckets")


# ## <div class="alert alert-block alert-warning"> <i> The proportion of fraud applications against non fraud applications is higher for elderly people . Thus we can say that in reality out of all applications, there are much higher chances of fraud for elderly people. This is because they are less tech savvy, and are more prone to being victims of stolen identity<i>

# In[41]:


age_fraud_rate = 100*data.groupby('customer_age')['fraud_bool'].mean().reset_index()
age_fraud_rate['customer_age'] = age_fraud_rate['customer_age']/100
age_fraud_rate.columns = ['customer_age','fraud rate in percentage']
age_fraud_rate


# ### <div class="alert alert-block alert-warning"> Fraud rate increases as age increases

# #### We saw from pie chart that people aged 60 or above have good proportion of fraud cases. Lets see if such wealthy elderly people are more prone to doing fraud?

# In[42]:


old_age_data = data[data['customer_age'] > 59]  #42660
#old_age_data['fraud_bool'].value_counts()
old_age_data['income'].describe()  #75th percentile is 0.8
old_age_data[old_age_data['income']>0.8]['fraud_bool'].value_counts()

# 6.7% chances


# ### <div class="alert alert-block alert-warning"> <i> People who are aged 60 or above and they earn in top 25% of this age bucket, have 7% chances of getting defrauded <i>

# #### Lets see if they use website or app to send the application

# In[43]:


old_age_data['source'].value_counts()


# ### <div class="alert alert-block alert-warning"> 98.5% people who are aged 60 or above use website browsers to submit applications. But on further investigation no specific fraud trend found
# 
# ### <div class="alert alert-block alert-warning"> Conclusion -  <i> Fraud rate increases as customer's age increases. People who are aged 60 or above and they earn in top 25% of this age bucket, have 7% chances of getting defrauded <i>

# ###  <div class="alert alert-block alert-info"> intended_balcon_amount analysis

# In[44]:


data.intended_balcon_amount.describe()


# **We see the value ranges between -15 to 112. Firstly can the values be negative? Yes since there are cases where banks give line of credit/loans to new users on the basis of their credit history. So the first transaction can be negative also. But highest value is 108. It seems like this feature has been scaled down and should be treated accordingly.**

# ### Derived Feature - line_of_credit = 1 if intended_balcon_amount<0 else 0

# In[45]:


data['line_of_credit'] = np.where(data['intended_balcon_amount']<0,1,0)


# ### People with low income but high balcon amount could potentially be a red flag as they might look to open an account to deposit illegal money
# 
# #### Lets study relationship of intended balcon amount + income against fraud
# 

# In[46]:


high_initial_transfer = data[data['intended_balcon_amount']>4.9]  #people with high initial transfer (above 75th percentile)
sns.countplot(high_initial_transfer[high_initial_transfer['fraud_bool'] == 1], x='income_category')
plt.title("Income of Fraudsters who have high initial transfer amount")


# ### Surprisingly, we didn't observe the expected behavior

# ###  <div class="alert alert-block alert-info"> Payment plan analysis

# #### Lets see distribution of payment plans across fraudulent data

# In[47]:


sns.countplot(data = fraud_data, x = 'payment_type')


# ### Trend - > AB = AC > AA = AD > AE

# In[48]:


data['payment_type'].value_counts()


# In[49]:


fraud_data['payment_type'].value_counts()


# ### For group AE, only 1 fraudulent row is present

# #### For fraudulent data,are there any prevalent payment  plans and  income group?

# In[50]:


fraud_data = data[data.fraud_bool == 1]
income_payment_data = fraud_data.groupby('payment_type')['income_category'].value_counts()
prop = pd.DataFrame(income_payment_data.groupby(level=0).apply(lambda x: 100*x / x.sum()))
prop.columns = ['Income category proportion for fraud applications']
prop


#  
# * #### <div class="alert alert-block alert-warning"> <i> For plan AE, we see that all the attacks come from medium income category i.e bands [0.3- 0.8] <i> . But this is not a trend as there was only 1 fraudulent row.
# * #### <div class="alert alert-block alert-warning"> <i> For rest of the group, medium and high income applicants have much more chances to be fraud than low income applicants
#     
# ### <div class="alert alert-block alert-warning"> Conclusion - No specific income group in a payment plan depict a high chance of fraud
#     

# #### lets test the same thing with address stability

# In[51]:


data


# In[52]:


fraud_data = data[data['fraud_bool'] == 1]
address_payment_data = fraud_data.groupby('payment_type')['address_stability'].value_counts()
prop = pd.DataFrame(address_payment_data.groupby(level=0).apply(lambda x: 100*x / x.sum()))
prop.columns = ['Address stability proportion for fraud applications']
prop


# #### This is in accordance to bar plot plotted earlier which told medium address history were more prevelant to fraud. We see the same trend here

# ###  <div class="alert alert-block alert-info"> Velocity variables analysis

# ### We have month variable. Lets find out average velocity (6h,24h,4w) for each month. And see for rows where veloctiy was greater than mean, did the fraud increase?

# In[53]:


data.groupby(['month'])[['velocity_6h','velocity_24h','velocity_4w']].mean()


# In[54]:


data.groupby(['month','fraud_bool'])[['velocity_6h','velocity_24h','velocity_4w']].mean()


# ### <div class="alert alert-block alert-warning">There is a clear pattern that for each month, whenever mean velocity at any time is lower, there is a higher chance of application being fraud than non fraud. This is quite surprising as one would expect the fraudsters to use times of heavy traffic to infiltrate the system.

# ### Derived Feature - velocity_6h_risky,velocity_24h_risky,velocity_4w_risky : for each row, given the month that row belongs to, if the average velocity is less than mean velocity for the month, flag it 1 else 0 

# In[55]:


monthly_mean_velocity = data.groupby(['month'])[['velocity_6h','velocity_24h','velocity_4w']].mean().reset_index()
monthly_mean_velocity.columns = ['month', 'mean_velocity_6h', 'mean_velocity_24h', 'mean_velocity_4w']
monthly_mean_velocity


# In[56]:


# create dictionary for all 3 variables:
dict_velocity_6h = monthly_mean_velocity['mean_velocity_6h'].to_dict()
dict_velocity_24h = monthly_mean_velocity['mean_velocity_24h'].to_dict()
dict_velocity_4w = monthly_mean_velocity['mean_velocity_4w'].to_dict()


# In[57]:


data['velocity_6h_risky'] = ""
data['velocity_24h_risky'] = ""
data['velocity_4w_risky'] = ""
data['velocity_6h_risky'] = data.apply(lambda row: 1 if row['velocity_6h'] < dict_velocity_6h[row['month']] else 0, axis=1)
data['velocity_24h_risky'] = data.apply(lambda row: 1 if row['velocity_24h'] < dict_velocity_24h[row['month']] else 0, axis=1)
data['velocity_4w_risky'] = data.apply(lambda row: 1 if row['velocity_4w'] < dict_velocity_4w[row['month']] else 0, axis=1)

        
       


# #### Lets see if this derived feature is able to capture frauds or not

# In[58]:


print("Fraud Capture rate using 6hr traffic : ", 100 * data[data['velocity_6h_risky'] ==1][data[data['velocity_6h_risky'] ==1]['fraud_bool'] == 1].shape[0] / 11029) # 11029 is total fraud cases
print("Fraud Capture rate using 24hr traffic: ", 100 * data[data['velocity_24h_risky'] ==1][data[data['velocity_24h_risky'] ==1]['fraud_bool'] == 1].shape[0] / 11029) # 11029 is total fraud cases
print("Fraud Capture rate using 4w traffic  : ", 100 * data[data['velocity_4w_risky'] ==1][data[data['velocity_4w_risky'] ==1]['fraud_bool'] == 1].shape[0] / 11029) # 11029 is total fraud cases


# ### <div class="alert alert-block alert-warning"> This is pretty good. These features capture more than 50 % of fraudulent cases

# ### <div class="alert alert-block alert-info"> employment_status analysis 

# In[59]:


sns.countplot(fraud_data, x = 'employment_status')
plt.title("Number of fraud occurences in eacch employment group")


# In[60]:


data.groupby('employment_status')['fraud_bool'].mean()* 100


# ### It seemed like all frauds were occuring in group CA. But the proportion says otherwise. Group CA, CC, CG are equally likely to be fraudulent. Lets do label encoding based on fraud rate

# In[61]:


employment_rank = pd.DataFrame(data.groupby('employment_status')['fraud_bool'].mean()).sort_values(by = 'fraud_bool',ascending = False)
employment_rank
employment_rank_dict = employment_rank.to_dict()['fraud_bool']
data['employment_status_encoded'] = data.apply(lambda row : employment_rank_dict[row['employment_status']],axis = 1)


# ### <div class="alert alert-block alert-info"> credit_risk_score analysis
# 

# #### What is the average credit score for fraud vs non fraud

# In[62]:


data['credit_risk_score'].describe()
# -170 to 389


# In[63]:


sns.boxplot(y = data['credit_risk_score'],x = data['fraud_bool'])


# ### There are a lot of outliers. But since no info has been given as to why the values can be negative. We cant estimate the cause of these outliers. It can be the case that the values have been scaled down. Lets keep the outliers
# 
# ### <div class="alert alert-block alert-warning"> credit risk scores are higher for fraud cases

# In[64]:


data.corr()['credit_risk_score'].sort_values()


# ### People having high  credit score risk, and asking for high credit limit can be suspicious. Lets see that

# In[65]:


sns.scatterplot(fraud_data, x = 'credit_risk_score', y ='proposed_credit_limit')


# #### No fixed trend that high credit risk score and high proposed credit limit necessitates fraud

# ### Derived Feature : credit_risk_category

# In[66]:


data['credit_risk_category'] = pd.qcut(data['credit_risk_score'], q=[0, .25, .75, 1], labels=['low', 'medium', 'high'])


# ### Derived Feature - difference between mean credit score and current credit score

# In[67]:


data['credit_risk_score'].mean()


# In[68]:


data['credit_risk_score'].mean()
data['credit_risk_score_delta'] = (data['credit_risk_score'] - data['credit_risk_score'].mean())/ data['credit_risk_score'].std()


# ### <div class="alert alert-block alert-info"> email_is_free analysis
# 

# In[69]:


sns.countplot(x = fraud_data['email_is_free'])


# ### <div class="alert alert-block alert-info"> housing_status analysis
# 

# ### Lets find total proportion of fraud/total rows for each housing category

# In[70]:


housing_rank = pd.DataFrame(data.groupby('housing_status')['fraud_bool'].mean()).sort_values(by = 'fraud_bool',ascending = False)
housing_rank


# ### <div class="alert alert-block alert-warning"> Chances of fraud - BA>BD>BC>BB>BF>BF>BE
# ### Do label encoding - replace housing categories by fraud rate

# In[71]:


housing_rank_dict = housing_rank.to_dict()['fraud_bool']
data['housing_status_encoded'] = data.apply(lambda row : housing_rank_dict[row['housing_status']],axis = 1)


# ###  <div class="alert alert-block alert-info">phone_home_valid, phone_mobile_valid analysis 

# In[72]:


data['phone_home_valid'].value_counts()  #nearly equal split


# In[73]:


data.groupby('phone_home_valid')['fraud_bool'].value_counts()


# In[74]:


#who dont give and fraud
823800/(8238+574685)


# In[75]:


#who give but fraud
279100/(2791+414286)


# ### <div class="alert alert-block alert-warning"> People who dont provide valid home number have slightly higher chances of fraud

# In[76]:


data['phone_mobile_valid'].value_counts()  #most of the people provide a valid phone number while opening account


# In[77]:


data.groupby('phone_mobile_valid')['fraud_bool'].value_counts()


# In[78]:


164800/(1648+108676)


# In[79]:


938100/(9381+880295)


# ### Whats the fraud proportion for people who dont give either contacts?
# 

# In[80]:


no_numbers = data[(data['phone_mobile_valid'] ==0) & (data['phone_home_valid'] ==0)] # 20% of customers provide neither
no_numbers['fraud_bool'].value_counts()


# In[81]:


61900/(619+21613)


# ### <div class="alert alert-block alert-warning"> The proportion of fraud increases form 1.4% -> 2.7% if both the numbers are not provided. Lets create a derived feature : no_number_provided

# In[82]:


data['no_number_provided'] = 1
data['no_number_provided'].iloc[data[(data['phone_mobile_valid'] ==0) & (data['phone_home_valid'] ==0)].index] = 0


# In[83]:


data.corr()['no_number_provided'].sort_values()


# ### <div class="alert alert-block alert-info"> bank_months_count analysis
# 

# ### For all fraud applications, see what's the average duration fraudsters tend to keep old accounts, before opening a new account?

# In[84]:


fraud_data.bank_months_count.mean()


# In[85]:


data.bank_months_count.mean()


# ###  <div class="alert alert-block alert-info"> has_other_cards analysis
# 

# In[86]:


data.has_other_cards.value_counts()  ## Most of the people dont have other cards from the banking company


# In[87]:


data.groupby('has_other_cards').fraud_bool.value_counts()


# In[88]:


1009800/(10098+766914)


# In[89]:


data


# ### <div class="alert alert-block alert-warning"> Higher chance of fraud occuring if they dont have any cards

# In[90]:


data


# ### <div class="alert alert-block alert-info"> proposed_credit_limit analysis

# In[91]:


sns.boxplot(data['proposed_credit_limit'])


# In[92]:


sns.boxplot(fraud_data['proposed_credit_limit'])


# In[93]:


data['address_counts'].describe()


# ### <div class="alert alert-block alert-info"> foreign_request analysis

# In[94]:


data['foreign_request'].value_counts()  #very few foreign requests


# In[95]:


data.groupby('foreign_request')['fraud_bool'].value_counts()
# 2.19% of foreign requests were fraud


# ### <div class="alert alert-block alert-info"> source analysis

# In[96]:


data.source.value_counts()  #Very few people use app


# In[97]:


data.groupby('source')['fraud_bool'].value_counts()
# 1% of the total fraud comes from app. 


# ### <div class="alert alert-block alert-info"> session_length_in_minutes analysis

# In[98]:


data.session_length_in_minutes.mean()


# In[99]:


data.groupby('source').session_length_in_minutes.mean()


# #### Apps have half session length than browsers which is expected

# In[101]:


data.groupby(['source','fraud_bool']).session_length_in_minutes.mean().reset_index()[data.groupby(['source','fraud_bool']).session_length_in_minutes.mean().reset_index()['fraud_bool'] ==1]


# ### <div class="alert alert-block alert-warning"> IN both the sources, fraudsters take more than usual time in their sessions. This may occur as they may take time to double check things, start vpn and other services to conceal their identities

# In[107]:


session_mean_dict = data.groupby('source').session_length_in_minutes.mean().to_dict()
session_mean_dict


# ### Derived feature - session_overtime : 1 if more than mean based on source = app/browser

# In[108]:


data['session_overtime'] = data.apply(lambda row: 1 if row['session_length_in_minutes'] > session_mean_dict[row['source']] else 0, axis = 1)


# ### <div class="alert alert-block alert-info"> device_os analysis
# 

# In[109]:


sns.countplot(data,x='device_os')


# ###  <div class="alert alert-block alert-warning"> Linux and other are most prevalent os

# #### Lets see which os is used in app?

# In[110]:


data[data.source == 'TELEAPP']['device_os'].value_counts()


# ### <div class="alert alert-block alert-warning">  Mobile users use others OS

# In[111]:


paymentplan_rank = pd.DataFrame(data.groupby('device_os')['fraud_bool'].mean()).sort_values(by = 'fraud_bool',ascending = False)
paymentplan_rank


# ### <div class="alert alert-block alert-warning"> Windows and mac face maximum fraud and surprisingly the most used OS - other and linux are rarely used by fraudsters

# In[112]:


data[data['customer_age']>50].groupby(['device_os','customer_age'])['fraud_bool'].mean()*100


# ### <div class="alert alert-block alert-warning"> Even though most people use Linux, Other os, frauds for elderly (who are more prone to identity frauds) take place in Windows and Mac users

# ### <div class="alert alert-block alert-info"> device_fraud_count analysis 

# In[113]:


data['device_fraud_count'].value_counts() # only 0. Better drop it


# ### We will drop this feature as no repeat fraudsters data available

# <br> 
# <br>
# 

# # <div class="alert alert-block alert-success"> Feature Engineering & Modeling

# In[114]:


categorical_columns = list(data.select_dtypes(['object']))
categorical_columns


# ### <div class="alert alert-block alert-info"> label encode payment type, device os

# In[115]:


paymentplan_rank = pd.DataFrame(data.groupby('payment_type')['fraud_bool'].mean()).sort_values(by = 'fraud_bool',ascending = False)
paymentplan_rank
paymentplan_rank_dict = paymentplan_rank.to_dict()['fraud_bool']
data['paymentplan_encoded'] = data.apply(lambda row : paymentplan_rank_dict[row['payment_type']],axis = 1)


# In[116]:


device_os_rank = pd.DataFrame(data.groupby('device_os')['fraud_bool'].mean()).sort_values(by = 'fraud_bool',ascending = False)
device_os_rank
device_os_rank_dict = device_os_rank.to_dict()['fraud_bool']
data['device_os_encoded'] = data.apply(lambda row : device_os_rank_dict[row['device_os']],axis = 1)


# ### encode source - 1 for app, 0 for browser

# In[117]:


data['source'] = np.where(data['source'] == 'INTERNET',1,0)


# ### <div class="alert alert-block alert-info"> Drop original categorical columns

# In[118]:


data.drop(['payment_type', 'employment_status', 'housing_status', 'source', 'device_os'],axis = 1, inplace = True)


# #### <div class="alert alert-block alert-warning"> 15 derived features made
# * <div class="alert alert-block alert-warning">  address_counts, address_stability,
# * <div class="alert alert-block alert-warning"> line_of_credit, income_category,
# * <div class="alert alert-block alert-warning"> velocity_4w_risky, velocity_6h_risky,velocity_24h_risky
# * <div class="alert alert-block alert-warning"> credit_risk_category, credit_risk_score_delta
# * <div class="alert alert-block alert-warning"> no_number_provided, session_overtime
# * <div class="alert alert-block alert-warning"> paymentplan_encoded, device_os_encoded,housing_status_encoded,employment_status_encoded

# ### There are 3 features which are still category (income_category/address_stability/credit_risk_category). We will use one hot encoding for them

# In[119]:


data = pd.get_dummies(data, columns=['income_category','address_stability','credit_risk_category'],drop_first =True)


# ### Convert all datatypes to float

# In[120]:


data= data.astype(float)


# ### Drop device fraud count as it only had 1 value

# In[121]:


data.drop('device_fraud_count',axis=1,inplace=True)


# ## <div class="alert alert-block alert-info"> Checking correlation to avoid multi collinearity

# In[122]:


"""
# calculate VIF for each feature
vif = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

# print VIF values
for i, v in enumerate(vif):
    print(f"Feature {i}: {v}")
"""


# In[123]:


data.drop('credit_risk_score_delta',axis=1,inplace=True)


# ## <div class="alert alert-block alert-warning"> Dropped highly corelated variables

# In[124]:


data.corr()


# In[125]:


data.corr().fraud_bool.sort_values()


# ### <div class="alert alert-block alert-warning"> Top 5 negatively  correlated features:
# * Keep_alive_session
# * Dob_distinct_emails_4w
# * Name_email_similarity
# * Has_other_cards
# * Phone_home_valid	
# 
# ###  <div class="alert alert-block alert-warning"> Top 5 positively correlated features:
# * Housing_status_encoded
# * Device_os_encoded
# * Credit_risk_score
# * Proposed_credit_limit
# * Customer_age	
# 

# ### <div class="alert alert-block alert-info"> Feature scaling/transformation
#     
# ### <div class="alert alert-block alert-warning"> Since I am going to implement tree based models primarily, I am not using any transformation method since they are scale invariant. Else, I'd have also checked if data distribution was linear or not besides scaling. If the distribution was not normal according to QQ Plot, I'd have tried different transformations like - logarithmic, reciprocal, exponential, boxcox transformations

# ###  <div class="alert alert-block alert-info"> One hot encoding/ Encoding - Done in the EDA step 
# 

# ###  <div class="alert alert-block alert-info"> Handling Imbalanced dataset - Currently I am not handling it because of time crunch. If time permits, i can try SMOTE or any undersampler 
# 

# ###  <div class="alert alert-block alert-info"> Modeling stage & Hyper parameter optimization
# 

# In[128]:


X = data.iloc[:,1:]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 42)


# In[129]:


xgb_classifier = XGBClassifier(max_depth = 3,
    min_child_weight=50, n_estimators=500,
    random_state = 42,nthread = -1, 
    learning_rate = 0.05,
    objective = 'binary:logistic',
    booster='gbtree', 
    importance_type = 'gain', base_score = 0.011, 
    reg_lambda = 1, subsample = 1, eval_metric = 'aucpr')

xgb_classifier.fit(X_train,y_train)
y_pred = xgb_classifier.predict(X_test)


# In[130]:


cm_xgboost = confusion_matrix(y_pred,y_test)


# In[131]:


print(cm_xgboost)


# ### ROC AUC curve

# In[132]:


roc_auc = roc_auc_score(y_test, xgb_classifier.predict_proba(X_test)[:, 1])


# In[133]:


roc_auc


# ### <div class="alert alert-block alert-warning"> The score increased from 89-90%. Not much improvement. But the features added were useful in terms of business 

# In[144]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, xgb_classifier.predict_proba(X_test)[:, 1])

plt.figure(figsize = (3,3))
plt.plot(fpr, tpr,label = 'XG boost Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC curve for XG Boost model")
plt.plot([0, 1], [0, 1], linestyle='--', label='Average')
plt.grid('on')
plt.legend()


# ### Feature importance

# In[135]:


impt  = pd.DataFrame(xgb_classifier.feature_importances_*100)
impt['columns'] = ""
impt['columns']= data.columns[1:]


# In[150]:


impt.columns = ['feature importance (%)','feature']
impt = impt.sort_values(by ='feature importance (%)',ascending=False)
impt


# ### <div class="alert alert-block alert-success"> Model Inference - Decision rules

# In[160]:


y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]
y_pred_proba_df = pd.DataFrame(y_pred_proba,columns = ['prob1'])


# In[165]:


X_test_df = X_test.reset_index(drop=True)
X_test_df['xgb_score'] = y_pred_proba_df['prob1']
X_test_df


# In[200]:


# add scores to test dataset
X_test_df = X_test.reset_index(drop=True)
X_test_df['xgb score'] = y_pred_proba_df['prob1']

# decision tree

dt_clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=30,
                                min_samples_leaf=50, min_weight_fraction_leaf=0.0, max_features=None,
                                random_state=42, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None,
                                ccp_alpha=0.0, )

dt_clf.fit(X_test_df,y_test)


# In[204]:


42,1,8,7,10,25,6,5,11,3,13
print(X_test_df.columns[42])
print(X_test_df.columns[1])
print(X_test_df.columns[8])
print(X_test_df.columns[7])
print(X_test_df.columns[10])
print(X_test_df.columns[25])
print(X_test_df.columns[6])
print(X_test_df.columns[5])
print(X_test_df.columns[11])
print(X_test_df.columns[3])
print(X_test_df.columns[13])


# In[249]:


print(X_test_df.columns[25])


# In[201]:


#tree plot
plt.figure(figsize=(40,40))
tree.plot_tree(dt_clf,filled=True,fontsize=6)
plt.show()


# ### For each leaf node, left value is no of actual non frauds, right value is no of actual frauds. 
