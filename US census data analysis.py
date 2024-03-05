#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# readinag given csv file
# and creating dataframe
df = pd.read_csv("adult.data")
# storing this dataframe in a csv file
df.to_csv('adult.data',index = None)
df


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df.head()


# In[8]:


df.dtypes


# In[4]:


# Removing the initial white spaces from dataframe
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=df[i].map(str.strip)


# In[5]:


df.describe()


# 1.Average age of people 38.
# 2.Average years for education is 10 years. on an average rate sslc has been completed.
# 3.people are working for Averagly 8 hours per day.

# # Remove data with missing values and Remove outliers

# In[7]:


# checking any negative value in numerical data's
print(len(df.loc[(df.Fnlwgt < 0)]))
print(len(df.loc[(df['Education-num'] < 0),:]))
print(len(df.loc[(df['capital-gain'] < 0),:]))
print(len(df.loc[(df['capital-loss']< 0),:]))
print(len(df.loc[(df['hours-per-week'] < 0),:]))


# In[6]:


#text file has '?' for missing values, so changing into unknown
df.replace(to_replace = '?', value ='unknown',inplace = True)


# In[7]:


# checking the null values
check_nan = df.isnull().values.any()
check_nan


# In[10]:


# b) Counting rows that have missing values somewhere:
sum([True for idx,row in df.iterrows() if any(row.isnull())])


# In[11]:


# trimming the outliers of Age data
upperlimit_Age= df['Age'].quantile(0.99)
lowerlimit_Age = df['Age'].quantile(0.01)
print('upperlimit_Age:', upperlimit_Age)
print('lowerlimit_Age:', lowerlimit_Age)
# trimming the outliers of hours-per-week data
df1 = df.loc[(df['Age']<= upperlimit_Age) & (df['Age']>=lowerlimit_Age)]
print('before removing outliers:', len(df))
print('after removing outliers:', len(df1))
print('outlier data:',len(df)-len(df1))
sns.boxplot(df['Age'])


# In[12]:


# When compared to original boxplot of age, after removing outiers data below boxplot looks normally distributed data.
sns.boxplot(df1['Age'])


# In[16]:


# trimming the outliers of 	Education-num data
upperlimit_Education_num = df1['Education-num'].quantile(0.99)
lowerlimit_Education_num = df1['Education-num'].quantile(0.01)
print('upperlimit_Education-num:', upperlimit_Education_num)
print('lowerlimit_Education-num:', lowerlimit_Education_num)
# trimming the outliers of Education-num data
df2 = df1.loc[(df1['Education-num']<= upperlimit_Education_num) & (df1['Education-num']>=lowerlimit_Education_num)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(df2))
print('outlier data:',len(df1)-len(df2))
sns.boxplot(df['Education-num'])


# In[17]:


# When compared to original boxplot of Education-num, after removing outiers data below boxplot looks normally distributed data.
sns.boxplot(df2['Education-num'])


# In[18]:


# trimming the outliers of hours-per-week data
upperlimit_hours = df2['hours-per-week'].quantile(0.99)
lowerlimit_hours = df2['hours-per-week'].quantile(0.01)
print('upperlimit_hours:', upperlimit_hours)
print('lowerlimit_hours:', lowerlimit_hours)
# trimming the outliers of hours-per-week data
df3 = df2.loc[(df2['hours-per-week']<= upperlimit_hours) & (df2['hours-per-week']>=lowerlimit_hours)]
print('before removing outliers:', len(df2))
print('after removing outliers:', len(df3))
print('outlier data:',len(df2)-len(df3))


# In[24]:





# we didn't remove the outliers for Captialgain,loss,fnlwgt columns , since the lot of data loss.

# # ANALYSING relation between MALE AND FEMALE 

# In[20]:


df['sex'].value_counts()


# In[21]:


Age=[df[df['sex']=='Female']['Age'].mean(),df[df['sex']=='Male']['Age'].mean()]
working_hours=[df[df['sex']=='Female']['hours-per-week'].mean(),df[df['sex']=='Male']['hours-per-week'].mean()]
Education=[df[df['sex']=='Female']['Education-num'].mean(),df[df['sex']=='Male']['Education-num'].mean()]

malefemale= pd.DataFrame(list(zip(Age,working_hours,Education)),
               columns =['Age', 'working_hours','Education'],index=['Female','Male'])
malefemale


# Females are working 7 hours per day and men are working for 8 hours per day approximately
# Both female and male average spent time for education is almost same all were mostly completed 10 level education.

# In[45]:


fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])


x = ['Female','Male']
y = malefemale['working_hours']
a = malefemale['Education']
z = malefemale['Age']
plt.bar(x,z,width=0.4,color = 'b', label ='AVG-Age')
plt.bar(x,y,width=0.17,color = 'r', label ='AVG-working_hours')
plt.bar(x,a,width=0.12,color = 'y', label ='AVG-Education')
plt.legend()
plt.show()


# In[64]:


a1=df.loc[(df['sex']== 'Female') & (df['Annual-incomerange'] == '>50K')].count()['Age']
a2=df.loc[(df['sex']== 'Male') & (df['Annual-incomerange'] == '>50K')].count()['Age']
a3=[a1,a2]
plt.pie(a3,autopct = '%0.1f%%',colors =['y','green'],explode = [0.05,0], shadow= True,labels=['Female who earns >50K','Male who earns >50K'])
plt.show()
print('Only One-fifth of female earning more than 50K')


# In[130]:


s1=df.loc[(df['sex']=='Female')& (df['capital-gain'] != 0)].count()['Age']
s2=df.loc[(df['sex']=='Male')& (df['capital-gain'] != 0)].count()['Age']
print(f'{s1} Female who have capital gain\n{s2} Male who have capital gain')


# # Analysis based on workclass

# In[116]:


d=df['Workclass'].value_counts()
d1=pd.DataFrame(d)
plt.pie(d1['count'],labels=d1.index,autopct = '%0.1f%%',radius=2)
print('Approximately 70% of people are working for private')


# In[140]:


c1=df.loc[(df['Workclass']== 'Private') & (df['Annual-incomerange'] == '>50K')].count()['Age']
c2=df.loc[(df['Workclass']== 'Private') & (df['Annual-incomerange'] == '<=50K')].count()['Age']
c3=df['Workclass'].count()
rich_private_workers = round(c1/c3*100)
poor_private_workers = round(c2/c3*100)
print(f'percent_rich_private_workers:{rich_private_workers}%')
print(f'percent_poor_private_workers:{poor_private_workers}%')


# In[141]:


df['Workclass'].value_counts()


# In[158]:


print('Most of the rich private employees completed thier bachelors and High_school graduation degree')
df.loc[(df['Annual-incomerange']=='>50K')&((df['Workclass']=='Private')),'Education'].value_counts()


# In[156]:


print('Most of the rich government employees completed thier bachelors and masters degree')
df.loc[(df['Annual-incomerange']=='>50K')&((df['Workclass']=='Local-gov') | (df['Workclass']=='State-gov') | (df['Workclass']=='Federal-gov')),'Education'].value_counts()


# # Relation between age and income

# In[194]:


print('Most of the rich people lies in the age of 30 to 55')
w=df.loc[(df['Annual-incomerange']=='>50K'),'Age'].value_counts()
sns.scatterplot(w)


# # Establish the importance of the weekly working hours on earning potential

# In[195]:


# Establish the importance of the weekly working hours on earning potential
print('People who work for more than 40 hours per week are getting more than 50k earning annually')

sns.barplot(x=df['Annual-incomerange'], y=df['hours-per-week'])


# # Analysis based on marital status

# In[197]:


df['Marital-status'].value_counts()


# In[241]:


married=df.loc[(df['Marital-status']=='Married-civ-spouse') | (df['Marital-status']=='Married-spouse-absent') | (df['Marital-status']== 'Married-AF-spouse')]
single=df.loc[(df['Marital-status']=='Never-married') | (df['Marital-status']=='Divorced') | (df['Marital-status']== 'Separated') | (df['Marital-status']== 'Widowed')]

a=married.loc[married['Annual-incomerange']=='>50K'].count()['Age']
b=single.loc[single['Annual-incomerange']=='>50K'].count()['Age']
count_married =married['Age'].count()
count_single =single['Age'].count()
print(f'Out of {count_married} people {a} are rich\nOut of {count_single} people {b} are rich\nPeople who are married were wealthier than singles')

ma_percent=[a,count_married-a]
si_percent=[b,count_single-b]
plt.figure()
plt.pie(ma_percent,radius = 1.3,center=(1,0),labels=['rich-married','poor-married'],autopct = '%0.1f%%')
plt.pie(si_percent,radius = 1.3,center=(4,0),labels=['rich-single','poor-single'],autopct = '%0.1f%%',startangle=90)


# # Analysis based on education number

# In[264]:


print('''1.Most of the rich people whose years of education lies between 10 to 16 years''')
plt.figure(figsize=[15,10])
sns.scatterplot(x=df['Education-num'],y=df['Age'],hue=df['Annual-incomerange'])


# In[313]:


# Creating df having education num wise income details
df_corr_edu_income = df[['Annual-incomerange','Education-num']].groupby('Annual-incomerange').value_counts().unstack().fillna(0)
df_corr_edu_income = df_corr_edu_income.T
print(pd.DataFrame(df_corr_edu_income))
fig = plt.figure()
x=df_corr_edu_income.index
y=df_corr_edu_income['<=50K']
z=df_corr_edu_income['>50K']

ax= fig.add_axes([0.5,0.5,1,1])
ax.plot(x,y,label='Poor_people',marker = '.')
ax.plot(x,z,label='Rich_people',marker = '*')
ax.set_ylabel('no of people',fontsize=12)
ax.set_xlabel('no of years for education',fontsize=12)
plt.legend()


# # Analysis based on native country

# In[20]:


a=df.loc[df['native-country']=='United-States'].count()['Age']
b=df.loc[(df['native-country']!='United-States') & (df['native-country']!='unknown') ].count()['Age']
c=df.loc[df['native-country']=='unknown'].count()['Age']
x=[a,b,c]
plt.pie(x,labels=['United-States','Other-countries','unknown'],autopct = '%0.1f%%')


# In[21]:


df.head()


# In[52]:


rich_americans = df.loc[(df['native-country'] == 'United-States') & (df['Annual-incomerange']=='>50K')].count()['Age']
rich_indian = df.loc[(df['native-country'] == 'India') & (df['Annual-incomerange']=='>50K')].count()['Age']
Totalamericans=df.loc[df['native-country'] == 'United-States'].count()['Age']
Totalindian=df.loc[df['native-country'] == 'India'].count()['Age']

richUS = round(rich_americans/Totalamericans*100)
richIND = round(rich_indian/Totalindian*100)
z=[richUS,richIND]
fig = plt.figure()
ax= fig.add_axes([0,0,1,1])
plt.barh(y=['percent_of_Richamericans','percent_of_Richindians'],width=z)


# In[55]:


american_gain=df.loc[df['native-country'] == 'United-States'].sum()['capital-gain']
american_loss=df.loc[df['native-country'] == 'United-States'].sum()['capital-loss']
print(american_gain,american_loss)
print('Total capital gain of americans is more than the total capital loss of americans')


# In[ ]:




