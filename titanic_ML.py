# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 16:50:48 2019

@author: kzawi
"""

from __future__ import print_function
import os
data_path = ['data']

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore', module='sklearn')

from sklearn.preprocessing import MinMaxScaler

filepath = os.sep.join(data_path + ['train.csv'])
train_data = pd.read_csv(filepath)

#print(train_data)

#analiza brakujących informacji
missing_inf_total = train_data.isnull().sum().sort_values(ascending=True)
#print(missing_inf_total)
#print('\n\n')
missing_inf_percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=True)
#print(missing_inf_percent)


filepath = os.sep.join(data_path + ['test.csv'])
test_data = pd.read_csv(filepath)

#print(test_data)

missing_inf_total = test_data.isnull().sum().sort_values(ascending=True)
#print(missing_inf_total)
#print('\n\n')
missing_inf_percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=True)
#print(missing_inf_percent)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    
    data["Embarked"] = data["Embarked"].fillna("S")
    
    le = LabelEncoder()
    ohc = OneHotEncoder()
    
    data_ohc = data.copy()
    
    ohc_col = data.loc[:, ['Embarked', 'Pclass']]
    #print(ohc_col)
    
    for col in ohc_col:
        #print(data_ohc[col])
        #print(col)
        # Integer encode the string categories
        
        dat = le.fit_transform(data_ohc[col]).astype(np.int)
    
        # Remove the original column from the dataframe
        data_ohc = data_ohc.drop(col, axis=1)

        # One hot encode the data--this returns a sparse array
        new_dat = ohc.fit_transform(dat.reshape(-1,1))

        # Create unique column names
        n_cols = new_dat.shape[1]
        col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]
        
        # Create the new dataframe
        new_df = pd.DataFrame(new_dat.toarray(), 
                              index=data_ohc.index, 
                              columns=col_names)
        
        # Append the new data to the dataframe
        data_ohc = pd.concat([data_ohc, new_df], axis=1)
        
        data = data_ohc
        
    msc = MinMaxScaler()
    
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    data = pd.DataFrame(msc.fit_transform(data),  # this is an np.array, not a dataframe.
                   columns=data.columns)
    
    return data

#train_data = clean_data(train_data)
#print(train_data)

import matplotlib.pyplot as plt
import seaborn as sns

print(train_data)

print_data_male = train_data.loc[train_data['Sex'] == 'male']
print_data_female = train_data.loc[train_data['Sex'] == 'female']
print_data_survived = train_data.loc[train_data['Survived'] == 1]
print_data_died = train_data.loc[train_data['Survived'] == 0]

train_data.Survived.value_counts(normalize=True).plot(kind="bar", alpha = 1)
plt.show()

train_data.groupby(['Survived','Sex']).size().unstack().plot(kind='bar',stacked=True)
plt.show()

print_data_male.Survived.value_counts(normalize=True).plot(kind="bar", alpha = 1)
plt.show()

print_data_female.Survived.value_counts(normalize=True).sort_values(ascending=True).plot(kind="bar", alpha = 1)
plt.show()

train_data[['Age']].plot(kind='hist',bins=[0,10,20,30,40,50,60,70,80,90,100],rwidth=0.8)
plt.show()

print_data_survived[['Age']].plot(kind='hist',bins=[0,10,20,30,40,50,60,70,80,90,100],rwidth=0.8)
plt.show()

print_data_died[['Age']].plot(kind='hist',bins=[0,10,20,30,40,50,60,70,80,90,100],rwidth=0.8)
plt.show()

print_data_age = train_data.copy()
for index, x in print_data_age.iterrows():
    #x['Age'] = x['Age']//10
    print_data_age.at[index, 'Age'] = x['Age']//10
    
#print(print_data_age)
print_data_age.groupby(['Survived','Age']).size().unstack().plot(kind='bar',stacked=True)
plt.show()


ax = sns.countplot(x="Pclass", data=print_data_age)
g = sns.catplot(x="class", hue="who", col="survived", data=titanic, kind="count", height=4, aspect=.7);

