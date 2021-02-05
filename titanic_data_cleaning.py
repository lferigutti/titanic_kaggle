# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:46:34 2020

@author: leo_f


I learned that is a good idea to clean the data and do most of the engineering with the train and test altoghether

"""

import pandas as pd
import numpy as np

dftr = pd.read_csv('data_sets/titanic/train.csv')
dfte = pd.read_csv('data_sets/titanic/test.csv')
df = pd.concat([dftr,dfte], ignore_index=True)

# make the columns name lowercase
df.columns = df.columns.str.lower()

# Create a column with only the surname of the people

df['surname'] = df.name.apply(lambda x: x.split(',')[0])

# Create a column with the tittle of the people

df['mr']=df.name.apply(lambda x: x.split(',')[1].split('.')[0])

df.mr = df.apply(lambda x: x['mr'].replace(' ',''), axis=1)

# Create a column with Family size
df['family_size'] = df.sibsp + df.parch + 1

# Join dr, cond, et, as MR or Mrs
plp_title = list(df.mr.unique())

# Exclude Mr, Mrs, Miss and Master
plp_title_adult = plp_title[4:]

df.mr = df.apply(lambda x: 'Mr' if ((x.mr in plp_title_adult) & (x.sex == 'male')) else x.mr, axis = 1)
df.mr = df.apply(lambda x: 'Mrs' if ((x.mr in plp_title_adult) & (x.sex == 'female')) else x.mr, axis = 1)


# Create a funcion that put diffetnt values to age, depending if the person is master mister or something simiar
age_mean_adult =df.age[df.age>=13].mean()
age_mean_child = df[df.age<13].age.mean()
age_mean_total = df.age.mean()


df['new_age'] =df.apply(lambda x: age_mean_adult if ((pd.isnull(x.age)) & ((x.mr == 'Mrs')|(x.mr=='Mr'))) else x.age, axis = 1)
df['new_age']= df.apply(lambda x: age_mean_child if ((pd.isnull(x.age)) & (x.mr == 'Master')) else x.new_age, axis=1)
df['new_age']= df.apply(lambda x: age_mean_adult if (pd.isnull(x.age)) & (x.family_size==0) else x.new_age, axis=1)
df['new_age']= df.apply(lambda x: age_mean_adult if (pd.isnull(x.age)) & (x.parch==0) else x.new_age, axis=1)
df['new_age']= df.apply(lambda x: age_mean_child if (pd.isnull(x.new_age)) & (x.sibsp >= 2) else x.new_age, axis=1)
df['new_age']= df.apply(lambda x: age_mean_total if (pd.isnull(x.new_age)) else x.new_age, axis=1)


print (df.new_age.isnull().sum())

# Create a new column that show if you are a child or not
df['chilid'] = df.new_age.apply(lambda x: 1 if x <= 13 else 0)

# Create a new column fare/person
df.fare = df.fare.fillna(0)

df['fare_per_person'] = df.apply(lambda x: (x.fare)/(x.family_size) if (x.fare !=0) & (x.family_size !=0) 
                                 else (x.fare if (x.fare !=0) & (x.family_size ==0) else 0), axis = 1)


# embarked has two NaN values

df['embarked']= df.embarked.fillna('S')

# Do Categoies features with label encored (sex and embarked) (NO, THIS IS PART OF PRE PROCCESING)

# Create a csv file to analize in Jupiter notbook
df.to_csv('titanic_df_clean.csv')

