# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:30:27 2020


author: leo_f

I used some guide from Ken Jee https://www.kaggle.com/kenjee/titanic-project-example


I've already clean the data and did some feature engineering in titanic_data_cleaning.py
In this script I did all the prepoccesing stuff, like scaling, normalizing(with log), and get_dummies of the categorical
vairables.

Then I built a simple base modeles, of Logistic Regression, KNearNeighbors, Desicion Trees, Random Forest, 
Gradient Boosting, and Support Vector Machine.

After that I made a ensambled with voting with all this models and I get a score of 0.770
After that I use param_grid with GridSearchCV to star tunning the models.
I got some improvements in most of the models.

At the end I tried with:
Only GBoosting
Voting with all models which result were more than 0.8 (in cross Validation) using hard voting
Voting with all models which result were more than 0.8 (in cross Validation) using soft voting
Only with SVC where I got the best result

Best results SVC tunned Train 0.829 Test 0.7799
Worst result GB Tunned Train 0.844 and Test 0.744 (so overfitted)
Could be because I scaled the data, and I normalized it, so all this maked a benefit to SVC and a penalized for GB and RF



"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

titanic = pd.read_csv('C:\\Users\\leo_f\\Documents\\ML\\Kaggle Competitions\\Titanic\\titanic_df_clean.csv', index_col=0)

# Preproccesing data
# Drop, name, mr, ticket, cabin, surname
titanic.drop(['name','mr','ticket','cabin','fare', 'surname','age'], inplace= True, axis=1)
# Create bins for new_age and fare per person (not yet)


# Normalize fare, parch, sibsp, fare_per_person
 # Im gonna do it with log

#titanic['fare_norm'] = np.log(titanic.fare+1)
#titanic['parch_norm'] = np.log(titanic.parch+1)
#titanic['sibsp_norm'] = np.log(titanic.sibsp+1)
titanic['fare_per_person_norm'] = np.log(titanic.fare_per_person+1)

#titanic.fare_norm.hist()
#titanic.parch_norm.hist() 
#titanic.sibsp_norm.hist()
#titanic.fare_per_person_norm.hist()
# Im not gonnna use parch norm and sibsp norm.

# Create dummies variables for sex, pclass, new_age, embarked, fare per person
# Creat Pclass as str
titanic.pclass = titanic.pclass.astype('str')
titanic_dummies = pd.get_dummies(titanic)

# Standarize with StandarScaler, 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
titanic_dummies_scaled = titanic_dummies.copy()
titanic_dummies_scaled[['new_age','fare_per_person_norm','family_size','sibsp','parch']]= scaler.fit_transform(titanic_dummies[['new_age','fare_per_person_norm','family_size','sibsp','parch']])

# Separete X, y, X_final_test

X = titanic_dummies_scaled.drop(['survived','passengerid','fare_per_person'],axis=1).loc[:890,:]
X_final_test = titanic_dummies_scaled.drop(['survived','passengerid','fare_per_person'],axis=1).loc[891:,:]
y = titanic_dummies_scaled.loc[:890,'survived']

# Model Buliding (we are gonna do the basic without tunning)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score


# Logistic Regression
lr = LogisticRegression()
cv_lr = cross_val_score(lr,X,y,cv=5,scoring='accuracy')
print('Logistic Regression')
print(cv_lr)
print(cv_lr.mean())

# Knear Neighbors
knn = KNeighborsClassifier()
cv_knn = cross_val_score(knn,X,y,cv=5,scoring='accuracy')
print('KNN')
print(cv_knn)
print(cv_knn.mean())

# Desicion Tree
dt = DecisionTreeClassifier(random_state=1)
cv_dt = cross_val_score(dt,X,y,cv=5,scoring='accuracy')
print('Decision Tree')
print(cv_dt)
print(cv_dt.mean())

# Random Forest
rf = RandomForestClassifier(random_state=1)
cv_rf = cross_val_score(rf,X,y,cv=5,scoring='accuracy')
print('Random Forest')
print(cv_rf)
print(cv_rf.mean())

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=1)
cv_gb = cross_val_score(gb,X,y,cv=5,scoring='accuracy')
print('Gradient Boosting')
print(cv_gb)
print(cv_gb.mean())

# Suport Vector Classifier
svm = SVC(probability=(True))
cv_svm = cross_val_score(svm,X,y,cv=5,scoring='accuracy')
print('Support Vector Machine')
print(cv_svm)
print(cv_svm.mean())

# We can see Scaling makes really good for Support Vector machines 

# Data frame with base scores
basescore= [cv_lr.mean(),cv_knn.mean(),cv_dt.mean(),cv_rf.mean(),cv_gb.mean(),cv_svm.mean()]
BaseScores = pd.DataFrame({'Model':['Logistic Regression','KNN', 'Decision Tree','Random Forest','Gradient Boostinng','Support Vector Machine'],'Base Score': basescore})

# Voting Classifier
#Voting classifier takes all of the inputs and averages the results. 
#For a "hard" voting classifier each classifier gets 1 vote "yes" or "no" and 
#the result is just a popular vote. For this, you generally want odd numbers
#A "soft" classifier averages the confidence of each of the models. 
#If a the average confidence is > 50% that it is a 1 it will be counted as such

voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gb',gb),('svm',svm),('dt',dt)], voting = 'soft') 
cv = cross_val_score(voting_clf,X,y,cv=5,scoring='accuracy')
print('Voting Classifier')
print(cv)
print(cv.mean())

# Feature Importance
gb.fit(X,y)
plt.barh(X.columns,gb.feature_importances_)

# Try a submision
voting_clf.fit(X, y)
y_pred_final = voting_clf.predict(X_final_test).astype(int)
submision_base = pd.DataFrame({'PassengerId': titanic.loc[891:,'passengerid'],'Survived':y_pred_final})
#submision_base.to_csv('titanic_eighth_sub.csv', index=False)

# Parameter Tunning (last thing Im gonna do in this script)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Logistic regression

param_grid = { 'C': np.logspace(-4,4,20),
              'max_iter': [2000],
              'penalty':['l1','l2'],
              'solver': ['liblinear']},
              

mt_lr = GridSearchCV(lr, param_grid=param_grid,cv=5, n_jobs=-1)
best_mt_lr = mt_lr.fit(X,y)
print('Logistic Regression')
print('best score: ', best_mt_lr.best_score_)
print('Best params:', best_mt_lr.best_params_ )
    
    

# KNN
param_grid = { 'n_neighbors': list(range(2,30,2)),
              'weights':['distance','uniform'],
             'algorithm':['auto', 'ball_tree', 'kd_tree'],
             'p':[1,2]}
              

mt_knn = GridSearchCV(knn, param_grid=param_grid,cv=5, n_jobs=-1)
best_mt_knn = mt_knn.fit(X,y)
print('KNN')
print('best score: ', best_mt_knn.best_score_)
print('Best params:', best_mt_knn.best_params_ )


# Random Forest
# Beacuse there are too many options we are gonna do a random seacrch

param_grid = {'n_estimators': [100,500,1000], 
                                  'bootstrap': [True,False],
                                  'max_depth': [3,5,10,20,50,75,100,None],
                                  'max_features': ['auto','sqrt'],
                                  'min_samples_leaf': [1,2,4,10],
                                  'min_samples_split': [2,5,10]}

mt_rf =  RandomizedSearchCV(rf, param_distributions=param_grid,cv=5,n_iter=100, n_jobs=-1)
best_mt_rf = mt_rf.fit(X,y)
print('Random Forest')
print('best score: ', best_mt_rf.best_score_)
print('Best params:', best_mt_rf.best_params_ )



# Gradient Boosting
param_grid = {'n_estimators': [50,100,500,1000], 
                                  'learning_rate': [0.01,0.1,0.2,0.5,0.6,0.9,1],
                                  'max_depth': [3,5,10,20,50,75,100,None],
                                  'max_features': ['auto','sqrt',None],
                                  'min_samples_leaf': [1,2,4,10],
                                  'min_samples_split': [2,5,10]}

mt_gb =  RandomizedSearchCV(gb, param_distributions=param_grid,cv=5,n_iter=100, n_jobs=-1)
best_mt_gb = mt_gb.fit(X,y)
print('Gradient Boosting')
print('best score: ', best_mt_gb.best_score_)
print('Best params:', best_mt_gb.best_params_ )




# SVC

param_grid = [{ 'kernel': ['linear'],
              'C':[0.1,1,10,100,1000]},
             {'kernel':['rbf'],
             'C':[0.1,1,10,100,1000],
             'gamma':[0.1,0.5,1,5,10]}]


# Took a lot of time(5min)
# best param is : C:1, gamma:0.1, kernel:'rbf'

mt_svm = GridSearchCV(svm, param_grid=param_grid,cv=5, n_jobs=-1)
best_mt_svm = mt_svm.fit(X,y)
print('Support Vector Machine')
print('best score: ', best_mt_svm.best_score_)
print('Best params:', best_mt_svm.best_params_ )


# table with model tunning
model_tunning = [best_mt_lr.best_score_,best_mt_knn.best_score_,'NaN',best_mt_rf.best_score_,best_mt_gb.best_score_,best_mt_svm.best_score_]

BaseScores['Model tunning'] = model_tunning

#Prepare the best
best_lr = best_mt_lr.best_estimator_
best_knn = best_mt_knn.best_estimator_
best_dt = dt
best_rf = best_mt_rf.best_estimator_
best_gb = best_mt_gb.best_estimator_
best_svm = best_mt_svm.best_estimator_



# Try best score (GB)   Train 0.844   Test 0.74880   (supper overfitting)
best_gb.fit(X, y)
y_pred_final_gb = best_gb.predict(X_final_test).astype(int)
submision_gb = pd.DataFrame({'PassengerId': titanic.loc[891:,'passengerid'],'Survived':y_pred_final_gb})
submision_gb.to_csv('titanic_ninth_sub.csv', index=False)

# Voting with score > 0.8 hard   Train 0.8395   Test 0.77033

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('gb',best_gb),('svm',best_svm)], voting = 'hard') 
cv = cross_val_score(voting_clf_hard,X,y,cv=5,scoring='accuracy')
print('Voting Classifier_hard')
print(cv)
print(cv.mean())

voting_clf_hard.fit(X, y)
y_pred_final_hard = voting_clf_hard.predict(X_final_test).astype(int)
submision_hard = pd.DataFrame({'PassengerId': titanic.loc[891:,'passengerid'],'Survived':y_pred_final_hard})
submision_hard.to_csv('titanic_tenth_sub.csv', index=False)


# Voting with score > 0.8 soft   Train 0.8327  Test 0.77272
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('gb',best_gb),('svm',best_svm)], voting = 'soft') 
cv = cross_val_score(voting_clf_soft,X,y,cv=5,scoring='accuracy')
print('Voting Classifier_soft')
print(cv)
print(cv.mean())

voting_clf_soft.fit(X, y)
y_pred_final_soft = voting_clf_soft.predict(X_final_test).astype(int)
submision_soft = pd.DataFrame({'PassengerId': titanic.loc[891:,'passengerid'],'Survived':y_pred_final_soft})
submision_soft.to_csv('titanic_eleventh_sub.csv', index=False)

# Mix of SVC,gb   Train 0.8338  Test 0.77751
voting_clf_mix = VotingClassifier(estimators = [('gb',best_gb),('svm',best_svm)], voting = 'soft') 
cv = cross_val_score(voting_clf_mix,X,y,cv=5,scoring='accuracy')
print('Voting Classifier_mix')
print(cv)
print(cv.mean())

voting_clf_mix.fit(X, y)
y_pred_final_mix = voting_clf_mix.predict(X_final_test).astype(int)
submision_mix = pd.DataFrame({'PassengerId': titanic.loc[891:,'passengerid'],'Survived':y_pred_final_mix})
submision_mix.to_csv('titanic_twelveth_sub.csv', index=False)

# only SVC    Train 0.77990 

best_svm.fit(X, y)
y_pred_final_svm = best_svm.predict(X_final_test).astype(int)
submision_svm = pd.DataFrame({'PassengerId': titanic.loc[891:,'passengerid'],'Survived':y_pred_final_svm})
submision_svm.to_csv('titanic_Thirteenth_sub.csv', index=False)

## Best Score SVC only, last step its gonna make a better parameter tunning, maybe the scale stuff make it better

