# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 14:21:12 2016

@author: Trace
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
try:
    from sklearn.model_selection import cross_val_predict
except ImportError:
    from sklearn.cross_validation import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import neighbors
import os

# Set the path for the working directory
wd = 'C:\\Users\Trace\Documents\MS\CPSC-605 Data Mining & Analysis\Project\Census Data 1'

# Change the working directory to the folder containing the census data file
os.chdir(wd)

# Read the CSV file
data1 = pd.read_csv('census-income-train.csv', delimiter=',')
data2 = pd.read_csv('census-income-test.csv', delimiter=',')
data_both = [data1, data2]
data = pd.concat(data_both)

# This text file has no column names, so they must be created
#colnames = ['Age', 'ClassOfWorker', 'IndustryCode', 'OccupationCode', 'AdjustedGrossIncome', 'Education', 'WagePerHour', 'EnrolledInEduInstLastWeek', 'MaritalStatus', 'MajorIndustryCode', 'MajorOccupationCode', 'Mace', 'HispanicOrigin', 'Sex', 'MemberOfLaborUnion', 'ReasonForUnemployment', 'FullOrPartTimeEmploymentStat', 'CapitalGains', 'CapitalLosses', 'DividendsFromStocks', 'FederalIncomeTaxLiability', 'TaxFilerStatus', 'RegionOfPreviousResidence', 'StateOfPreviousResidence', 'DetailedHouseholdAndFamilyStat', 'DetailedHouseholdSummaryInHousehold', 'InstanceWeight', 'MigrationCode-ChangeInMSA', 'MigrationCode-ChangeInReg', 'MigrationCode-MoveWithinReg','LiveInThisHouse1YearAgo', 'Migration prev res in sunbelt', 'Num Persons worked for employer', 'Family members under 18', 'Total person earnings', 'Country of Birth Father', 'Country of Birth Mother', 'Country of Birth', 'Citizenship', 'Total Person Income', 'Own Business or Self-Employed', 'Taxable Income Amount', 'Fill Inc Questionnaire for Veterans Admin', 'Veterans Benefits', 'Weeks Worked in Year']

# Set the column names to the ones created
#data.columns = colnames

# Drop the Education and Fnlwgt columns. Education is captured in the Education-Years column
data = data.drop(['HispanicOrigin', 'OccupyHouse1Year', 'Migration-Region', 'Migration-WithinRegion', 'Migration-MSA', 'Migration-Sunbelt', 'VeteransAdmin', 'LaborUnion', 'UnemploymentReason', 'IndustryCode', 'OccupationCode', 'DetailedHousehold', 'InstanceWeight', 'PreviousResidenceRegion', 'PreviousResidenceState'], axis=1)

# Replace ? with Unknown
data.replace(to_replace='?', value='Unknown')
data.replace(to_replace='Not in universe', value='Unknown')

# Fix the marital status values
# Set Married-civ-spouse and Married-AF-spouse to simply married
data.MaritalStatus = data.MaritalStatus.replace(to_replace=' Married-civilian spouse present', value='Married')
data.MaritalStatus = data.MaritalStatus.replace(to_replace=' Married-A F spouse present', value='Married')

data.Citizenship = data.Citizenship.replace(to_replace=' Native- Born in the United States', value='Native')
data.Citizenship = data.Citizenship.replace(to_replace=' Foreign born- Not a citizen of U S ', value='Foreign')
data.Citizenship = data.Citizenship.replace(to_replace=' Foreign born- U S citizen by naturalization', value='Naturalized')

# Set Married-spouse-absent to Separated
data.MaritalStatus = data.MaritalStatus.replace(to_replace='Married-spouse-absent', value='Separated')

# Convert the Income column to 1 for <50K and 0 for <=50K
data['Income'] = data['Income']==' 50000+.'
data.Income = data.Income * 1

# Convert the NativeCountry column to 1 for United-States and 0 for all others
#data['NativeCountry'] = data['NativeCountry']=='United-States'
#data.Income = data.Income * 1

#hours = np.array(data.HoursPerWeek.values)
#bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#digitized = np.digitize(hours, bins)
#data.HoursPerWeek = digitized

# Convert the remaining attributes to numeric classes
def process(df) :
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.WorkerClass = le.fit_transform(processed_df.WorkerClass)
    processed_df.Education = le.fit_transform(processed_df.Education)
    processed_df.EnrolledInSchool = le.fit_transform(processed_df.EnrolledInSchool)
    processed_df.MaritalStatus = le.fit_transform(processed_df.MaritalStatus)
    processed_df.MajorIndustryCode = le.fit_transform(processed_df.MajorIndustryCode)
    processed_df.MajorOccupationCode = le.fit_transform(processed_df.MajorOccupationCode)
    processed_df.Race = le.fit_transform(processed_df.Race)
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.FullPartTime = le.fit_transform(processed_df.FullPartTime)
    processed_df.TaxFilerStatus = le.fit_transform(processed_df.TaxFilerStatus)
    processed_df.DetailedHouseholdSummary = le.fit_transform(processed_df.DetailedHouseholdSummary)
    processed_df.FamilyU18 = le.fit_transform(processed_df.FamilyU18)
    processed_df.FatherBirthCountry = le.fit_transform(processed_df.FatherBirthCountry)
    processed_df.MotherBirthCountry = le.fit_transform(processed_df.MotherBirthCountry)
    processed_df.BirthCountry = le.fit_transform(processed_df.BirthCountry)
    processed_df.Citizenship = le.fit_transform(processed_df.Citizenship)
    return processed_df

# Call the process function defined above  
data_processed = process(data)

'''
# Shuffle the columns
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_processed = data_processed[cols]
'''

# Split the dataset into test and train datasets, with 80% of the data going to the training set
train, test = sklearn.cross_validation.train_test_split(data_processed, train_size = 0.8)

# Split the known attributes from the attribute to be classified
X_train = train.drop(['Income'], axis = 1).values
Y_train = train['Income'].values
X_test = test.drop(['Income'], axis = 1).values
Y_test = test['Income'].values

# Build a Random Forest model using the x_train and y_train data
clf_rf = ske.RandomForestClassifier(n_estimators=50)
clf_rf.fit(X_train,Y_train)

# Predict whether Income is >50K or not using the X_test data
Y_pred = clf_rf.predict(X_test)

# Print the score
print(clf_rf.score(X_train, Y_train))
print(metrics.accuracy_score(Y_test, Y_pred))

'''

X = data_processed.drop(['Income'], axis = 1).values
Y = data_processed['Income'].values

clf_rf = ske.RandomForestClassifier(n_estimators=50)
predicted_rf = cross_val_predict(clf_rf, X, Y, cv=10)
print(metrics.accuracy_score(Y, predicted_rf))

clf_dt = tree.DecisionTreeClassifier(max_depth=10)
predicted_dt = cross_val_predict(clf_dt, X, Y, cv=10)
print(metrics.accuracy_score(Y, predicted_dt))

clf_gnb = GaussianNB()
predicted_gnb = cross_val_predict(clf_gnb, X, Y, cv=10)
print(metrics.accuracy_score(Y, predicted_gnb))

clf_SVC = SVC()
predicted_SVC = cross_val_predict(clf_SVC, X, Y, cv=10)
print(metrics.accuracy_score(Y, predicted_SVC))

clf_KNN = neighbors.KNeighborsClassifier()
predicted_KNN = cross_val_predict(clf_KNN, X, Y, cv=10)
print(metrics.accuracy_score(Y, predicted_KNN))

'''