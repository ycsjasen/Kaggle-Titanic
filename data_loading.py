import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import svm

# Loading data
# data_training_set = "../input/titanic/train.csv"
# data_testing_set = "../input/titanic/test.csv"
project_path = "C:/Users/ycsja/Desktop/Python/Kaggle projects/Kaggle_Titanic"
data_training_set = "/train.csv"
data_testing_set = "/test.csv"

df_training = pd.read_csv(project_path + data_training_set)
df_testing = pd.read_csv(project_path + data_testing_set)

# Data cleaning
# Dropping Name, Ticket number, Passenger ID factors
df_training = df_training.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# describing missing values
count_null = df_training.isnull().sum()
percent_null = count_null / len(df_training.index) * 100
total_null = pd.concat([count_null, percent_null], axis=1, keys=['Amount NA', 'Percent NA'])
print(total_null)

# dropping Cabin attribute due to 77% missing values
df_training = df_training.drop(['Cabin'], axis=1)

# Dropping 2 rows of missing "Embarked" values
df_training = df_training.dropna(axis=0, subset=['Embarked'])

# Replacing missing values in Age with the median value for Age
age_med = df_training['Age'].median()
df_training['Age'] = df_training['Age'].fillna(age_med)

# Descriptive statistics
print(df_training.describe())
# df_training.to_csv('train_clean.csv', index=False)

# Exploratory Data Analysis
attribs = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Sex', 'Embarked']
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15,10))

# Visualizing survivor data
mylist = []
for x in range(2):
    for y in range(4):
        mylist.append((x,y))

for i in range(len(attribs)):
        axes[mylist[i]].hist([df_training[df_training['Survived'] == 1][attribs[i]],
                              df_training[df_training['Survived'] == 0][attribs[i]]], stacked=True,
                             edgecolor='black', linewidth=1.5, label=['Survived', 'Deceased'])
        axes[mylist[i]].legend()
        axes[mylist[i]].set_title(attribs[i])
plt.tight_layout()
plt.savefig('initial eda visual')
plt.show()
