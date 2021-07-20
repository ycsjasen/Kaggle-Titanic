import pandas as pd
from matplotlib import pyplot as plt
from helper import survive_df

# Loading data
df_training = pd.read_csv('train.csv')

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
df_training.to_csv('train_clean.csv', index=False)


# Exploratory Data Analysis
attribs = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Sex', 'Embarked']

# Quantifying survivor data in contingency table
# Displaying contingency table of each attribute
# Age (continuous)
print(survive_df(df_training, 'Age', True))

# SibSp (discreet)
print(survive_df(df_training, 'SibSp', False))

# Parch (discreet)
print(survive_df(df_training, 'Parch', False))

# Fare (continuous)
print(survive_df(df_training, 'Fare', True))

# Pclass (discreet)
print(survive_df(df_training, 'Pclass', False))

# Sex (discreet)
print(survive_df(df_training, 'Sex', False))

# Embarked (discreet)
print(survive_df(df_training, 'Embarked', False))


# Visualizing survivor data
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
mylist = []
for x in range(2):
    for y in range(4):
        mylist.append((x, y))

for i in range(len(attribs)):
    axes[mylist[i]].hist([df_training[df_training['Survived'] == 1][attribs[i]],
                          df_training[df_training['Survived'] == 0][attribs[i]]], stacked=True, edgecolor='black',
                         linewidth=1.5, label=['Survived', 'Deceased'])
    axes[mylist[i]].legend()
    axes[mylist[i]].set_title(attribs[i])
plt.tight_layout()
plt.savefig('initial eda visual')
plt.show()

