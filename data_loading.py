import pandas as pd
from matplotlib import pyplot as plt

# Loading data
# data_training_set = "../input/titanic/train.csv"
# data_testing_set = "../input/titanic/test.csv"
project_path = "C:/Users/ycsja/Desktop/github/Kaggle_Titanic/Kaggle-Titanic"
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
df_training.to_csv('train_clean.csv', index=False)

# Exploratory Data Analysis
attribs = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Sex', 'Embarked']

# Quantifying survivor data
# Helper function


def survive_df(df, attribute, cont):
    """
    Calculate total count of survivors and survival rate given an attribute existing in DataFrame and outputs
    a summary DataFrame

    <b>Parameters: </b> <b>df</b> : <i>DataFrame</i>
                        <b>attribute</b> : <i>string</i>
                                    attribute to summarize
                        <b>cont</b> : <i>bool, default False</i>
                                    If True, separates attribute values into 10 equal sized bins
    <b>Returns: </b>    <b>DataFrame</b>
                            A DataFrame of the calculated total passengers, total amount of survivors and the rate of
                            survival for each category/bin of the attribute
    """
    if cont:
        df['Bin' + attribute] = pd.cut(df[attribute], bins=10)
        total_count = df[['Bin' + attribute, 'Survived']].groupby(['Bin' + attribute]).count()
        survive_count = df[['Bin' + attribute, 'Survived']].groupby(['Bin' + attribute]).sum()
        survive_rate = df[['Bin' + attribute, 'Survived']].groupby(['Bin' + attribute]).mean()

        total_count.rename(columns={'Survived': 'Total Passenger'}, inplace=True)
        survive_count.rename(columns={'Survived': 'Survivor Count'}, inplace=True)
        survive_rate.rename(columns={'Survived': 'Survivor Rate'}, inplace=True)

        summary_df = pd.merge(total_count, survive_count, how='inner', on='Bin' + attribute)
        summary_df = pd.merge(summary_df, survive_rate, how='inner', on='Bin' + attribute)
        return summary_df
    else:
        total_count = df[[attribute, 'Survived']].groupby([attribute]).count()
        survive_count = df[[attribute, 'Survived']].groupby([attribute]).sum()
        survive_rate = df[[attribute, 'Survived']].groupby([attribute]).mean()

        total_count.rename(columns={'Survived': 'Total Passenger'}, inplace=True)
        survive_count.rename(columns={'Survived': 'Survivor Count'}, inplace=True)
        survive_rate.rename(columns={'Survived': 'Survivor Rate'}, inplace=True)

        summary_df = pd.merge(total_count, survive_count, how='inner', on=attribute)
        summary_df = pd.merge(summary_df, survive_rate, how='inner', on=attribute)
        return summary_df


# Displaying summary of each attribute
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
