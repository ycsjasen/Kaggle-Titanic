import pandas as pd

# Loading data
project_path = "C:/Users/ycsja/Desktop/github/Kaggle_Titanic/Kaggle-Titanic"
data_testing_set = "/test.csv"

df_testing = pd.read_csv(project_path + data_testing_set)

# Data cleaning
# Dropping Name, Ticket number, Passenger ID factors
df_testing = df_testing.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# describing missing values
count_null = df_testing.isnull().sum()
percent_null = count_null / len(df_testing.index) * 100
total_null = pd.concat([count_null, percent_null], axis=1, keys=['Amount NA', 'Percent NA'])
print(total_null)

# Replacing missing value in Fare with the median value for Fare
fare_med = df_testing['Fare'].median()
df_testing['Fare'] = df_testing['Fare'].fillna(fare_med)

# Replacing missing values in Age with the median value for Age
age_med = df_testing['Age'].median()
df_testing['Age'] = df_testing['Age'].fillna(age_med)

# Descriptive statistics
print(df_testing.describe())
df_testing.to_csv('test_clean.csv', index=False)




