import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from helper import survive_df
from math import log2

df_training = pd.read_csv('train_clean.csv')

# Separating continuous attributes into bins
df_training['Binned Age'] = pd.cut(df_training['Age'], bins=10)
df_training['Binned Fare'] = pd.cut(df_training['Fare'], bins=10)

# Calculating entropy with respect to each feature


attribs = ['Binned Age', 'SibSp', 'Parch', 'Binned Fare', 'Pclass', 'Sex', 'Embarked']
#contingency_df(df_training, attribs)
df_entr_age = df_training[['Sex', 'Survived']].groupby(['Sex', 'Survived']).size()
# df_entr_fare = df_training[['Binned Fare', 'Survived']].groupby(['Binned Fare', 'Survived']).size()
df_join = df_training[['Sex', 'Survived']].groupby(['Sex']).count()
#print(df_entr_age)


df_entr = pd.crosstab(df_training['Sex'], df_training['Survived'])
# print(df_entr.join(df_join, on='Sex'))




# Calculating Entropy and information gain

def entropy_calc(class1, class2):
    x = -(class1 * log2(class1) - (class2 * log2(class2)))
    return x

# Information Gain = H(Surviving) - E(Survive, Sex)
# Entropy of Surviving
df_entr_sex = survive_df(df_training, 'Sex', False)
total_passenger = df_entr_sex['Total Passenger'].sum()
total_survived = df_entr_sex['Survivor Count'].sum()

# H(Surviving)
p_survive = total_survived / total_passenger
p_decease = (total_passenger - total_survived) / total_passenger
H_survive = entropy_calc(p_survive, p_decease)


# Entropy of Surviving given gender
female_survive = df_entr_sex['Survivor Count']['female']
female_total = df_entr_sex['Total Passenger']['female']
male_survive = df_entr_sex['Survivor Count']['male']
male_total = df_entr_sex['Total Passenger']['male']

# E(Survive, Sex)
p_survive_female = female_survive / female_total
p_decease_female = (female_total - female_survive) / female_total
p_survive_male = male_survive / male_total
p_decease_male = (male_total - male_survive) / male_total
p_female = female_total / total_passenger
p_male = male_total / total_passenger

E_survive_sex = p_female * entropy_calc(p_survive_female, p_decease_female) +\
                p_male * entropy_calc(p_survive_male, p_decease_male)

# Information Gain
ig_sex = H_survive - E_survive_sex
print(ig_sex)
