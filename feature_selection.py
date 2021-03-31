import pandas as pd
from helper import survive_df, entropy_calc

df_training = pd.read_csv('train_clean.csv')

# Separating continuous attributes into bins
df_training['Binned Age'] = pd.cut(df_training['Age'], bins=10)
df_training['Binned Fare'] = pd.cut(df_training['Fare'], bins=10)

# Creating contingency tables
attribs = ['Binned Age', 'SibSp', 'Parch', 'Binned Fare', 'Pclass', 'Sex', 'Embarked']

# To add for loop later

# Contingency Table for Sex
df_join = df_training[['Sex', 'Survived']].groupby(['Sex']).count()
df_entr = pd.crosstab(df_training['Sex'], df_training['Survived'])
print(df_entr)


# Information Gain = H(Surviving) - E(Survive, Sex)
# Entropy of Surviving
df_entr_sex = survive_df(df_training, 'Sex', False)
total_passenger = df_entr_sex['Total Passenger'].sum()
total_survived = df_entr_sex['Survivor Count'].sum()

# H(Surviving)
H_survive = entropy_calc(total_survived, total_passenger)
print('Entropy of surviving: %f bits' % H_survive)

# Entropy of Surviving given gender
female_survive = df_entr_sex['Survivor Count']['female']
female_total = df_entr_sex['Total Passenger']['female']
male_survive = df_entr_sex['Survivor Count']['male']
male_total = df_entr_sex['Total Passenger']['male']

# E(Survive, Sex)
p_female = female_total / total_passenger
p_male = male_total / total_passenger
E_survive_female = entropy_calc(female_survive, female_total)  # E(Survive, Sex = female)
print('Entropy of female surviving: %f bits' % E_survive_female)
E_survive_male = entropy_calc(male_survive, male_total)  # E(survive, Sex = male)
print('Entropy of male surviving: %f bits' % E_survive_male)
E_survive_sex = p_female * E_survive_female + p_male * E_survive_male  # Weighted average of entropy
print('Weighted average of entropy surviving given sex: %f bits' % E_survive_sex)

# Information Gain
ig_sex = H_survive - E_survive_sex
print('Information Gain: %f bits' % ig_sex)
