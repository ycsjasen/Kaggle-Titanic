import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from helper import ohe_fn

# loading dataset and defining training and testing sets
df_train = pd.read_csv('../Data loading and cleaning/train_clean.csv')
df_test = pd.read_csv('../Data loading and cleaning/test_clean.csv')
cate_var = ['Sex', 'Embarked', 'Pclass']
cont_var = ['SibSp', 'Parch', 'Age', 'Fare']

X_train = df_train[cate_var]
y_train = df_train['Survived']
X_test = df_test[cate_var]

# initializing encoder object
ohe = OneHotEncoder()
ohe.fit(X_train)

# using helper function to encode categorical variables and joining with continuous variables
ohe_train = ohe_fn(df_train, X_train, ohe, cont_var)
ohe_test = ohe_fn(df_test, X_test, ohe, cont_var)

# initializing and fitting logistic regression model
logreg = LogisticRegression(solver='liblinear')
logreg.fit(ohe_train, y_train)

# predicting survivors using logistic regression model
y_preds = logreg.predict(ohe_test)
df_test['Survived'] = y_preds

# exporting predictions
final_result = df_test.loc[:, ['PassengerId', 'Survived']]
final_result.to_csv('log_reg_simple_submission.csv', index=False)
