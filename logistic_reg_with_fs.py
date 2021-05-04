import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from helper import ohe_fn

# loading dataset and defining training and testing sets
df_train = pd.read_csv('train_clean.csv')
df_test = pd.read_csv('test_clean.csv')
cate_var = ['Sex', 'Embarked', 'Pclass']
cont_var = ['SibSp', 'Parch', 'Age', 'Fare']

X_train = df_train[cate_var]
y_train = df_train.Survived
X_test = df_test[cate_var]

# initializing encoder object
ohe = OneHotEncoder()
ohe.fit(X_train)

# using helper function to encode categorical variables and joining with continuous variables
ohe_train = ohe_fn(df_train, X_train, ohe, cont_var)
ohe_test = ohe_fn(df_test, X_test, ohe, cont_var)

# initializing tree object to obtain feature importance values
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf = clf.fit(ohe_train, y_train)

# calculating and collecting feature importance values
fi_col = []
fi = []
for i, column in enumerate(ohe_train):
    fi_col.append(column)
    fi.append(clf.feature_importances_[i])

fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns=['Feature', 'Feature Importance'])
print(fi_df.sort_values('Feature Importance', ascending=False))

'''
Output: 
       Feature  Feature Importance
1     Sex_male            0.511353
7     Pclass_3            0.168112
10         Age            0.146929
11        Fare            0.121140
8        SibSp            0.034988
9        Parch            0.009095
2   Embarked_C            0.008382
0   Sex_female            0.000000
3   Embarked_Q            0.000000
4   Embarked_S            0.000000
5     Pclass_1            0.000000
6     Pclass_2            0.000000
'''

# determining which features to keep
columns_to_keep = fi_df['Feature'][fi_df['Feature Importance'] > 0]

# finalizing training and testing sets
X_train = ohe_train[columns_to_keep]
X_test = ohe_test[columns_to_keep]

# initializing and fitting logistic Regression model
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)

# predicting survivors using logistic regression model
y_preds = logreg.predict(X_test)
df_test['Survived'] = y_preds

# exporting predictions
final_result = df_test.loc[:, ['PassengerId', 'Survived']]
final_result.to_csv('log_reg_submission.csv', index=False)
