import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from helper import ohe_fn

# loading dataset and defining training and testing sets
df_train = pd.read_csv('../Data loading and cleaning/train_clean.csv')
df_test = pd.read_csv('../Data loading and cleaning/test_clean.csv')
cate_var = ['Sex', 'Embarked', 'Pclass']
cont_var = ['SibSp', 'Parch', 'Age', 'Fare']

X_train = df_train[cate_var]
y_train = df_train[['Survived']]
X_test = df_test[cate_var]

# initializing encoder object
ohe = OneHotEncoder()
ohe.fit(X_train)

# using helper function to encode categorical variables and joining with continuous variables
ohe_train = ohe_fn(df_train, X_train, ohe, cont_var)
ohe_test = ohe_fn(df_test, X_test, ohe, cont_var)

# initializing and fitting decision tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf = clf.fit(ohe_train, y_train)

# plotting decision tree
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=900)
tree.plot_tree(clf, feature_names=ohe_train.columns, class_names=np.unique(y_train).astype('str'), filled=True)
plt.savefig('decision_tree')
plt.show()

# predicting test values
y_preds = clf.predict(ohe_test)
df_test['Survived'] = y_preds

# exporting predictions
final_result = df_test.loc[:, ['PassengerId', 'Survived']]
final_result.to_csv('decision_tree_submission.csv', index=False)
