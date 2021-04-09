import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

df_train = pd.read_csv('train_clean.csv')
df_test = pd.read_csv('test_clean.csv')
cate_var = ['Sex', 'Embarked', 'Pclass']
cont_var = ['SibSp', 'Parch', 'Age', 'Fare']

X_train = df_train[cate_var]
y_train = df_train[['Survived']]

ohe = OneHotEncoder()
ohe.fit(X_train)
X_train_ohe = ohe.transform(X_train).toarray()
ohe_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names(X_train.columns))
join_df = df_train[cont_var]
ohe_df = ohe_df.join(join_df)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)

clf = clf.fit(ohe_df, y_train)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=900)
tree.plot_tree(clf, feature_names=ohe_df.columns, class_names=np.unique(y_train).astype('str'), filled=True)
plt.savefig('decision_tree')
plt.show()

# Testing

X_test = df_test[cate_var]
X_test_ohe = ohe.transform(X_test).toarray()
ohe_test = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names(X_test.columns))
join_test = df_test[cont_var]
ohe_test = ohe_test.join(join_test)

y_preds = clf.predict(ohe_test)
df_test['Survived'] = y_preds

final_result = df_test.loc[:, ['PassengerId', 'Survived']]
final_result.to_csv('decision_tree_submission.csv', index=False)
