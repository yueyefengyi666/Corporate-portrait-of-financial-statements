from scipy.io import arff
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier



import warnings
warnings.filterwarnings('ignore')
os.chdir(r'C:\Users\Administrator\PycharmProjects\untitled2') # set the current directory where the files are downloaded
# Read the file names in a list

file_names = []
for i in range(1,6):
    counter = str(i)
    name = counter+'year.arff'
    file_names.append(name)
#文件名

print("LIST OF FILE NAMES: " , file_names)
# Load the metadata in a list called file_data using the file_names:

file_data = []
for file in file_names:
    file_data.append(arff.loadarff(file))

print(file_data[0:2])  # Show how the metadata looks like
# Create a readable dataframe object from our .arff metadata:

for i in range(0, len(file_data)):
    if i == 0:
        df = pd.DataFrame.from_records(data=file_data[i][0])
        continue

    if i != 0:
        d = pd.DataFrame.from_records(data=file_data[i][0])
        df = df.append(d)
print("Shape of our DataFrame: ", df.shape)
print(df.head())
df.columns = map(str.lower, df.columns) # lowercase all column names
df['class'] = df['class'].astype('int') # convert the last column (target column) to integer format.转换y
df.reset_index()
print(df.head()) # let's check our new dataframe
import seaborn as sns
# Graphical representation of the missing values.
x = df.columns
y = df.isnull().sum()
sns.set()
sns.set(rc={'figure.figsize':(22,10)})
sns.barplot(x,y)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            int(height),
            fontsize=15, ha='center', va='bottom')
sns.set(font_scale=2)
ax.set_xlabel("Data Attributes")
ax.set_ylabel("count of missing records for each attribute")
plt.xticks(rotation=90)
print(plt.show())
#画出每个属性的缺失值

total_len = len(df['class'])
percentage_labels = (df['class'].value_counts()/total_len)*100
percentage_labels
sns.set()
sns.countplot(df['class']).set_title('Data Distribution for target variable')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=3)
ax.set_xlabel("y(Labels for class distribution)")
ax.set_ylabel("Numbers of the records")
plt.show()
#画出y的取值分布

sns.set(style="white")


# Compute the correlation matrix
corr = df.corr()


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
df1 = df.fillna(df.mean())  # df1 is mean replaced
X = df1.drop('class', axis=1)
y = df1['class']
print(df.describe())
# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler
# Create the scaler
ss = StandardScaler()

X_scaled = ss.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, stratify = y, random_state = 42)
from sklearn.model_selection import cross_val_score


def cvDictGen(functions, scr, X_train, y_train, cv=5, verbose=1):
    cvDict = {}
    for func in functions:
        cvScore = cross_val_score(func, X_train, y_train, cv=cv, verbose=verbose, scoring=scr)
        cvDict[str(func).split('(')[0]] = [cvScore.mean(), cvScore.std()]

    return cvDict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define some inital models:

# K-Nearest Neighbor (KNN)

knMod = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                             metric_params=None)

# Logistic Regression

glmMod = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None,
                            random_state=None, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=2)

# AdaBoost

adaMod = AdaBoostClassifier(base_estimator=None, n_estimators=200, learning_rate=1.0)

## GradientBoosting

gbMod = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_depth=3,
                                   init=None, random_state=None, max_features=None, verbose=0)

#  RandomForest
rfMod = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1,
                               random_state=None, verbose=0)
# Bagging Classifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=None), n_estimators=5, random_state=None)

# Neural Network Classifier - Multi layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(12, 12, 12), random_state=None)

cvd = cvDictGen(functions=[knMod, glmMod, adaMod, gbMod, rfMod,bagging,mlp], X_train = X_train, y_train= y_train, scr='roc_auc')
print(cvd)
#选择基准模型
#输出roc-auc





#继续进行模型比较，输出更多参考值
metrics = pd.DataFrame(index=['roc_auc','precision','recall'],
                      columns=['GradienBoosting', 'Adaboost', 'RandomForest'])
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
# Gradient Boosting:
gbMod.fit(X_train, y_train)
y_pred_test1 = gbMod.predict(X_test)


test_labels1 = gbMod.predict_proba(np.array(X_test))[:,1]

metrics.loc['roc_auc','GradienBoosting'] = roc_auc_score(y_test,test_labels1 , average='macro', sample_weight=None)
metrics.loc['precision','GradienBoosting'] = precision_score(y_pred=y_pred_test1, y_true=y_test)
metrics.loc['recall','GradienBoosting'] = recall_score(y_pred=y_pred_test1, y_true=y_test)
# Gradient Boosting:  X_test1 & y_test1 are mean replaced data
adaMod.fit(X_train, y_train)
y_pred_test2 = adaMod.predict(X_test)

test_labels2 = adaMod.predict_proba(np.array(X_test))[:,1]

metrics.loc['roc_auc','Adaboost'] = roc_auc_score(y_test,test_labels2 , average='macro', sample_weight=None)
metrics.loc['precision','Adaboost'] = precision_score(y_pred=y_pred_test2, y_true=y_test)
metrics.loc['recall','Adaboost'] = recall_score(y_pred=y_pred_test2, y_true=y_test)
# RandomForestClassifier:  X_test1 & y_test1 are mean replaced data
rfMod.fit(X_train, y_train)
y_pred_test3 = rfMod.predict(X_test)


test_labels3 = rfMod.predict_proba(np.array(X_test))[:,1]

metrics.loc['roc_auc','RandomForest'] = roc_auc_score(y_test,test_labels3 , average='macro', sample_weight=None)
metrics.loc['precision','RandomForest'] = precision_score(y_pred=y_pred_test3, y_true=y_test)
metrics.loc['recall','RandomForest'] = recall_score(y_pred=y_pred_test3, y_true=y_test)
fig, ax = plt.subplots(figsize=(12,8))
metrics.plot(kind='barh', ax=ax)
ax.grid();
plt.show()
print(metrics.T)
def CMatrix(CM, labels=['operating','bankrupt']):
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df

CM1 = confusion_matrix(y_pred=y_pred_test1, y_true=y_test)
CM2 = confusion_matrix(y_pred=y_pred_test2, y_true=y_test)
CM3 = confusion_matrix(y_pred=y_pred_test3, y_true=y_test)
print(CMatrix(CM1))
print(CMatrix(CM2))
print(CMatrix(CM3))
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
gbHyperParams = {'loss' : ['deviance', 'exponential'],
                 'n_estimators': randint(10, 500),
                 'max_depth': randint(1,10)}


gridSearchGB = RandomizedSearchCV(estimator=gbMod, param_distributions=gbHyperParams, n_iter=10,
                                   scoring='roc_auc', fit_params=None, cv=None, verbose=2).fit(X_train, y_train)
print(gridSearchGB.best_params_, gridSearchGB.best_score_)
gbMod_best = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=449, subsample=1.0,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_depth=8,
                                   init=None, random_state=None, max_features=None, verbose=0)
bestGbModFitted = gridSearchGB.best_estimator_.fit(X_train, y_train)
test_labels_best=bestGbModFitted.predict_proba(np.array(X_test))[:,1]
y_pred_best = rfMod.predict(X_test)
CM_best = confusion_matrix(y_pred=y_pred_best, y_true=y_test)
print(CMatrix(CM_best))
metrics.loc['roc_auc','GradienBoosting_Hypertuned'] = roc_auc_score(y_test,test_labels_best , average='macro', sample_weight=None)
metrics.loc['precision','GradienBoosting_Hypertuned'] = precision_score(y_pred=y_pred_best, y_true=y_test)
metrics.loc['recall','GradienBoosting_Hypertuned'] = recall_score(y_pred=y_pred_best, y_true=y_test)
print(metrics.T)
fig, ax = plt.subplots(figsize=(12,8))
metrics.plot(kind='barh', ax=ax)
ax.grid();
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_best))
from xgboost.sklearn import XGBClassifier
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=35,
 seed=27)
xgb1.fit(X_train,y_train)
y_pred_xgb = xgb1.predict(X_test)
predictions = [round(value) for value in y_pred_xgb]
CMatrix(confusion_matrix(y_pred = predictions, y_true=y_test))
print(classification_report(y_test,predictions))
print(roc_auc_score(y_test,predictions , average='macro', sample_weight=None))
gbm_param_grid = {
        'learning_rate' : np.arange(0.05,1.05,0.05),
        'n_estimators'  : np.arange(200,1600,200),
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

print(gbm_param_grid)

xgb2 = XGBClassifier(
 max_depth=5,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=35,
 seed=42)
random_search = RandomizedSearchCV(estimator=xgb2, param_distributions=gbm_param_grid, n_iter=25, scoring='recall',cv=4,verbose=1)
random_search.fit(X_train,y_train)

print(random_search.best_params_)
bestxbg = random_search.best_estimator_.fit(X_train, y_train) # fit the model with best parameters
y_pred_xgb_best = bestxbg.predict(X_test) # predict the labels
predictions = [round(value) for value in y_pred_xgb_best]
CMatrix(confusion_matrix(y_pred = predictions, y_true=y_test))
print(classification_report(y_test,predictions))
print(roc_auc_score(y_test,predictions , average='macro', sample_weight=None))
features_label = X.columns
feature_list = dict(zip(features_label, bestxbg.feature_importances_*100))
#feature_list = list(zip(features_label, bestxbg.feature_importances_))
# Use the __getitem__ method as the key function
top_ten_features = sorted(feature_list, key=feature_list.__getitem__, reverse= True)[:11]
print(top_ten_features)
#输出前十的重要特征
