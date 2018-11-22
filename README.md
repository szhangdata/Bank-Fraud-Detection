# Bank-Fraud-Detection
**The test set contains customer’s basic attributes and payment histories. The output will predict whether these customers are credible or not.**
# **1. Load the data**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('Desktop/DAL/train.csv', header=0)
test = pd.read_csv('Desktop/DAL/test.csv', header=0)
print(train.shape)
print(test.shape)
train.head()
test.head()
train.info()
train.isnull().sum()
test.isnull().sum()

# **2. Distinguish categorical and continuous variables**
cat_v = []
con_v = []
for c in train.columns:
    if len(train[c].value_counts().index)<=15:
        cat_v.append(c)
    else:
        con_v.append(c)
cat_v.remove('Y')
target = ['Y']
print("The continuous variables: ", con_v, "\n")
print("The categorical variables: ", cat_v)

# **3. Basic feature analysis**
# i. Check the pattern differences between the training data and testing data

count=1
for i in range(len(cat_v)):
    fig = plt.figure(figsize=(30,80))
    plt.subplot(len(cat_v),2,count)
    plt.bar(train[cat_v[i]].value_counts().index, train[cat_v[i]].value_counts().values)
    plt.title("train "+cat_v[i])
    
    plt.subplot(len(cat_v),2,count+1)
    plt.bar(test[cat_v[i]].value_counts().index, test[cat_v[i]].value_counts().values)
    plt.title("test "+cat_v[i])
    count+=2
    
count=1
for i in range(len(con_v)):
    fig = plt.figure(figsize=(20,100))
    plt.subplot(len(con_v),2,count)
    plt.violinplot(train[con_v[i]],showmeans=True)
    plt.title("train "+con_v[i])
    
    plt.subplot(len(con_v),2,count+1)
    plt.violinplot(test[con_v[i]],showmeans=True)
    plt.title("test "+con_v[i])
    count+=2
    
# ii. Check the if there are linear relationships between features
def plot_corr(df,size=15):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr,cmap=plt.get_cmap('rainbow'))
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(cax)
plot_corr(train)

# iii. Check pattern of the label
fig = plt.figure(figsize=(20,10))
plt.bar(train['Y'].value_counts().index, train['Y'].value_counts().values)
plt.xticks(train['Y'].value_counts().index,fontsize=15)
plt.show()

# **4. Build a baseline model**
from sklearn.model_selection import train_test_split
Y = train['Y']
X = train.drop(['Y', 'id'], axis= 1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
RF = RandomForestClassifier(class_weight = {0:1, 1:3})
RF = RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)
print (metrics.classification_report(y_test, y_pred))

# **5. Basic parameter tuning: Grid Searching**
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
def search_model(x_train, y_train, est, param_grid, n_jobs, cv):
    model = GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring = 'f1_weighted',
                                     verbose = 10,
                                     n_jobs = n_jobs,
                                     iid = True,
                                     cv = cv)
    # Fit Grid Search Model
    model.fit(x_train, y_train)   
    return model
param_grid = {'n_estimators':[100,300,500],
             'criterion':['gini', 'entropy'],
             'class_weight': [{0:1, 1:3}]}

RF = search_model(X.values
            , Y.values
            , RandomForestClassifier()
            , param_grid
            , -1
            , 5)

print("Best score: %0.3f" % RF.best_score_)
print("Best parameters set:", RF.best_params_)
print("Scores:", RF.grid_scores_) 

# **6. Model Ensemble**

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

clf1 = RandomForestClassifier(n_estimators = 300, criterion = 'entropy',class_weight = {0:1, 1:3})
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = LogisticRegression (class_weight = {0:1, 1:3})

# results from your gridsearch
eclf = VotingClassifier(estimators=[('Random_Forest',clf1), ('KNN', clf2),('Logistic', clf3)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['Random_Forest', 'KNN','Logistic', 'Ensemble']):
    scores = cross_val_score(clf, X, Y, cv=3, scoring='f1_weighted')
    print ("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# **7. Generate the final submission**
eclf.fit(X, Y)
y = pd.DataFrame(eclf.predict(test.drop(['id'],axis=1)), columns=['y'])
predict_data = pd.concat([y, test['id']], axis =1)
predict_data.to_csv('Submmission.csv', index=False)

output = pd.read_csv('Submmission.csv')
output.head()
output.y.value_counts()
plt.bar(output.y.value_counts().index, output.y.value_counts().values)
