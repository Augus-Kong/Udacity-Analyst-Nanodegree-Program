
# coding: utf-8

# ### Task 1 Select what features you'll use
# 

# In[1]:

import sys
import pickle
sys.path.append("../tools/")
#import all necessary libraries for manipulating data and building the ML models

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'deferral_payments', 'email_address','total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees','to_messages', 
                  'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi',]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print data_dict


# #### 先对数据集进行探索性的分析，得到相关的基本统计信息例如
# the records, total poi numbers, total NaN value

# In[2]:

#basic information about the dataset

print 'Total records in dataset: ', len(data_dict)

n = 0
poi_name =[]
for employee in data_dict:
    if data_dict[employee]['poi'] == True:
        n += 1
        poi_name.append(employee)
print 'Total poi numbers in dataset: ', n
print poi_name

nan_features = features_list
nan_count = 0
for employee in data_dict:
    for feature in nan_features:
        if data_dict[employee][feature] == 'NaN':
            nan_count +=1
print 'Total NaN value in dataset: ', nan_count


# In[3]:

#Find the 'NaN' number 

features_list = ['poi','salary', 'deferral_payments', 'total_payments','email_address', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees','to_messages', 
                  'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi',]

def countNa(feature):
    count_NaN = 0
    for employee in data_dict:
        if data_dict[employee][feature] == 'NaN':
            count_NaN += 1
    print 'NaN data for ',str(feature), count_NaN
    
Na_list = []
for feature in features_list:
    Na_list.append(countNa(feature))


# In[4]:

count_total_incentives_NaN = 0
for entry in data_dict:
    if data_dict[entry]['long_term_incentive'] == 'NaN':
        count_total_incentives_NaN += 1
print 'NaN data for long_term_incentives:', count_total_incentives_NaN


# ### Task2 : Remove the outliers

# In[5]:

#根据观察发现两个多项信息均为NaN的异常值,其中‘THE TRAVEL AGENCY IN THE PARK’不能够表示一个嫌疑人数据，'LOCKHART EUGENE E'数据中NaN值无效信息过多，
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)


# In[6]:

import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
get_ipython().magic(u'matplotlib inline')

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# In[7]:

#remove the total key
data_dict.pop('TOTAL', 0)


# ### Task3 Features Selection + Create new features
# 
# #### 将主要变量均为'NaN'的值异常值全部清理之后，我们需要一个自行添加的新变量来提高对于POI的判断准确率。在text learning的课程中提到了poi 成员之间会有比较密集的邮件通信，根据这一信息添加一个fraction_poi_emails变量来间接判断一个成员是否为poi值。

# In[8]:

def get_email_information(key1, key2):
    new_list = []
    for i in data_dict:
        if data_dict[i][key1] == 'NaN' or data_dict[i][key2] == 'NaN':
            new_list.append(0.) #add . to use float number
        elif data_dict[i][key1] >= 0:
            new_list.append(float(data_dict[i][key1]) + float(data_dict[i][key2]))
    return new_list

total_poi_emails = get_email_information('from_this_person_to_poi','from_poi_to_this_person')
total_emails = get_email_information('to_messages','from_messages') 
#get the total emails

def fraction_list(list1, list2):
    'devide one list by other'
    fraction = []
    for i in range(0, len(list1)):
        if list2[i] == 0.0:
            fraction.append(0.0)
        else:
            fraction.append(float(list1[i])/float(list2[i]))
    return fraction
print total_poi_emails


# In[9]:

#get the fraction of poi emails
fraction_poi_emails = fraction_list(total_poi_emails, total_emails)
print fraction_poi_emails


# In[10]:

#add this new feature to my data
count = 0
for i in data_dict:
    data_dict[i]['fraction_poi_emails'] = fraction_poi_emails[count]
    count += 1
#test    
print 'HUMPHREY GENE E: ', data_dict['HUMPHREY GENE E']['fraction_poi_emails']


# ### Task4: Try a varity of classifiers

# #### 使用决策树算法 
# ##### 先使用包含新特征 fraction_poi_emails 的features_list, 再使用不包含新特征的feature_list 分别计算算法的表现。

# In[11]:


from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
import numpy as np
np.random.seed(42)
from time import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# features_list
features_list = ['poi','salary', 'fraction_poi_emails','from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'director_fees', 'deferred_income', 'long_term_incentive']
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


DTacc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(DTacc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'Decision Tree algorithm run time: ', round(time()-t0, 3), 's'


# list of importance features
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(10):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


# In[12]:


from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
import numpy as np
np.random.seed(42)
from time import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# features_list
features_list = ['poi','salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'director_fees', 'deferred_income', 'long_term_incentive']
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


DTacc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(DTacc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'Decision Tree algorithm run time: ', round(time()-t0, 3), 's'


# list of importance features
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(10):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


# #### 新特征增加了算法的accuracy 和 precision值， 降低了recall值 

# #### 使用逻辑回归算法

# In[13]:

from sklearn.linear_model import LogisticRegression
t0 = time()
clf = LogisticRegression()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# features_list
features_list = ['poi','salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'director_fees', 'deferred_income', 'long_term_incentive']
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

from sklearn.metrics import accuracy_score
LRacc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(LRacc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)
print 'Logistic regression algorithm run time: ', round(time()-t0, 3), 's'


# In[14]:

from sklearn.linear_model import LogisticRegression
t0 = time()
clf = LogisticRegression()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# features_list
features_list = ['poi','salary','fraction_poi_emails', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'director_fees', 'deferred_income', 'long_term_incentive']
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

from sklearn.metrics import accuracy_score
LRacc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(LRacc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)
print 'Logistic regression algorithm run time: ', round(time()-t0, 3), 's'


# #### 使用支持向量机算法

# In[15]:

#do feature scaling for SVM
from sklearn.preprocessing import MinMaxScaler
features_list = ['poi','salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'director_fees', 'deferred_income', 'long_term_incentive']
data = featureFormat(data_dict, features_list)
valuedata = np.array(data)
scaler = MinMaxScaler()
rescaled_value = scaler.fit_transform(valuedata)
print rescaled_value


# In[16]:

labels, features = targetFeatureSplit(rescaled_value)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = 42)
from sklearn import svm
t0 = time()
clf = svm.LinearSVC(C=0.5)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
SVCacc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(SVCacc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)
print 'SVC algorithm run time: ', round(time()-t0, 3), 's'


# In[17]:

labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)
from sklearn import svm
t0 = time()
clf = svm.LinearSVC(C = 1.2)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
SVCacc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(SVCacc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)
print 'SVC algorithm run time: ', round(time()-t0, 3), 's'


# ### Tuning by SelectKBest 函数
# 

# In[18]:

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# load the data
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

# featureas_list
features_list = [ 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
                 'from_messages', 'other', 'from_this_person_to_poi','director_fees',
                 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=6)
selectedFeatures = selector.fit(features,labels)
feature_names = [features_list[i] for i in selectedFeatures.get_support(indices=True)]
print 'Best features: ', feature_names
print selector.scores_


# In[19]:

features_list = ['poi','director_fees', 'deferred_income','deferral_payments', 'exercised_stock_options', 'bonus', 'total_stock_value', 'expenses']

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
from tester import test_classifier
test_classifier(clf, data_dict, features_list)


# ### Tuning by GridSearchCV 函数

# In[24]:

from sklearn.grid_search import GridSearchCV
t0 = time()
param_grid = {
         'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
          'max_depth': [1, 2, 3, 4, 5, 6, 7,8],
            'max_features': range(3,7)
          }
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'f1')
clf = clf.fit(features_train, labels_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
clf = clf.best_estimator_
from tester import test_classifier
test_classifier(clf, data_dict, features_list)


# ### 使用GridSearchCV进行参数调整，通过多次运行之后，确定的参数为 class_weight=None, criterion='gini', max_depth=7,max_features=3, max_leaf_nodes=None, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=None, splitter='best'  将最后的DT算法中的参数调整为GridSearchCV所得到的结果
# 

# In[25]:

clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
            max_features=3, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
clf = clf.fit(features_train, labels_train)
from tester import test_classifier
test_classifier(clf, data_dict, features_list)


# In[26]:

dump_classifier_and_data(clf, data_dict, features_list)


# In[ ]:



