import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# ignore warning messages to improve code readability and suppress unnecessary output
warnings.filterwarnings('ignore')

# read dataset and add to dataframe bank_data
bank_data = pd.read_csv("C:/Users/arwin/python folder/bank-customer-churn-dataset/Churn_Modelling.csv", sep=",")

# first 5 rows of data
print(bank_data.head())
# last 5 rows of data
print(bank_data.tail())
# number of rows and columns
print(bank_data.shape)

# names of the columns
print(bank_data.columns)

# data type and non-null count of each column
print(bank_data.info())

# count of uniques in each column
print(bank_data.nunique())

# descriptive statistics of numeric columns
print(bank_data.describe())

# descriptive statistics of categorical columns
print(bank_data.describe(include = ["object"]))

# histogram of credit score 
sns.histplot(bank_data, x="CreditScore" , bins = 100)
plt.show()

# histogram of all ages
sns.histplot(bank_data["Age"] , bins = 92-18+1)
plt.show()

# histogram ignoring all outliers
new_age = bank_data["Age"][bank_data["Age"]<= 70]
sns.histplot(new_age , bins = 70-25+1)
plt.show()

# tenure column
sns.countplot(bank_data, x="Tenure")
plt.xticks(rotation=90)
plt.show()

# histogram of balance column
sns.histplot(bank_data["Balance"], bins=100)
plt.show()

# identifying outliers
balance_data = bank_data["Balance"]

Q1 = np.percentile(balance_data, 25)
Q3 = np.percentile(balance_data, 75)

IQR = Q3 - Q1

lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

print("Lower whisker:", lower_whisker)
print("Upper whisker:", upper_whisker)

# histogram of balance column without outliers
sns.histplot(bank_data["Balance"][(bank_data["Balance"] >= -191466.4) & (bank_data["Balance"] <= 319110.6)])
plt.show()

# numofproducts column
sns.countplot(bank_data, x="NumOfProducts")
plt.xticks(rotation=90)
plt.show()

# hascrcard column
sns.countplot(bank_data, x="HasCrCard")
plt.xticks(rotation=90)
plt.show()

# hascrcard column pie chart
has_card = bank_data["HasCrCard"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(has_card, labels=has_card.index, autopct='%1.1f%%', startangle=45)
plt.show()

# isactivemember column
sns.countplot(bank_data, x="IsActiveMember")
plt.xticks(rotation=90)
plt.show()

# hascrcard column pie chart
has_card = bank_data["IsActiveMember"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(has_card, labels=has_card.index, autopct='%1.1f%%', startangle=45)
plt.show()

# estimatedsalary column
sns.histplot(bank_data["EstimatedSalary"], bins=100)
plt.show()

# identifying outliers
salary_data = bank_data["EstimatedSalary"]

Q1 = np.percentile(salary_data, 25)
Q3 = np.percentile(salary_data, 75)

IQR = Q3 - Q1

lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

print("Lower whisker:", lower_whisker)
print("Upper whisker:", upper_whisker)

# histogram of salary column without outliers
sns.histplot(bank_data["EstimatedSalary"][(bank_data["EstimatedSalary"] >= -96577.1) & (bank_data["EstimatedSalary"] <= 296967.5)])
plt.show()

# exited column
sns.countplot(bank_data, x='Exited')
plt.show()

# exited column pie chart
exit_status = bank_data["Exited"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(exit_status, labels=exit_status.index, autopct='%1.1f%%', startangle=45)
plt.show()

# exited value count
print(bank_data["Exited"].value_counts())

# geography column
sns.countplot(bank_data, x='Geography')
plt.show()

# geography pie chart
exit_status = bank_data["Geography"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(exit_status, labels=exit_status.index, autopct='%1.1f%%', startangle=45)
plt.show()

# gender column
sns.countplot(bank_data, x='Gender')
plt.show()

# gender pie chart
exit_status = bank_data["Gender"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(exit_status, labels=exit_status.index, autopct='%1.1f%%', startangle=45)
plt.show()

# bivirate
data_copy = bank_data.copy()
numeric_data = data_copy.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True)
plt.show()

# encoding

# converting male/female to 1/0
bank_data['Gender'] = bank_data['Gender'].replace({"Male": 1 , "Female": 0})

# convert previous to 1/0 and dropping Surname
bank_data['Gender'] = bank_data['Gender'].apply(lambda x: 0 if x == 0 else 1)
bank_data.drop(columns=['Surname'], inplace=True)

# label encoding of the ordinal categorical features (Geography)
bank_data['Geography'] = bank_data['Geography'].replace({"France": 2 , "Germany": 1 , "Spain": 0})

# checking new table
print(bank_data.head())
print(bank_data.tail())
print(bank_data.columns)

# outlier handeling with IQR
outlier_columns = ["Balance" , "EstimatedSalary"]
lower_whisker = {"Balance":-191466.4 , "EstimatedSalary": -96577.1}
upper_whisker = {"Balance":319110.6 , "EstimatedSalary":296967.5}
for col in outlier_columns:
    bank_data[col] = np.where(bank_data[col] < lower_whisker[col], lower_whisker[col] , bank_data[col])
    bank_data[col] = np.where(bank_data[col] > upper_whisker[col] , upper_whisker[col] , bank_data[col])

# scaling using minmax scaler
columns = bank_data.columns
scaler = MinMaxScaler()
bank_data = scaler.fit_transform(bank_data)
bank_data = pd.DataFrame(bank_data , columns=[columns])

# data augmentation using SMOTE
y = bank_data["Exited"]
X = bank_data.drop("Exited" , axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

sm = SMOTE()
X_train , y_train = sm.fit_resample(X_train, y_train)
print(y_train.value_counts())

print(y_test.value_counts())

print(X_train.shape)

# modeling + evaluation

# utility function for plotting confusion matrix
def plot_confusion_matrix(y_test , y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    sns.heatmap(cm_norm, annot=cm, cmap='Blues', fmt='d', cbar=False)  
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show() 

# logistic regression
lr_model = LogisticRegression(max_iter=200 , random_state = 42)
lr_model.fit(X_train,y_train)
y_predlr = lr_model.predict(X_test)

plot_confusion_matrix(y_test , y_predlr)
print(classification_report(y_test, y_predlr))

# support vector machine
svc_model = SVC(random_state = 42)
svc_model.fit(X_train, y_train)
y_predsvc = svc_model.predict(X_test)

plot_confusion_matrix(y_test , y_predsvc)
print(classification_report(y_test, y_predsvc))

# k nearest neighbor
knn_model = KNeighborsClassifier(n_neighbors = 4)
knn_model.fit(X_train, y_train)
y_predknn = knn_model.predict(X_test)

plot_confusion_matrix(y_test , y_predknn)
print(classification_report(y_test, y_predknn))

# xgboost
xgb_model = xgb.XGBClassifier(eta = 0.25 , n_estimator = 100 , max_depth = 6, random_stare = 42)
xgb_model.fit(X_train, y_train)
y_predxgb = xgb_model.predict(X_test)

plot_confusion_matrix(y_test , y_predxgb)
print(classification_report(y_test, y_predxgb))

# adaboost
adb_model = AdaBoostClassifier(n_estimators=100,random_state=42)
adb_model.fit(X_train, y_train)
y_predadb = adb_model.predict(X_test)

plot_confusion_matrix(y_test , y_predadb)
print(classification_report(y_test, y_predadb))

# decision tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_preddt = adb_model.predict(X_test)

plot_confusion_matrix(y_test , y_preddt)
print(classification_report(y_test, y_preddt))

