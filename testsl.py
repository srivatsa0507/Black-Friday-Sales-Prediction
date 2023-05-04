import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
from pywaffle import Waffle
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# load data from MongoDB
from pymongo import MongoClient
client = MongoClient("localhost", 27017) 
db = client.ADTProject
train_collection = db.Train
test_collection = db.Test

train= pd.DataFrame(list(train_collection.find()))
train.drop('_id', inplace=True, axis=1)
test= pd.DataFrame(list(test_collection.find()))
test.drop('_id', inplace=True, axis=1)

# Combine train and test data
test['Purchase'] = np.nan
train['Type of data'] = 'Train'
test['Type of data'] = 'Test'
test_data = test[train.columns]
data = pd.concat([train, test_data], axis=0)
data.drop('Product_Category_3', axis=1, inplace=True)

# Create a Streamlit app
st.title('Black Friday Sales Prediction System')

# display information about the dataset

st.title("Data Exploration:")

if st.checkbox('Display dataset information'):
    st.write("Training data has {} rows and {} columns".format(train.shape[0],train.shape[1]))
    st.write("Testing data has {} rows and {} columns".format(test.shape[0],test.shape[1]))

# Show the top 10 rows of the dataset
if st.checkbox('Show raw data'):
    st.write(data.head(10))


# display descriptive statistics of the dataset
if st.checkbox('Display statistics on dataset'):
    st.write("Descriptive Statistics of the dataset:")
    st.write(train.describe())

# display missing data information
st.title("Data Pre-Processing")
st.checkbox("Missing Data Information:")
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()*100/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
st.write(missing_data.head(10))

#Imputing nan for Purchase column in test data
test['Purchase']=np.nan
train['Type of data']='Train'
test['Type of data']='Test'
test=test[train.columns]
data=pd.concat([train,test],axis=0)
data.shape
data.head()
data.drop('Product_Category_3',axis=1,inplace=True)

#imputed missing values with random values in the same probability distribution as given feature already had
vc = data.Product_Category_2.value_counts(normalize = True)
miss = data.Product_Category_2.isna()
data.loc[miss, 'Product_Category_2'] = np.random.choice(vc.index, size = miss.sum(), p = vc.values)
st.checkbox("Data Cleaning:")
st.write(data.isna().sum())

#using the train data part from combined dataset for eda
train = data[data['Type of data']=='Train']


st.title("Exploratory Data Analysis:")
# plot distribution of Purchase variable
if st.button('Purchase Distribution analysis'):
    st.sidebar.header('Visualizations')
    # Create a slider for the figure size
    figsize = st.sidebar.slider('Figure size', 5, 30, 17, 1)
    # Create the figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[figsize,7])
    sns.set(style="whitegrid")
    # Plot the boxplot
    sns.boxplot(x = train["Purchase"],ax=ax[0]).set_title("Purchase boxplot", fontsize=18)
    # Plot the violinplot
    sns.violinplot(train["Purchase"],ax=ax[1]).set_title("Purchase violinplot", fontsize=18)
    # Show the figure
    st.pyplot(fig)



if st.button('Gender Distribution analysis'):
    sns.catplot(x='Gender', kind='count', data=train)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


if st.button('Marital status Analysis'):
    marital_purchase = train.groupby("Marital_Status").mean()["Purchase"]
    fig = plt.figure(figsize=(8,5))
    sns.barplot(x=marital_purchase.index, y=marital_purchase.values)
    plt.title("Marital_Status and Purchase Analysis")
    plt.xlabel("Marital_Status")
    plt.ylabel("Average Purchase")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

if st.button('Occupation Analysis'):
    plt.figure(figsize=(8,6))
    sns.countplot(x ='Occupation', data = train)
    plt.title('Occupation Count Plot')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


if st.button('City Analysis'):
    st.write('## City Category Count Plot')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='City_Category', data=train, ax=ax)
    st.pyplot(fig)

if st.button('City and Purchase Analysis'):
    fig, ax = plt.subplots()
    train.groupby("City_Category").mean()["Purchase"].plot(kind='bar', ax=ax)
    ax.set_title("City Category and Purchase Analysis")
    # Display the plot in Streamlit
    st.pyplot(fig)

train['Age'] = train['Age'].apply(lambda x: np.mean([int(i.replace('+','')) for i in x.split('-')]))


if st.button('Age and purchase'):  
    data1 = train.groupby('Age')['Purchase'].mean()
    # Create a line plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(data1.index, data1.values, marker='o', color='g')
    # Add labels and title
    ax.set_xlabel('Age group')
    ax.set_ylabel('Average_Purchase amount in $')
    ax.set_title('Age group vs average amount spent')
    # Show plot using Streamlit
    st.pyplot(fig)



st.title("Building a model:")

train = data[data['Type of data']=='Train']
train.drop(['Type of data'], axis=1, inplace=True)
test = data[data['Type of data']=='Test']
test.drop(['Purchase', 'Type of data'], axis=1, inplace=True)
# displaying the output DataFrame
output = pd.DataFrame()
output['User_ID'] = test['User_ID']
output['Product_ID'] = test['Product_ID']
output['Purchase'] = np.nan
st.write(output)


#splitting the data back into train and test as it was already provided
train = data[data['Type of data']=='Train']
del train['Type of data']
test = data[data['Type of data']=='Test']
test.drop(['Purchase','Type of data'],axis=1,inplace=True)

del data

#splitting the data into X and y
X = train.drop('Purchase',axis=1)
y = train['Purchase']

#train test split for model building
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

# Display the dataframes
st.write('**Train data**')
st.write(train)

st.write('**Test data**')
st.write(test)

st.write('**X_train data**')
st.write(X_train)

st.write('**y_train data**')
st.write(y_train)

st.write('**X_test data**')
st.write(X_test)

st.write('**y_test data**')
st.write(y_test)


from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


st.title("Machine Learning models that displays the RMSE")

if st.checkbox('Liner Regression Model'):
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    lr = LinearRegression()
    lr.fit(X_train,y_train) # training the algorithm
    # Getting the coefficients and intercept
    #st.write('Coefficients:', lr.coef_)
    #st.write('Intercept:', lr.intercept_)
    #Predicting on the test data
    y_pred = lr.predict(X_test)
    st.write('r2_score:', metrics.r2_score(y_test,y_pred)) 
    st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

if st.checkbox('Decision Tree'):
    # Decision Tree Model
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
    DT.fit(X_train, y_train)
    y_pred = DT.predict(X_test)
    st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

if st.checkbox('Random Forest Tree'):
    # Defining the function to train the Random Forest model
    def train_rf(X_train, y_train, n_estimators, max_depth):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=3)
        rf.fit(X_train, y_train)
        return rf
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    rf = train_rf(X_train, y_train, n_estimators=25, max_depth=10)
    # Making predictions on the test set
    y_pred = rf.predict(X_test)
    # Computing the evaluation metrics
    r2 = metrics.r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Displaying the evaluation metrics and the predictions using Streamlit
   # st.write("r2_score: {r2}")
    #st.write("rmse: {rmse}")
    #st.write("Predictions: {y_pred}")
    st.write('r2_score:', metrics.r2_score(y_test, y_pred))
    st.write('rmse:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from xgboost import XGBRegressor
if st.checkbox('XGBoost Regressor'):
    xgb2 = XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.05)
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    xgb2.fit(X_train, y_train)
    y_pred = xgb2.predict(X_test)
    st.write('XGBoost Regression Model')
    st.write('r2_score:', metrics.r2_score(y_test, y_pred))
    st.write('rmse:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.neighbors import KNeighborsClassifier# Create KNN classifier
if st.checkbox('KNN Classifier'):
    knn = KNeighborsClassifier(n_neighbors = 3)
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    # Fit the classifier to the data
    knn.fit(X_train,y_train)
    # Predicting on the test data
    y_pred = knn.predict(X_test)
    # Metrics
    st.write('r2_score:', metrics.r2_score(y_test,y_pred)) 
    st.write('rmse:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))