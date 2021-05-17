# Predict the car price using the features given in the car data-dataset.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import  RandomizedSearchCV
import pickle

df = pd.read_csv('car data.csv')
df.head()
df.shape
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())

#check missing or null values

df.isnull().sum()
# Removing the car_name, it is not a useful feature to predict the car price.
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
                    'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'
                    ]]
final_dataset.head()
# creating a new feature
final_dataset['Current_year']=2020
final_dataset['no_year'] = final_dataset['Current_year']-final_dataset['Year']
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.drop(['Current_year'],axis=1,inplace=True)
# converting categorical features into one-hot encoded

final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.corr()

sns.pairplot(final_dataset)
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
plt.show()
g = sns.heatmap(final_dataset[top_corr_features].corr(),annot=True, cmap='RdYlGn')
plt.show()

X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]

X.head()
y.head()

# Feature importance

model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train.shape
rf_random = RandomForestRegressor()
# Hyperparameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num=12)]
print(n_estimators)

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num=12)]
# Number of features to consider at every split
max_features = ['auto','sqrt']
# Maximum number of levels in Tress
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2,5,10,15,100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]
#RandomizedSearchCV is faster than RandomForestRegressor
#RandomizedSearchCV will find the best parameters from the random_grid.
random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               scoring='neg_mean_squared_error',
                               n_iter=10,
                               cv=5,
                               verbose=2,
                               random_state=42,
                               n_jobs = 1)
rf_random.fit(X_train,y_train)
predictions = rf_random.predict(X_test)
sns.distplot(y_test-predictions) # The difference is giving us a normal distribution which is a sign that model is working good.
plt.show()

plt.scatter(y_test,predictions)
plt.show()

# open a file, where you ant to store the data. We will use the pickle file for deployment.
file = open('random_forest_regression_modelv2.pkl','wb')
# dump information to that file
pickle.dump(rf_random,file)