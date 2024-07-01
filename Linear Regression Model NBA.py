#--------------------------------------data

#Loading Data
import pandas as pd
df = pd.read_csv('nba-player-stats-2019.csv')
#Data Preparation
y = df['PTS']
x = df.drop(['PTS','Player', 'Tm', 'Pos'], axis=1)

#Data Splitting
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state= 100)

#--------------------------------------linear regression

#training the model and applying it to make a prediction
from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(xtrain, ytrain)
lr_trainpred = lr.predict(xtrain)
lr_testpred = lr.predict(xtest) 

#evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score
lr_trainmse = mean_squared_error(ytrain,lr_trainpred)
lr_trainr2 = r2_score(ytrain, lr_trainpred)

lr_testmse = mean_squared_error(ytest,lr_testpred)
lr_testr2 = r2_score(ytest, lr_testpred)
lr_results = pd.DataFrame(['Linear Regression', lr_trainmse, lr_trainr2, lr_testmse, lr_testr2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(x=ytrain, y=lr_trainpred, alpha = 0.3)

plt.plot()
plt.ylabel('Predict Pts')
plt.xlabel('Experimental Pts')