# importing the necessary libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as seabornInstance

# extracting data from the csv file


d = pd.read_csv(r'C:\Users\HP\Desktop\incd_edited.csv', encoding="utf-8", encoding_errors="ignore")

# plotting data points on a 2-D graph

d.plot(x='Lower 95% Confidence Interval', y= 'Upper 95% Confidence Interval', style='o')
plt.title('Lower 95% Confidence Interval vs Upper 95% Confidence Interval')
plt.style.use('seaborn-whitegrid')
plt.xlabel('Lower 95% Confidence Interval')
plt.ylabel('Upper 95% Confidence Interval')

# checking the average of Upper 95% Confidence Interval
seabornInstance.displot(d['Upper 95% Confidence Interval'])
plt.show()


# printing the first 20 columns
print(d.head(20))

# extracting and reshaping the needed data columns
x = d['Lower 95% Confidence Interval'].values.reshape(-1,1)
y = d['Upper 95% Confidence Interval'].values.reshape(-1,1)
print(x)
print(y)


# splitting the datas between the train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# training the algorithm
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# retrieving the intercept
print(regressor.intercept_)

# retrieving the slope
print(regressor.coef_)

# making a prediction
y_pred = regressor.predict(x_test)


df = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted' : y_pred.flatten()})
print(df)
