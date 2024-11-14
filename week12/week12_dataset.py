import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.common import random_state
from scipy.ndimage import label
# from fontTools.subset import subset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

titanic = sns.load_dataset('titanic')
median_age = titanic['age'].median()
titanic_fill_row = titanic.fillna({'age' : median_age})

X = titanic_fill_row[['age']]
y = titanic_fill_row[['survived']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(3, 2))
plt.scatter(X_test,y_test, color='blue',label='Real')
plt.title('Linear Regression : Real vs Predict')
plt.xlabel('Age')
plt.ylabel('Survivied')
plt.show()

titanic_fill_row['survived'] = titanic_fill_row['survived'].astype(float)
print(titanic, median_age,titanic_fill_row)