import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

lifesat = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# print(lifesat.head(27))

lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

#model = LinearRegression()  # 선형 회귀 모델 적용
model = KNeighborsRegressor(n_neighbors=3)  # K 최근접 이웃 모델 적용, 한국과 1인당 GDP가 가장 가까운 나라 셋

model.fit(X, y)

X_new = [[32_142.0]]  # 대한민국 1인당 GDP 2022년 자료
print(model.predict(X_new))