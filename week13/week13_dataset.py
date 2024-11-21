from unittest.mock import inplace

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor  # 적용모델 : K 최근접 이웃 회귀 모델
from sklearn.model_selection import train_test_split  # 훈련 / 검증 셋트 분할 함수

mpg = sns.load_dataset('mpg') # 데이터셋 로딩

# 데이터 전처리
mpg.drop(['name'], axis=1, inplace=True) # 불필요한 칼럼 삭제 (원본 데이터 변경)
mpg.dropna(inplace=True) # 결측치가 있는 행 제거 (원본 데이터 변경)
mpg = pd.get_dummies(mpg, columns=['origin'], drop_first=True) # One-hot encoding
#print(mpg.info())

# 독립변수, 의존변수
X = mpg.drop(['mpg'], axis=1) # 레이블 컬럼 제거
y = mpg['mpg'] # 레이블 (타겟 변수)

print(X.dtypes) # 각 열의 데이터 타입 확인