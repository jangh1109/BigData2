import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor  # 적용모델 : K 최근접 이웃 회귀 모델
from sklearn.model_selection import train_test_split  # 훈련 / 검증 셋트 분할 함수

mpg = sns.load_dataset('mpg')
print(mpg.info())
mpg.dropna(inplace=True) # 결측치가 있는 행 제거 (원본 데이터 변경)
print(mpg.info())
print(mpg.head())
print(mpg.tail())
print(mpg[['origin']])
print(mpg.describe())