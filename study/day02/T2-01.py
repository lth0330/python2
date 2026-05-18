import pandas as pd
df = pd.read_csv('study/day01/fish.csv')


#[2] Perch(농어)만 추출
target_fish = df[df['Species'].isin(['Perch'])]
target_fish.info()

# 농어의 길이 / 무게 추출
perch_length = target_fish['Length2'].values
perch_weight = target_fish['Weight'].values
# print(perch_length, perch_weight)

# '농어' 길이에 따른 무게 예측 
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.show()

# [3] 학습 모델 만들기 , 준비 : 학습용과 테스트용 분리한다. 애? 모델평가에 사용된다.
from sklearn.model_selection import train_test_split
# train_test_split(학습자료, 정답자료?, test_size=분리비율 , random_state=분리기준난수)
# random_state = 분리할 때 사용되는 난수값 , # 난수값에 따라 분리한다. # 고정값 넣어주면 항상 동일한 분리값 넣을 수 있다.
train_input , test_input, train_target, test_target = train_test_split(perch_length, perch_weight , test_size=0.3 , random_state=42)

# (2) (사이킷런)모델 학습 , 대부분 2차원 사용한다. 
import numpy as np
array = np.array([1,2,3,4])
print(array.shape)  # shape : 배열의 모양을 반환 , (4,)

array2 = np.array([[1,2], [3,4],[5,6]]) 
print(array2.shape)   # (3,2)

# print(train_input.shape) # (39,) 1차원 배열 --> 사이킷런 모델들을 1차원배열 학습이 불가능하다.
# print(train_input)       # 1차원으로 구성된 '농어' 길이
# [17.4 36.  25.  40.]
# T1-01(zip활용) , T1-02(column_stack활용) , T2-01(reshape) : 1차원 -> 2차원

# [4]
# .reshape(행개수, 열개수)   :  행개수에는 -1 넣어서 자동 뜻 ( 자료개수만 자동) , 열개수는 1개 
train_input = train_input.reshape(-1,1)
# print(train_input) # [[17.4][36. ][25. ]]
# print(train_input.shape) # (39,1) 2차원 배열 
test_input = test_input.reshape(-1,1)  # 테스트용 학습 2차원 변환

# train_target = train_target.reshape(-1,1)
# test_target = test_target.reshape(-1,1)

# [5] 모델 학습
from sklearn.neighbors import KNeighborsClassifier # K최근접 이웃 찾기 
from sklearn.neighbors import KNeighborsRegressor # K최근접 이웃 회귀 

knr = KNeighborsRegressor() # 모델 객체 생성
knr.fit(train_input, train_target) # 모델 학습 # (길이 , 무게) # '길이'에 따른 '무게' 학습 
print(knr.score(test_input,test_target)) # 모델 평가  0.99292
print(knr.predict(test_input))           # 모델 예측

print(test_input)  [8.4] # [ 8.4][18. ][27.5 [21.3] ~ 
print(knr.predict(test_input))  # [  61.4   78.   248.   117.  ] ~ 