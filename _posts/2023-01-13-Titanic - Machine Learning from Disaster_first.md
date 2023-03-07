---
layout: post
title:  "모각소 - Kaggle Titanic"
---


# Titanic - Machine Learning from Disaster

![Untitled](https://user-images.githubusercontent.com/114178570/212255132-8ba03ad5-bb4c-40a2-a7c8-140f428a2f7d.png)


[https://www.kaggle.com/competitions/titanic/overview](https://www.kaggle.com/competitions/titanic/overview)

## 문제

2224 중에 1502명의 사망자를 낳았다. 보면 어느 집단의 사람들은 다른 사람들에 비해 좀 더 잘 살아남았다. 

→ 주어진 데이터(gender, name, socia-economic 등)를 이용해서 어떤 부류의 사람들이 좀 더 생존할 가능성이 높은지 찾아라

## Data

train.csv(1~891명)

test.csv(892~892+418명)

→ train에는 survived가 있지만, test에는 survived가 없음

gender submisson → test.csv 사람들 survived만 있음

→ 훈련데이터에서 학습하고 test에서 예측하고 gender submisson과 비교해서 정확도 분석하는 것. 

1. survived : 생존 유무
2. Pclass : 객실 등급
3. Sex : 성별
4. Age : 나이
5. SibSp : 동반한 Sibling(형제자매)와 Spouse(배우자)의 수
6. Parch : 동반한 Parent(부모) Child(자식)의 수
7. Fare : 요금
8. Cabin : 객실 번호
9. Embarked : 어디서 탑승했는지

## 해결 방법

[[Subinium Tutorial] Titanic (Beginner)](https://www.kaggle.com/code/subinium/subinium-tutorial-titanic-beginner/notebook)

참고 블로그에서 나오는 몇몇 개념 

encoding(인코딩) 

: 컴퓨터를 이용해 영상 - 이미지 - 소리 데이터를 생성할 때 데이터의 양을 줄이기 위해 데이터를 코드화하고 압축하는 것 

decoding(디코딩) 

: 압축/변형된 데이터를 원형으로 변환하는 것

 **One-hot-Encoding(원핫 인코딩)**

: 데이터를 One-hot 데이터 형태로 변형 / 압축하는 것

→ 데이터는 이미지 데이터, 텍스트 데이터, 카테고리 데이터가 될 수도 있음

→ 복잡한 데이터를 압축해서 전송하는 것이 훨씬 빠르고 안정적임. 컴퓨터가 처리하기 쉽게 숫자로 변형해 줌.

→ 특히 범주형 데이터를 원핫 인코딩해서 사용하는 경우가 많음. 텍스트로 되어있는 값들을 그대로 모델의 입력값으로 사용할 수 없기 때문에 숫자로 바꿔주는 것이 필요!! 

서포트 벡터 머신(SVM : Support Vector Machine)

: 두 클래스로부터 최대한 멀리 떨어져 있는 **결정 경계**
를 찾는 분류기로 특정 조건을 만족하는 동시에 클래스를 분류하는 것을 목표로 함. → 두 클래스들과 선 사이의 거리가 최대로 떨어져 있어야 즉, 마진이 커야 잘 분류되었다고 간주한다. 거리 = 유사도 

→ 분류와 회귀 문제에서 모두 사용 가능하다. 분류에서는 소프트 마진(마진이 클수록 좋기 때문), 회귀에서는 하드 마진(마진이 좁을수록 데이터들을 대표할 수 있는 회귀선을 잘 만들 수 있기 때문)을 사용하는 것이 좋다. 

![Untitled 1](https://user-images.githubusercontent.com/114178570/212255203-7682f111-d530-4ffe-8523-2804d2bad351.png)


그럼 나는 어떤 예측 모델을 이용해서 이 문제를 해결할 수 있을까?

지금 생각나는 방법은 **Logistic regression**이 제일 먼저 생각난다. 

→ 어떤 사건(event)이 발생할지에 대한 직접 예측이 아니라 그 사건이 발생할 확률을 예측하는 것

[https://idkim97.github.io/machine learning/MachineLearning_LogisticRegression/](https://idkim97.github.io/machine%20learning/MachineLearning_LogisticRegression/)

Logistic regression은 종속변수 Y가 범주형이면서 0 OR 1의 값을 가질 때(즉,Binary한 데이터를 처리할 때) 사용하는 것이 유용하다. 선형회귀는 이 같은 경우에서는 fitting하기 어려움
