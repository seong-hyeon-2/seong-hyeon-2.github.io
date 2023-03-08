---
layout: post
title:  "모각소 2주차 - Kaggle Titanic - 완성본"
---

# Titanic - Machine Learning from Disaster

![Untitled](https://user-images.githubusercontent.com/114178570/214028943-a48a9073-bf99-4612-b345-0794d409f9de.png)


[https://www.kaggle.com/competitions/titanic/overview](https://www.kaggle.com/competitions/titanic/overview)

## 문제

2224 중에 1502명의 사망자를 낳았다. csv를 살펴보면 어느 집단의 사람들은 다른 사람들에 비해 생존율이 더욱 높다.  

→ 주어진 데이터(Sex, Name, Pclass 등)를 이용해서 어떤 부류의 사람들이 좀 더 생존할 가능성이 높은지 찾아라.

Classification(분류화)문제 

## 개요

1. 데이터 정보 살펴보기(columns.value - ex) Pclass, Sex, Name..)
2. 각 feature의 생존율 살펴보기
3. 결손값 채우기 
4. train_df, test_df에서 필요없는 feature를 drop하기
5. 모델 돌리기 
6. csv 파일 제출 

## Data

캐글에서 제공하는 csv 파일은 3가지가 있다. 

- train.csv(1~891명)
    
    → 훈련용 데이터 - 모델을 학습시키는 데이터 
    
- test.csv(892~892+418명)
    
    → 학습된 모델에 넣을 데이터 - train에는 ‘Survived’ feature가 있지만, test에는 ‘Survived’ feature가 없음
    
- gender submisson → test.csv 사람들 ‘Survived’ feature만 있고 여성만 생존한다고 가정해 놓은 파일 → 예시임
    
    → 이 파일을 제출용 파일로 사용하면 됨  
    

### Data feature

data feature에는 10가지 특징들이 있다. 

```python
# 데이터 변수 확인 
train_df.columns.values

['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']

test_df.columns.values
['PassengerId' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare'
 'Cabin' 'Embarked']
```

- Survived - 생존 여부 (0 = 사망, 1 = 생존)
- Pclass - 티켓 클래스 (1 = 1등석, 2 = 2등석, 3 = 3등석)
- Sex - 성별
- Age - 나이
- SibSp - 함께 탑승한 자녀 / 배우자 의 수
- Parch - 함께 탑승한 부모님 / 아이들 의 수
- Ticket - 티켓 번호
- Fare - 탑승 요금
- Cabin - 수하물 번호
- Embarked - 선착장 (C = Cherbourg, Q = Queenstown, S = Southampton)

```python
# 각 열의 정보를 확인. 결손값, 타입 등 
train_df.info()
test_df.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
```

Age , Cabin, Embarked, Fare 특징들에서 결손값이 존재하는 것을 알 수 있다. 

- Age는 평균 값으로 null 값으로 채움
- Cabin의 null 값은 굉장히 많아 이 feature는 drop.
- Fare null 값 평균으로 채움
- Embarked의 null 값은 뒤에서 살펴보겠지만, 최빈값인 ‘S’로 채움

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
None
```

## Data Visualization

이제는 각 featare들의 생존율 시각화를 통해 살펴보도록 하자

```python
# train_df의 ‘Survived’ 살펴보기
sns.countplot(train_df['Survived'])
train_df['Survived'].value_counts()

0(dead)        : 549
1(survived)    : 342
```

cf)

 데이터별로 이런 타겟 변수가 한쪽의 값만 지나치게 많은 경우, 우리는 이를 'Class imbalanced problem' 이라고 부른다. 카드 사기 거래 여부, 하늘에서 운석이 떨어지는 경우를 예를 들어 보면 사기가 아닌 경우, 운석이 떨어지지 않을 경우가 반대의 경우보다 그 수가 압도적으로 많다. 이때 별다른 처리 없이 머신러닝에게 데이터를 학습시킨다면, 머신러닝이 모든 데이터를 0(사망, 혹은 정상거래, 혹은 운석 안떨어짐) 이라고 예측할지도 모른다. 그리고 이 정확도를 보면, 상당히 높게 나온다. 실제로 우리가 예측하려는 데이터에도 운석이 떨어지는 날은 몇일 되지 않을 것이니까. 하지만 이 녀석은 참으로 의미없는 머신러닝 모델이 될 것이다. 

 실제로 우리나라 일기예보도 '365일 비 안옴' 이라고 예측하면 정확도가 75% 정도 된다고 하지만, 이것이 의미있는 예측은 아니지 않은가. 이럴때는 여러 방법들을 통해 이 **불균형을 해결한 후 머신러닝 알고리즘으로 학습을 시켜야 의미있는 예측을 하는 경우가 대부분**이다. 그리고 타이타닉 데이터에 있는 이 불균형정도면, 상당히 양호한 편이라고 할 수 있다.

참조조: [https://jamm-notnull.tistory.com/11](https://jamm-notnull.tistory.com/11)

```python
# 나이에 따른 생존율

# 열(col)을 생존 여부로 나눔
 # FacetGrid -> Multy plot 하나의 데이터를 여러 개의 plot으로 나눠서 보고자 할 때 
g = sns.FacetGrid(train_df, col='Survived')

# 히스토그램으로 시각화, 연령의 분포를 확인, 히스토그램 bin을 20개로 설정
g.map(plt.hist, 'Age', bins=20)
```

```python
# Pclass(등석) 즉, 빈부격차에 따른 생존율

# x축은 Pclass, y축은 Survived, Sex로 나눠서 본다. 
sns.barplot(x='Pclass', y='Survived', hue='Sex',data=train_df)
```

데이터를 살펴보면 확연히 1등석에 탑승한 승객이 생존율이 더 높은 것을 볼 수 있다. 

또한 female의 생존율도 확연히 높다. 

![Untitled 1](https://user-images.githubusercontent.com/114178570/214028993-82095a61-2225-4754-b74b-0e889789e162.png)

![Untitled 2](https://user-images.githubusercontent.com/114178570/214029017-776d6199-2aac-4cf1-8d9a-4526bd2a5de8.png)

![Untitled 3](https://user-images.githubusercontent.com/114178570/214029057-0c5a7b62-195e-49ff-b178-95308b34d99d.png)


```python
# 각 특징에 따른 생존율

def bar_chart(feature):
    # 1 - 생존 O
    survived = train_df[train_df['Survived']==1][feature].value_counts()
    # 0 - 생존 X
    dead = train_df[train_df['Survived']==0][feature].value_counts()
		# 생존율
    per = survived / (survived + dead)
		# 새로운 데이터프레임 생성
    df = pd.DataFrame([survived, dead], index = ['survived','dead'])
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.title(feature)
		# x 글자 가로로 보이게
    plt.xticks(rotation=0)
    plt.show()
    print(df)
    print()
    print('생존 확률\n', per)
```

```python
# 성별에 따른 생존율

bar_chart("Sex")

female  male
survived     233   109
dead          81   468

생존 확률
 female    0.742038
male      0.188908
Name: Sex, dtype: float64
```

‘Sex’에서 female의 생존율이 male에 비해 훨씬 높다. 

```
# 동반한 Sibling(형제자매)와 Spouse(배우자)의 수에 따른 생존율 

bar_chart('SibSp')

0      1     2     3     4    5    8
survived  210.0  112.0  13.0   4.0   3.0  NaN  NaN
dead      398.0   97.0  15.0  12.0  15.0  5.0  7.0

생존 확률
 0    0.345395
1    0.535885
2    0.464286
3    0.250000
4    0.166667
5         NaN
8         NaN
Name: SibSp, dtype: float64
```

```
# 동반한 Parent(부모) Child(자식)의 수에 따른 생존율

bar_chart('Parch')

0     1     2    3    4    5    6
survived  233.0  65.0  40.0  3.0  NaN  1.0  NaN
dead      445.0  53.0  40.0  2.0  4.0  4.0  1.0

생존 확률
 0    0.343658
1    0.550847
2    0.500000
3    0.600000
4         NaN
5    0.200000
6         NaN
Name: Parch, dtype: float64
```

SibSp와 Parch에서 보면 각각 1명, 3명일 때 생존확률이 높은 것으로 보인다. 

```python
# 어디서 탑승했는지에 따른 생존율

bar_chart('Embarked')

S   C   Q
survived  217  93  30
dead      427  75  47

생존 확률
 S    0.336957
C    0.553571
Q    0.389610
Name: Embarked, dtype: float64
```

선착장 S에서 제일 많이 탑승하였고 C에서 탑승한 사람이 생존 확률이 높음

![Untitled 4](https://user-images.githubusercontent.com/114178570/214029084-5f5fa587-ad45-418a-84c1-5c3c83761d2c.png)

![Untitled 5](https://user-images.githubusercontent.com/114178570/214029126-f96f47d5-2303-458b-8970-345573a17206.png)

![Untitled 6](https://user-images.githubusercontent.com/114178570/214029153-ead44b50-a52f-4a47-a5f8-d063ed422ed4.png)

![Untitled 7](https://user-images.githubusercontent.com/114178570/214029171-c8c60d37-8eea-4c23-89f0-fdd170b5d52a.png)

시각화를 통해 각 feature들의 생존율을 살펴보았다. 

살펴본 바 ‘Sex’, ‘Pclass’, ‘Embarked’, ‘Age’가 ‘Survived’에 큰 영향을 줄 것으로 예상한다.

## 데이터 전처리 및 특징 추출

이 과정에서는 

1. 결손값이 있는 feature에 값을 채워줌
2. 범위가 넓은 값들은 큰 범주로 묶음
3. 데이터를 모델에 넣기 위해서 원핫인코딩 과정을 실행

**Name** 

: Mister의 약자로 남자에게 붙이는 칭호. 반드시 점(.)을 반드시 찍어야함.

Ms.(미즈)

: 결혼 여부에 관계없이 여성의 이름이나 성에 붙이는 칭호

Miss(미스) 

: 결혼하지 않은 여성에게 붙이는 칭호

Mrs.(미세스) 

: Mistress의 약자로 결혼한 여성에게 사용되는 칭호

Dr.(닥터)

: Doctor의 약자로 박사

### NAME

```python
# Name에서 title에 따라 분류 
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer','Lady','Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', "Miss")
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

# Master : 0
# Miss : 1
# Mr. : 2
# Mrs : 3
# other : 4

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace('Master', 0)
    dataset['Title'] = dataset['Title'].replace("Miss", 1)
    dataset['Title'] = dataset['Title'].replace('Mr', 2)
    dataset['Title'] = dataset['Title'].replace('Mrs', 3)
    dataset['Title'] = dataset['Title'].replace('Other', 4)

# groupby() 메서드는 데이터를 특정 기준으로 그룹화하여 처리
# 밑에 코드는 Title을 기준으로 그룹화하여 생존율을 평균낸 것
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```

### SEX

![Untitled 8](https://user-images.githubusercontent.com/114178570/214029198-9a4c9cc5-c1b8-4250-9e63-d51fd46e35c9.png)

```python
# Sex 특징 처리. 이 또한 숫자로 변형한다.
# male : 1
# female : 0

for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].replace('female', 0)
    dataset['Sex'] = dataset['Sex'].replace("male", 1)

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
```

### AGE

```python
# Age 결측치 채우기 - 평균값으로 채우겠다.
# 나이 범주 파악

for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

    

print (train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
```

Age 범주 파악 결과

0 ~ 16 : Child - 0

16 ~ 32 : Young - 1

32 ~ 48 : Middle - 2

48 ~ 64 : Prime - 3

64 ~ 80 : Old - 4

이렇게 범주가 생성되었다.

이것을 또 간략하게 숫자로 바꾸어주겠다.

![Untitled 9](https://user-images.githubusercontent.com/114178570/214029226-239741f9-a778-4e48-b49d-97230694e725.png)

![Untitled 10](https://user-images.githubusercontent.com/114178570/214029244-126f3231-3caf-457f-bab0-08ec5abb8bd2.png)

```python
#  나이 범주를 숫자로 바꿔주기
# 0 ~ 16 : Child - 0
# 16 ~ 32 : Young - 1
# 32 ~ 48 : Middle - 2
# 48 ~ 64 : Prime - 3
# 64 ~ 80 : Old - 4

for dataset in train_and_test:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    # dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)

print (train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()) 
print(train_df['Age'].head(5))
```

### Embarked

```python
# Embarked 결손값 최빈값(S)로 채움
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Embarked 3가지 범주로 변환
# S : 0
# C : 1
# Q : 2
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
```

![Untitled 11](https://user-images.githubusercontent.com/114178570/214029273-014f8dd0-16f4-41e7-af28-8f24738b70d0.png)

### Fare

```python
# Fare의 결손값을 평균값으로 채움
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

# 요금 4범주로 나눔
train_df['FareBand'] = pd.cut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# 요금 범주
# 0 ~ 7.91 : 0
# 7.91 ~ 14.454 : 1
# 14.454 ~ 31 : 2
# 31~ : 3

for dataset in train_and_test:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
train_and_test = [train_df, test_df]
```

이제 필요 없는 데이터(Name, Cabin, Ticket)는 Drop out한다.

```python
# train_df와 test_df 필요없는 feature를 지우고 같은 feature를 가지도록 만들어줌
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'AgeBand'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
```

## **모델에 넣을 데이터 준비**

```python
# 목적 변수 제거
X_train = train_df.drop("Survived", axis=1)

#목적 변수 역할
Y_train = train_df["Survived"]

#예측 대상 데이터 셋
X_test  = test_df
X_train.shape, Y_train.shape, X_test.shape
```

## 각 모델의 정확도 확인

```python
# model 준비
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```

### **Logistic Regression**

```python
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

# 정확도 : 79.46

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
```

![Untitled 12](https://user-images.githubusercontent.com/114178570/214029310-fac13fe5-f804-4209-84d7-353d422a8f66.png)

### **SVC(Support Vector Machines)**

```python
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

# 정확도 : 83.61
```

### **K-NN(K Nearest Neighberhood)**

```python
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# 정확도 : 84.62
```

### **Stochastic Gradient Descent**

```python
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# 정확도 : 75.87
```

### **Decision Tree**

```python
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# 정확도 : 89.23
```

### **Random Forest**

```python
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

# 정확도 : 89.23
```

### **모델 별 정확도 비교**

```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent',  'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_sgd, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

```

![Untitled 13](https://user-images.githubusercontent.com/114178570/214029352-8b80d7be-de5a-4c4b-8f67-c454d0e9686f.png)

Random forest와 Decision Tree는 overfitting 가능성이 있을 것이라 추측이 들어 

KNN이 제일 좋은 성능을 좋을 것이라 결론 지음.

```python
# 파일 제출 
submission = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/gender_submission.csv')

# 모델 성능이 높았던 KNN으로 제출 
 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

prediction = Y_pred
submission['Survived'] = prediction

submission.head(10)
```

## 해결 방법

[[Subinium Tutorial] Titanic (Beginner)](https://www.kaggle.com/code/subinium/subinium-tutorial-titanic-beginner/notebook)

몇몇 개념 

 **One-hot-Encoding(원핫 인코딩)**

: 데이터를 One-hot 데이터 형태로 변형 / 압축하는 것

→ 데이터는 이미지 데이터, 텍스트 데이터, 카테고리 데이터가 될 수도 있음

→ 복잡한 데이터를 압축해서 전송하는 것이 훨씬 빠르고 안정적임. 컴퓨터가 처리하기 쉽게 숫자로 변형해 줌.

→ 특히 범주형 데이터를 원핫 인코딩해서 사용하는 경우가 많음. 텍스트로 되어있는 값들을 그대로 모델의 입력값으로 사용할 수 없기 때문에 숫자로 바꿔주는 것이 필요!! 

cf) 

encoding(인코딩) 

: 컴퓨터를 이용해 영상 - 이미지 - 소리 데이터를 생성할 때 데이터의 양을 줄이기 위해 데이터를 코드화하고 압축하는 것 

decoding(디코딩) 

: 압축/변형된 데이터를 원형으로 변환하는 것

**서포트 벡터 머신(SVM : Support Vector Machine)**

: 두 클래스로부터 최대한 멀리 떨어져 있는 **결정 경계**
를 찾는 분류기로 특정 조건을 만족하는 동시에 클래스를 분류하는 것을 목표로 함. → 두 클래스들과 선 사이의 거리가 최대로 떨어져 있어야 즉, 마진이 커야 잘 분류되었다고 간주한다. 거리 = 유사도 

→ 분류와 회귀 문제에서 모두 사용 가능하다. 분류에서는 소프트 마진(마진이 클수록 좋기 때문), 회귀에서는 하드 마진(마진이 좁을수록 데이터들을 대표할 수 있는 회귀선을 잘 만들 수 있기 때문)을 사용하는 것이 좋다. 

![Untitled 14](https://user-images.githubusercontent.com/114178570/214029387-21436add-6b2b-4f2f-99ef-3a9a21d8977b.png)


그럼 나는 어떤 예측 모델을 이용해서 이 문제를 해결할 수 있을까?

지금 생각나는 방법은 **Logistic regression**이 제일 먼저 생각난다. 

→ 어떤 사건(event)이 발생할지에 대한 직접 예측이 아니라 그 사건이 발생할 확률을 예측하는 것

[https://idkim97.github.io/machine learning/MachineLearning_LogisticRegression/](https://idkim97.github.io/machine%20learning/MachineLearning_LogisticRegression/)

Logistic regression은 종속변수 Y가 범주형이면서 0 OR 1의 값을 가질 때(즉,Binary한 데이터를 처리할 때) 사용하는 것이 유용하다. 선형회귀는 이 같은 경우에서는 fitting하기 어려움 

<img width="961" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-01-18_00 19 37" src="https://user-images.githubusercontent.com/114178570/214029420-aa61922f-547f-48a6-a6f6-4c898b0c3f95.png">

[gender_submission.csv](https://github.com/seong-hyeon-2/seong-hyeon-2.github.io/files/10479198/gender_submission.csv)
