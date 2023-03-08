# 서울시 따릉이 대여량 예측


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 1. 데이터 확인하기


```python
# 데이터 불러오기
train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ttareungyi/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ttareungyi/test.csv')
```


```python
# 데이터 개수 확인
print(train_df.shape)
print(test_df.shape)
```

    (1459, 11)
    (715, 10)



```python
# feature 값 확인 
print(train_df.columns)
print('-'*100)
print(test_df.columns)
```

    Index(['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
           'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
           'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
          dtype='object')
    ----------------------------------------------------------------------------------------------------
    Index(['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
           'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
           'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5'],
          dtype='object')


 서울시 마포구의 날짜별, 시간별 기상상황과 따릉이 대여 수 데이터


- id 고유 id
- hour 시간 - 시간대
- temperature 기온
- precipitation 비가 오지 않았으면 0, 비가 오면 1
- windspeed 풍속(평균)
- humidity 습도
- visibility 시정(視程), 시계(視界)(특정 기상 상태에 따른 가시성을 의미)
- ozone 오존
- pm10 미세먼지(머리카락 굵기의 1/5에서 1/7 크기의 미세먼지)
- pm2.5 미세먼지(머리카락 굵기의 1/20에서 1/30 크기의 미세먼지)
- count 시간에 따른 따릉이 대여 수


ozone, pm10, pm2.5는 높을수록 안 좋다

train 데이터의 count를 제외하고는 feature 모두 같다.

count값을 예측해야한다


```python
# 통계값 확인 
print(train_df.describe())
```

                    id         hour  hour_bef_temperature  hour_bef_precipitation  \
    count  1459.000000  1459.000000           1457.000000             1457.000000   
    mean   1105.914325    11.493489             16.717433                0.031572   
    std     631.338681     6.922790              5.239150                0.174917   
    min       3.000000     0.000000              3.100000                0.000000   
    25%     555.500000     5.500000             12.800000                0.000000   
    50%    1115.000000    11.000000             16.600000                0.000000   
    75%    1651.000000    17.500000             20.100000                0.000000   
    max    2179.000000    23.000000             30.000000                1.000000   
    
           hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  \
    count         1450.000000        1457.000000          1457.000000   
    mean             2.479034          52.231297          1405.216884   
    std              1.378265          20.370387           583.131708   
    min              0.000000           7.000000            78.000000   
    25%              1.400000          36.000000           879.000000   
    50%              2.300000          51.000000          1577.000000   
    75%              3.400000          69.000000          1994.000000   
    max              8.000000          99.000000          2000.000000   
    
           hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count  
    count     1383.000000    1369.000000     1342.000000  1459.000000  
    mean         0.039149      57.168736       30.327124   108.563400  
    std          0.019509      31.771019       14.713252    82.631733  
    min          0.003000       9.000000        8.000000     1.000000  
    25%          0.025500      36.000000       20.000000    37.000000  
    50%          0.039000      51.000000       26.000000    96.000000  
    75%          0.052000      69.000000       37.000000   150.000000  
    max          0.125000     269.000000       90.000000   431.000000  



```python
# null 값 확인
print(train_df.info())
print('-'*100)
print(test_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   id                      1459 non-null   int64  
     1   hour                    1459 non-null   int64  
     2   hour_bef_temperature    1457 non-null   float64
     3   hour_bef_precipitation  1457 non-null   float64
     4   hour_bef_windspeed      1450 non-null   float64
     5   hour_bef_humidity       1457 non-null   float64
     6   hour_bef_visibility     1457 non-null   float64
     7   hour_bef_ozone          1383 non-null   float64
     8   hour_bef_pm10           1369 non-null   float64
     9   hour_bef_pm2.5          1342 non-null   float64
     10  count                   1459 non-null   float64
    dtypes: float64(9), int64(2)
    memory usage: 125.5 KB
    None
    ----------------------------------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 715 entries, 0 to 714
    Data columns (total 10 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   id                      715 non-null    int64  
     1   hour                    715 non-null    int64  
     2   hour_bef_temperature    714 non-null    float64
     3   hour_bef_precipitation  714 non-null    float64
     4   hour_bef_windspeed      714 non-null    float64
     5   hour_bef_humidity       714 non-null    float64
     6   hour_bef_visibility     714 non-null    float64
     7   hour_bef_ozone          680 non-null    float64
     8   hour_bef_pm10           678 non-null    float64
     9   hour_bef_pm2.5          679 non-null    float64
    dtypes: float64(8), int64(2)
    memory usage: 56.0 KB
    None



```python
# 결손값 개수 
print(train_df.isnull().sum())
print('-'*100)
print(test_df.isnull().sum())
```

    id                          0
    hour                        0
    hour_bef_temperature        2
    hour_bef_precipitation      2
    hour_bef_windspeed          9
    hour_bef_humidity           2
    hour_bef_visibility         2
    hour_bef_ozone             76
    hour_bef_pm10              90
    hour_bef_pm2.5            117
    count                       0
    dtype: int64
    ----------------------------------------------------------------------------------------------------
    id                         0
    hour                       0
    hour_bef_temperature       1
    hour_bef_precipitation     1
    hour_bef_windspeed         1
    hour_bef_humidity          1
    hour_bef_visibility        1
    hour_bef_ozone            35
    hour_bef_pm10             37
    hour_bef_pm2.5            36
    dtype: int64


train, test 둘 다 id와 hour 값(count 제외)을 제외한 모든 feature에 결손값이 존재한다.


ozone, pm10, pm2.5를 제외한 나머지 데이터는 평균값으로 결손값을 채운다. 


```python
# heatmap을 통해 상관 관계 확인

plt.figure(figsize=(13,10))
g = sns.heatmap(train_df.corr(),annot=True,cmap="RdYlGn")
```


    
![png](../../../Downloads/ttareungyi/output_12_0.png)
    



![Untitled.png](data:image/png;base64,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)

 상관계수가 +_0.4 이상이면 상관관계가 있다고 봄

hour, temperature, windspeed, humidity, ozone


또한 visibility와 pm수치의 상관관계가 크다. 

null 값을 넣을 때 이를 이용해 넣는 것이 좋을 듯하다.

## 2. 데이터 시각화


```python
# 시간대에 따른 따릉이 대여량
train_df.groupby('hour').mean()['count'].plot()
plt.ylabel('count')
plt.title('hour_count')
```




    Text(0.5, 1.0, 'hour_count')




    
![png](../../../Downloads/ttareungyi/output_16_1.png)
    


그래프를 보면 5-10시 사이 뾰족한 부분은 출근시간, 15-20시 뾰족한 부분은 퇴근시간으로 볼 수 있다. 

즉, 출퇴근 시간에 따릉이 대여량 급증하는 것으로 볼 수 있다. 

또한 사람들의 활동량이 많아지는 10시 이후부터 대여량이 증가하고 퇴근시간 이후로는 대여량이 감소하는 것을 볼 수 있다. 
 

미세먼지의 결측값은 visibility의 값과의 비율로 임의로 넣겠다. 


```python
# 각 feature의 비율 구하기(pm10, pm2.5, visiibility, ozone) -> 결측치를 이 비율로 넣을 거임 
def feature_per(feature_1 : str, feature_2 : str):
    # 모든 결측치가 있는 행을 제거하고 인덱스 복원
    train_and_test_dropna = [train_df.dropna().reset_index(drop=True), test_df.dropna().reset_index(drop=True)]
    per_list = []
    for dataset in train_and_test_dropna:
        # 각 index들의 feature끼리의 비율 
        for i in range(len(dataset)):
            f1 = dataset[feature_1][i]
            f2 = dataset[feature_2][i]
            per = f1 / f2
            per_list.append(per)
    # 각 비율들의 평균 
    mean_per = round(sum(per_list) / len(per_list), 4)
    return mean_per
```

pm2.5와 pm10은 visibility와 상관관계가 있음.

생각해보면 미세먼지가 높을수록 가시성이 안 좋음 


```python
# pm10과 visibility의 비율 
feature_per('hour_bef_pm10', 'hour_bef_visibility')
```




    0.056




```python
# pm2.5와 visibility의 비율
# feature_per('hour_bef_pm2.5', 'hour_bef_visibility')
feature_per('hour_bef_pm2.5', 'hour_bef_pm10')

```




    0.6071



오존의 농도는 기온과 풍속의 상관관계가 있다. 

오존과 기온의 상관관계가 더 높기 때문에 이 비율로 결손값을 채우겠다.


```python
# ozone과 temperature의 비율
feature_per('hour_bef_ozone', 'hour_bef_temperature')
```




    0.0022



## 3. 결측치 채우기 


```python
# 데이터 합치기 
train_and_test = [train_df, test_df]

# 결측치 있는 행들만 보기
def fill_value(feature_1 : str, feature_2 : str):
    for dataset in train_and_test:
        # null 값이 있는 index 리스트
        null_index = dataset[dataset[feature_1].isnull()].index
        # 비율 값으로 null 값 채우기 
        for i in null_index:
            dataset[feature_1] = dataset[feature_2] * feature_per(feature_1, feature_2) 
    print(train_df.isnull().sum())
```

feature 값 중 temperature, precipitation, windspeed,humidity,visibility는 결측값이 적음 

-> feature는 평균으로 채움


```python
list = ['hour_bef_temperature', 'hour_bef_windspeed','hour_bef_humidity','hour_bef_visibility']

# null 개수 별로 없는 feature 채우기 
for dataset in train_and_test:
    for i in list:
        dataset[i].fillna(round(dataset[i].mean(),0), inplace=True)
        dataset['hour_bef_precipitation'].fillna(0, inplace=True)
```


```python
print(train_df.isnull().sum())
```

    id                          0
    hour                        0
    hour_bef_temperature        0
    hour_bef_precipitation      0
    hour_bef_windspeed          0
    hour_bef_humidity           0
    hour_bef_visibility         0
    hour_bef_ozone             76
    hour_bef_pm10              90
    hour_bef_pm2.5            117
    count                       0
    dtype: int64



```python
# pm10 값 채우기 
fill_value('hour_bef_pm10', 'hour_bef_visibility')
```

    id                          0
    hour                        0
    hour_bef_temperature        0
    hour_bef_precipitation      0
    hour_bef_windspeed          0
    hour_bef_humidity           0
    hour_bef_visibility         0
    hour_bef_ozone             76
    hour_bef_pm10               0
    hour_bef_pm2.5            117
    count                       0
    dtype: int64



```python
# pm2.5 값 채우기
# fill_value('hour_bef_pm2.5', 'hour_bef_visibility')
fill_value('hour_bef_pm2.5', 'hour_bef_pm10')
```

    id                         0
    hour                       0
    hour_bef_temperature       0
    hour_bef_precipitation     0
    hour_bef_windspeed         0
    hour_bef_humidity          0
    hour_bef_visibility        0
    hour_bef_ozone            76
    hour_bef_pm10              0
    hour_bef_pm2.5             0
    count                      0
    dtype: int64



```python
fill_value('hour_bef_ozone', 'hour_bef_temperature')
```

    id                        0
    hour                      0
    hour_bef_temperature      0
    hour_bef_precipitation    0
    hour_bef_windspeed        0
    hour_bef_humidity         0
    hour_bef_visibility       0
    hour_bef_ozone            0
    hour_bef_pm10             0
    hour_bef_pm2.5            0
    count                     0
    dtype: int64



```python
print(train_df.head(30))
```

        id  hour  hour_bef_temperature  hour_bef_precipitation  \
    0    3    20                  16.3                     1.0   
    1    6    13                  20.1                     0.0   
    2    7     6                  13.9                     0.0   
    3    8    23                   8.1                     0.0   
    4    9    18                  29.5                     0.0   
    5   13     2                  13.6                     0.0   
    6   14     3                  10.6                     0.0   
    7   16    21                  16.0                     0.0   
    8   19     9                  13.8                     0.0   
    9   20    14                  17.2                     0.0   
    10  21     4                   5.7                     0.0   
    11  22    10                  15.4                     0.0   
    12  24     9                  14.1                     0.0   
    13  27    10                   9.2                     0.0   
    14  28     1                  20.0                     0.0   
    15  29    13                  14.0                     1.0   
    16  30    21                  18.8                     0.0   
    17  32    17                  11.5                     1.0   
    18  33    13                  22.6                     0.0   
    19  34    18                  18.0                     1.0   
    20  35    10                  15.4                     0.0   
    21  36     8                  12.6                     0.0   
    22  37    16                  19.4                     0.0   
    23  38    23                  15.4                     0.0   
    24  44     3                  17.2                     0.0   
    25  45     0                  11.7                     0.0   
    26  46     2                  13.0                     0.0   
    27  47     7                  10.9                     0.0   
    28  48    15                  18.0                     0.0   
    29  49     2                   8.9                     0.0   
    
        hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  \
    0                  1.5               89.0                576.0   
    1                  1.4               48.0                916.0   
    2                  0.7               79.0               1382.0   
    3                  2.7               54.0                946.0   
    4                  4.8                7.0               2000.0   
    5                  1.7               80.0               1073.0   
    6                  1.5               58.0               1548.0   
    7                  6.0               21.0               1961.0   
    8                  1.9               64.0               1344.0   
    9                  2.1               32.0               1571.0   
    10                 0.6               77.0               1960.0   
    11                 2.7               62.0               1362.0   
    12                 3.2               59.0               1808.0   
    13                 1.9               55.0                462.0   
    14                 1.8               58.0               2000.0   
    15                 2.8               42.0               1518.0   
    16                 2.2               34.0               2000.0   
    17                 3.0               91.0                555.0   
    18                 2.0               41.0                987.0   
    19                 1.9               82.0                685.0   
    20                 3.1               19.0               1225.0   
    21                 5.3               53.0               1576.0   
    22                 5.9               21.0               2000.0   
    23                 1.4               87.0                423.0   
    24                 1.6               59.0               1277.0   
    25                 3.5               80.0                895.0   
    26                 0.5               48.0               1992.0   
    27                 0.8               77.0               1092.0   
    28                 3.3               20.0               2000.0   
    29                 0.9               52.0                839.0   
    
        hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count  
    0          0.02934        25.1712       17.899240   49.0  
    1          0.03618        40.0292       28.464764  159.0  
    2          0.02502        60.3934       42.945747   26.0  
    3          0.01458        41.3402       29.397016   57.0  
    4          0.05310        87.4000       62.150140  431.0  
    5          0.02448        46.8901       33.343550   39.0  
    6          0.01908        67.6476       48.104208   23.0  
    7          0.02880        85.6957       60.938212  146.0  
    8          0.02484        58.7328       41.764894   39.0  
    9          0.03096        68.6527       48.818935   83.0  
    10         0.01026        85.6520       60.907137    6.0  
    11         0.02772        59.5194       42.324245   42.0  
    12         0.02538        79.0096       56.183727   59.0  
    13         0.01656        20.1894       14.356682   60.0  
    14         0.03600        87.4000       62.150140   74.0  
    15         0.02520        66.3366       47.171956    5.0  
    16         0.03384        87.4000       62.150140  217.0  
    17         0.02070        24.2535       17.246664   64.0  
    18         0.04068        43.1319       30.671094  208.0  
    19         0.03240        29.9345       21.286423   15.0  
    20         0.02772        53.5325       38.066961   58.0  
    21         0.02268        68.8712       48.974310  209.0  
    22         0.03492        87.4000       62.150140  122.0  
    23         0.02772        18.4851       13.144755   87.0  
    24         0.03096        55.8049       39.682864   30.0  
    25         0.02106        39.1115       27.812188   34.0  
    26         0.02340        87.0504       61.901539   29.0  
    27         0.01962        47.7204       33.933976   18.0  
    28         0.03240        87.4000       62.150140  106.0  
    29         0.01602        36.6643       26.071984   10.0  


## 4. 범주 분류

미세먼지 등급 조건 

![Untitled.png](data:image/png;base64,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)

미세먼지, 오존을 등급 별로 나누어 보겠다

미세먼지(PM-10)

- 좋음(0) : 0 ~ 30
- 보통(1) : 31 ~ 80
- 나쁨(2) : 81 ~ 150
- 매우나쁨(3) : 151 ~ 

초미세먼지(PM-2.5)

- 좋음(0) : 0 ~ 15
- 보통(1) : 16 ~ 35
- 나쁨(2) : 36 ~ 75
- 매우나쁨(3) : 76 ~ 

오존
- 좋음(0) : 0 ~ 0.030
- 보통(1) : 0.031 ~ 0.090
- 나쁨(2) : 0.091 ~ 0.150
- 매우나쁨(3) : 0.151 ~ 



```python
# 미세먼지 pm10 등급 별로 나누기 

for dataset in train_and_test:
    dataset.loc[dataset['hour_bef_pm10'] <= 30, 'hour_bef_pm10'] = 0
    dataset.loc[(dataset['hour_bef_pm10'] > 30) & (dataset['hour_bef_pm10'] <= 80), 'hour_bef_pm10'] = 1
    dataset.loc[(dataset['hour_bef_pm10'] > 80) & (dataset['hour_bef_pm10'] <= 150), 'hour_bef_pm10'] = 2
    dataset.loc[(dataset['hour_bef_pm10'] > 150), 'hour_bef_pm10'] = 3

# train_df['hour_bef_pm10'].astype(int)
# test_df['hour_bef_pm10'].astype(int)
```


```python
# 미세먼지 pm2.5 등급 별로 나누기 

for dataset in train_and_test:
    dataset.loc[dataset['hour_bef_pm2.5'] <= 15, 'hour_bef_pm2.5'] = 0
    dataset.loc[(dataset['hour_bef_pm2.5'] > 15) & (dataset['hour_bef_pm2.5'] <= 35), 'hour_bef_pm2.5'] = 1
    dataset.loc[(dataset['hour_bef_pm2.5'] > 35) & (dataset['hour_bef_pm2.5'] <= 75), 'hour_bef_pm2.5'] = 2
    dataset.loc[(dataset['hour_bef_pm2.5'] > 75), 'hour_bef_pm2.5'] = 3
    
# train_df['hour_bef_pm2.5'].astype(int)
# test_df['hour_bef_pm2.5'].astype(int)
```


```python
# 오존 등급 별로 나누기
for dataset in train_and_test:
    dataset.loc[dataset['hour_bef_ozone'] <= 0.030, 'hour_bef_ozone'] = 0
    dataset.loc[(dataset['hour_bef_ozone'] > 0.030) & (dataset['hour_bef_ozone'] <= 0.090), 'hour_bef_ozone'] = 1
    dataset.loc[(dataset['hour_bef_ozone'] > 0.090) & (dataset['hour_bef_ozone'] <= 0.150), 'hour_bef_ozone'] = 2
    dataset.loc[(dataset['hour_bef_ozone'] > 0.151), 'hour_bef_ozone'] = 3
```

풍속 조건

- 고요 : 0 ~ 0.2
- 실바람 : 0.3 ~ 1.5
- 남실바람 : 1.6 ~ 3.3
- 산들 바람 : 3.4 ~ 5.4
- 적당한 산들 바람 : 5.5 ~ 7.9
- 신선한 바람 : 8.0 ~ 10.7
- 강한 바람 : 10.8 ~ 13.8
- 보통 강풍 : 13.9 ~ 17.1
- 신선한 강풍 : 17.2 ~ 20.7
- 강한 강풍 : 20.8 ~ 24.4
- 매우 강한 강풍 : 24.5 ~ 28.4
- 폭풍 : 28.5 ~ 32.6
- 허리케인 : 32.7 ~ 36.9


```python
# 풍속 등급 별로 나누기
for dataset in train_and_test:
    dataset.loc[dataset['hour_bef_windspeed'] <= 0.2, 'hour_bef_windspeed'] = 0
    dataset.loc[(dataset['hour_bef_windspeed'] > 0.2) & (dataset['hour_bef_windspeed'] <= 1.5), 'hour_bef_windspeed'] = 1
    dataset.loc[(dataset['hour_bef_windspeed'] > 1.5) & (dataset['hour_bef_windspeed'] <= 3.3), 'hour_bef_windspeed'] = 2
    dataset.loc[(dataset['hour_bef_windspeed'] > 3.3) & (dataset['hour_bef_windspeed'] <= 5.4), 'hour_bef_windspeed'] = 3
    dataset.loc[(dataset['hour_bef_windspeed'] > 5.4) & (dataset['hour_bef_windspeed'] <= 7.9), 'hour_bef_windspeed'] = 4
    dataset.loc[(dataset['hour_bef_windspeed'] > 7.9) & (dataset['hour_bef_windspeed'] <= 10.7), 'hour_bef_windspeed'] = 5
    dataset.loc[(dataset['hour_bef_windspeed'] > 10.7) & (dataset['hour_bef_windspeed'] <= 13.8), 'hour_bef_windspeed'] = 6
    dataset.loc[(dataset['hour_bef_windspeed'] > 13.8) & (dataset['hour_bef_windspeed'] <= 17.1), 'hour_bef_windspeed'] = 7
    dataset.loc[(dataset['hour_bef_windspeed'] > 17.1) & (dataset['hour_bef_windspeed'] <= 20.7), 'hour_bef_windspeed'] = 8
    dataset.loc[(dataset['hour_bef_windspeed'] > 20.7) & (dataset['hour_bef_windspeed'] <= 24.4), 'hour_bef_windspeed'] = 9
    dataset.loc[(dataset['hour_bef_windspeed'] > 24.4) & (dataset['hour_bef_windspeed'] <= 28.4), 'hour_bef_windspeed'] = 10
    dataset.loc[(dataset['hour_bef_windspeed'] > 28.4) & (dataset['hour_bef_windspeed'] <= 32.6), 'hour_bef_windspeed'] = 11
    dataset.loc[(dataset['hour_bef_windspeed'] > 32.6), 'hour_bef_windspeed'] = 12
```

불쾌지수

온도와 습도를 통해 불쾌지수를 구함

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAqkAAAB1CAYAAABtRXi6AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAALEoAACxKAXd6dE0AABYfSURBVHhe7d1tjFzVfcfx/+4aG6e4Qrh+4k2mEhhXrtSuQOKFBV3eQSpAFglQEaUGBYMiBQWKiiCKMK1iRIVEAmpjjAS8KMgBXItHpy9SNkZEQoJu+sKRMbzYShELdtw48uM+zEzv/849O2fOnPs0Ow9n1t+PdDRz7z33ce7s/Oace2dH6hEBAAAAAjKaPAIAAADBCKIl9YsvvojLNddck4xp+vjjj+WTTz5Jhhpuvvlmufzyy5OhYnQ5a9askauuuioZk0/n0fWUXRcAAACWpq8tqc8//7yMj4/H4c/29ttvy7333psMdUbnt5dh1mXotKeffjoZagRQd1t0WOczdB7dNgAAAPSXN6TueP5/5J4XGuU7e34jf/ev/y3bf/Kx3PgvH8l1//xrufpHH8hf/OOkfP0Hv0zm6D4NlBoSNTRqkLTL448/Hk/79NNPk9oNp06dWqyjLbNF6DLMPAAAAAhDW0j99s+mZGTFKhlZfZnUL26U6qq1shCVuVV/JvOr1kePG2Tu4o0yPz+fzFVOkQC5efPmuPtfu+ftIGkuC9Ci3fc2rafhVctbb72VjG3SeTX02us3YViL2rNnT9yiarfCAgAAoL/aQup8tS4LY1+TuYW6zEbl3Hxdzs7V5IyW2Zqcmq3GZX72nNTnziZz5dNWThMcX3311fjRZYdLra+h85VXXomH77//frnlllvigKnj9NFtSdVA+8ILL8RF67p0Hg2hdkh9+OGHF+dRGn51XVoAAAAwGG0hdW6hJuejcHp2vhFK37j36/LO9/5c/niuFpfT5+elev6U1GZPRSH1TDJXPm2x1HB41113xeFSu+w1iKY5ffp03FL6xBNPyLvvviv33Xdfy3Od121J1WHTyuq72UnHT01NxY/G0aNH4xuzzM1ZOk2XrwUAAACD0RZSz85W5bS2mJ7XUFpNxoqcPFuVs+fOSvXsH6V27mRUosfZ/JCqwdS0kGpA1ZZLfdThO++8syWomhZQrW+69vXGpYceemixS17D7eTkZDyfdt3b3fo6j+mq1xbTInR+rVu0PgAAAHqv7Seotv3Tr2XkT9bH3f7a3a/d/HNzC/K/u/8ynn75/e9Iff6c1BfmZOHUl/L7//iHeHwac52otkxqODU0UGrrpY7XsKkhUVs5zTSXLkO78zXk2sxPRGnYnJmZScY2mRZRDa7aSmq69Q1zyYDvp6l0u66++uqWllcAAAD0XltI/avHDkntaxujgFqLS23urNRnz8jvfvo38fSNf//vUUg9L/XqvFTP/F5OvPOjeHwec42phkG3m94NqUpDp/3zTxpcdT47TGpgdcOlrkNbWk3QNfOk/baqhl/lhlelwVavTaXrHwAAoL/aQmrlB/8l9Us2RkF0Ng6ov3tmWzx+445XpD6nLaizUUCdk3ptQWpn/yD/95+74+l5NDRqINQw6LZM2q2qhnb12zc42TS8aoh1l2XCrmldNTS4akjWa1rdG6p0m3wttwYhFQAAoP/aQurl33tPxvTnp+bPS23+jHzxs7+Nx6+/49+SFtQooFYXojH1+LrUP/yy+QP5WbJCqo8JnFncZWnLp7aa7tu3LxnToAHVXP966NChZGyDbpOGYd+vAej6CakAAAD91xZSN333DRldfWn881K1uTNxV3/c5b8QBdSFWZHqvNTrtajmSHyH/8n3f9KYMYcJqb7ufsPuvjch1delb7gtphpENXBqqLTn0VZaXZYGWrdbn5AKAAAQnraQqtecjl78p41gGoXU+FFvlEpaUaXWuOM/7vaPwmrRkKpd7va/JfWxA6l7TaqPXmdqh0ttKdV1vP/++/FzQ0PxDTfcEC/fDcha39w85eOuAwAAAL3XFlI3fPulOJTW4mBqSuM6VA2oGkxHRkaiOcfk5K+eS+YKk7lRCwAAAMOlLaQCAAAAg9b2Y/4AAADAoBFSAQAAEBxCKgAAAIJDSAUAAEBwCKkAAAAIDiEVAAAAwWn7Carp6enkGQAAANAflUoledbgDambNm1KhgAAAIDempmZaQupdPcDAAAgOIRUAAAABIeQCgAAgOAQUgEAABAcQioAAACCQ0gFAABAcAipAAAACA4hFQAAAMEhpAIAACA4hFQAAAAEh5AKAACA4BBSAQAAEBxCKgAAAIJDSAUAAEBwCKkAAAAIDiEVAAAAwSGkAgAAIDiEVAAAAASHkAoAAIDgEFIBAAAQHEIqAAAAgkNIBQAAQHAIqQAAAAjOSD2SPI9NT0/Lpk2bkiEgTA888EDyTOTZZ59NnuFCw3kAAMvDzMyMVCqVZKiBkIqho8HEDiTuMC4MnAcAsHz4Qird/QAAAAgOIRUAAADBobsfQ8m+FlHRzXth4jwAgOWBa1KxLOVdi/hldOLPfPllMtRu08aNspFzfuhxHgDA8PKF1LFdkeR57OTJk7JmzZpkCAjfwYMH5aabbkqG2r340kvy5ptvytrLLpPDv/2tPL9nj6xctUrm5+ZkT/Rcz/etW7cmtWFoqPvs88/lyyjY2UWP18qVK5NaDVNTUy11zBfdtPG9wHkAAMPr9OnTcumllyZDDVyTiqHjdvHm+fyzz+TJ3bvlW7ffHreWKV2GDt/0jW/Ew8Ok7P536lT0B+P1116Txx59NA6bSo/l3Tt2yIsvvhgPGxpotd5L0Xj9Q6P00TfeKLIfWXWG/Tzo1+vYqazt09dVz4m0otO7KW1bdLyvpMmalqeTedOOk/teUG4dI238Uvn2p8g+2nXK1u+VTtfhm68f24viCKkByXpzFH3jmHpl32hF6pddZq9ol65uiylZXbz6YXDFlVcuduNqd6/dpbvmkkvi6Z2ytyOPXdedJ2uaTcen7W/aPJ26Mjou4+Pj8fN77rknfq6B7vaoaHj9LAp9hgl51113XVzUJdGxve766+Pn9njDvI5psvZVhXIe2NvgkzY9b//LSFuHypqWRutmHc/XXn89nq7B/4MPPoi/iOijDu/+8Y/j52XY22hKHq2j2+ArvvlNfVuR9Rhpy83S6y96vab7a5eyiszjrsMtebSO73UtMq/vNe3kdUbvEFIDY95cbnEVqWPz1TeliKL1+kX/kJiSRwOWoX/wr7ziimSoEa40jHVCj4m9HUWOkV1fiy1rWp5evT76wWqCqqEfvMruujcfwO41nSbILuWLQJYyx6sX50HeOdDJOVJW1jp6tf6yrdJ567W3UUs/mGNjb5s+N6Ubev1Fz9Wt7bZ1+rrotpj58rbLXodbyspaby+OD3qLkBoY+81pF1fedJdb1x3OUuaPTWj0j7wWQ8OUHUbc6SEzr4NPkdexE26Lox6/g++9F2+Lfdw09Ck7+CkNMyotAKadU1n72olQzgN3n9L238iaZnT7tc879r3unSjKHDtfcbffN87Mb8ab56bosCttfJYQv+iZffXti44ru482M78u3zDrWspyfdz1ZEmr6zsOvnEYDEIqAAAAgkNIDYx+e/OVQTDrtr99mm+YnW6Tmdcu/WJaJNyWin7K2u+saYOiLaRatGvy7rvvjh+fe+65ti5dc62cnh96/Zwpr0X1+9VKWVQ3zwP7vRGiXm1f2UsnerUdulxT7OGiTF19z7nzlVlOll73RtiKbLO9r/ro/r3RcXnL0Xnc+Yy0+Ysstxt0HWb7zPrs5xgu/E5qINLe8C77Tecyb0770cc3Lav+oPn2NUvafujNHHpTx7NRyEr7g+9bV9HjmHcMs+oXWVbZ5ftoHVfaPPphqkFz95NPxt2VWcdP6ymta0sbbyu6r75tz5K2X3nnQZljZCuyzb7lFN3XrG2w66fV863Hp2g946Ybb4xDq15r2Q1pxyNvm7Lq5M1fZp/L1FV6fPT9o+W9gwfja3h1Ge6XJP0CqDdTuZcGmC9Vr7/xRvy4FGnbbsa703317XG++kWUmcddv823fa5O6xSZD93Fj/kPmbw3iT3dPNdHwzevXc+e7g73ir19Rj/Wq8yHwMFf/CIZszRLPYZZ9X3T8pZfdv15zPHSD0fT4pMWSL71zW/GN3joNtiKBJhO9nUpun0eqLTtdceX3ddOjkPZdbjK1NUA9cD3vy+P/fCHuTf1FFVk+3W4CDNP3j6V2ecydfv1Ra9XfPtqjytzLIrq5mvhbqvy1fctp8x2oDt8IZXu/oDom8Iu7rg0Os1+M3XjjWWvN6uUpdvmln7Q7mj9gCjSZYYGvbnD7qq3b+zQ56aLXx+1mLu8DdN9GVJXfy/OA30f9Os87kQvt0//QYPa6Lz2rqJ/K4puq/nbkVcGze2qN0H+N8l7yaaB33cJir7X3NbVNGWOs690Q9ZyurWOInRd5hwwz7X0cxuwdITUgJg3ka9kyZtu2G9afcx6s5r12sU3fhhoWNJrwMxPuGhLWq/5jm3W8S7yh1OPd5F63aBhTj9I7V4VE0L1w1SPpwmfHxw6FD+6H7AaBtUlGf/BTvfHdx71Yl97cR6kbX9RefMXWXbWcepk+8oc+8XrKJcY+nV9Zbe16DaqvH3yTffVd7cxbxtC/KJn9sFX0vbZLkW485jSbVnbrNOK8NUtMz96i+7+gOS9ie03ja+uecPaj0bamy6tvk+ROv2g2+Ea1HbZ22JvQ9qxSquvsqYZactVWdPK0A9GEzAN012vwU5Dqg5rMNHQZz5o1V+Pj2eOdy1lf+zjZXRj/8sosg12Hd+0bmxz2jo6PUZFtktfX9MVrd3XveTbnrLHrkh9+3j56rrLyFqm7/joe0tvQNT3jz7XyySUvl90Oe5lE+bSlF5fTmEUPUamjq9+meXrcBFZ25S1PptZV5HtLbpMdBfXpAauG28Ms4yyyypSP5Q3bijbMSjLZf+Xes4N+3EIffuzti/ri0wv+LZFx2XxbftSjrk7b6fHp9tf9DqRduzKHhvfMUhbttHp8c+S9VpkSdv+Xmwj8hFSA9fNN3fZN1qR+qG8efkjAsV5AADLByEVy4KGExtB5cLEeQAAywd392PZ0EBiihtWcOHgPACA5YuQiqGjgQTgPACA5Y2QCgAAgOAQUjF06NaF4jwAgOWNG6cwlOyAQrfvhYvzAACWB+7uBwAAQHC4ux8AAABDgZAKAACA4BBSAQAAEBxCKgAAAIJDSAUAAEBwCKkAAAAIDiEVAAAAwSGkAgAAIDiEVAAAAASHkAoAAIDgEFIBAAAQHEIqAAAAgkNIBQAAQHAIqQAAAAgOIRUAAADBIaQCAAAgOCP1SPI8Nj09LZs2bUqGLiRfyUV7X5b6zkdkIRnT9JGsPCAyt/3aZNgWTXtyQkaSodjal2Vh5x1SbZtP11GR0RPJoG1xHqN1ufXbzsncZn0WLePApNS223UDc+LnsupQRWa9x6tLjj4lq/bvip+aYzP24Z0ytm6fVI8/JaNbHpH5tfHkVrpte3ckA7dKbee+pF7Wa5XUE7NfknE+ZLzGskuqj7aeXysOrJaxI43nzdc42pa901K1zge7XoPZdne7AQAYPjMzM1KpVJKhhvBbUjWMPPmUrEgGF8XjVyeldbp+oDenReXDrxoTWuZJyoGPGtPaaEg09ZwQ6tKA+eg5mY3LtNRkOqX+BpnfaerZZVJavilEVhyYEIlCi1lmffJOucgbfGwakKx9i8rKo8mkXO68OevzHcuoFF/fUkSvzX6JAl90bHa+LCP7PeeHV7SP+w9E4c4c1wdlZG/avPZrZYJsEWmv8TmpbpmSEeuYxqF6/XQyPe811oBrL6/MNgEAMHzCDqkahA5PRB/uybChrWGTlWYwvC36wG8Jm84H+rYNjdGbH2mO03LbLpH1ram96VqZW6zbHiK7bl3Fahn9Kgq5L0s1blVTG6S2VVoCTqp1dmA2LXN5NJBXZGSiOV9uCFo8llEoX9s83sXWt0QnpkWu39FokVx7R3R+7JLRQuE4+vKw7kFrv66VmhMcG+wvKFYAb2nJ7MS41K1jOnJsXKrm3NRwOzFe7DUGAOACMMCQqt2Uaa2YCQ1Cvm5MDSlbo/CaDMrmHVI7Ppm0iEUB73j8JMdXctGkWCHBVaIldami/Wld/oYoFO+QscXg9ZWMHpaWgNNNYx8+E7fadjVgHn+mGfD2/lzGktGqpaXbtHJHxj58Si6KSmNao1Vxsa69DOd41dffKiP7V8uKQ28mY9JUpB5tV7O18iMZPdIaHBvsLyjNMud+WeqlEztkRcvx2SVj5pglpT+t1gAADMYAQmrSrbw3SoiSBBkrqBSytiJyeNIKLZPONYDWB7oTkBbpPOsmPNefGl1uSU3pHo+LXiN5ZCJ+boLHwnbt/jV1tJWzV927GoDHpdbtFtAT44utqwtbDywGbreLu3a4YoWtKRk9NpFM06741Yt17WX46PWcC9ffmgyl0a74xnIbx/WZlGuQG5ph2uqGj1+ngl9atMV/8YtY+5en+vqp6HiYc18vYZhqBmZzCUnyJWphe+NYVrfotaiN531ptQYAYEAGEFKTa/YmREaOSOMDN7U1M8XaO+LQErc0adkfBbrFAOdcEzgxHYWN9hbbFYcOSO16t5W2h61V7qUGGtC2tHbNa2kGj9b9KBxIjictcHHJua7UWFfp/uUMW5pfAKpbtkfbpWGsEYhbu7h3ycjh5utT32pek2ib9DKCpG5zGRH9knJsuvE8MnKsTCuz/eUjI/hHXyqaYToKtvuTLztbJqPhDr+0OMe5um2fVI9VktdKr0H2b4/d8jx25E0Zta4dpjUVALBcDaa7X1uYDmuLmfXhX5J+wJsAN7sz+oDXUNOY1GrzhNSPTzvr+EhGj2+XWksgcMJtUoautarlmtSCra/R8enp5QyLdD3O66SBM01aeI7mqR9JLu+IzqWxttcyRNG+ey5DMS2kWeeaXcctc5uj0M2d/QCAZWgwIXXtHcm1ptEHbMvPLnXo6KSMtNx4lEPr29e0WrQ7unTrlLl+MC6V6KCmhKuW61yjekes+azLEnQbTEtZW/mw2YLYHXpT1lTBG4+WSo+LE4j1+uLSovNGb5bT47H3gNRuK3YOtf3qg1XaXvPNj1itnM9IveA60qWc63oZSKHLXdp/uWGx5F3bDQDAEBrgjVPd0riWr73rvkFvCmoNpHk3TPlktVZF09wWrtTg7akbl9bu45ZWYrdsy2h57FB124Mi+/vRddwIxM1fYtDXIv21y7R4+UTxa3WzWySTSpZm/YLr0B4COzxa1xq3lNxQmhJo236GSksffnkCAIABGNKQardI6s0vdohobXFacezB1mtec2+YGnIt16RGxQQibbFLu4ksDs/2jVpazPWseqwLXttaQHwdppjgVpHRrT8tHDKDpz0EbSHSU8pegw0AwAWI/zjl0K52/08ZaStW+p3gflHAK/TfgKJ6zn8YSheF8ND/49SAxL8ckPcfp1IVeK20pTT3P06VpMtc/A9YrZr/gUrpl6+U/2SlN3N1Y1sAABgQ33+cIqQCAABgoIbz36ICAADggkNIBQAAQHAIqQAAAAgOIRUAAADBIaQCAAAgOIRUAAAABIeQCgAAgOAQUgEAABAcQioAAACCQ0gFAABAcLz/FhUAAADop9z/3Q8AAAAMGt39AAAACA4hFQAAAIER+X95CH/Xpgb6sAAAAABJRU5ErkJggg==)

![image.png](data:image/png;base64,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)


```python
# 불쾌지수 공식
def discomfort_index(temperature : float, humidity: int):
    result = (9/5) * temperature - 0.55 * (1-humidity/100) * ((9/5)*temperature - 26) + 32
    return round(result)
```


```python
# 불쾌지수 구하기 
for dataset in train_and_test:
    discomfort_list = []
    for i in range(len(dataset)):
         discomfort_list.append(discomfort_index(dataset['hour_bef_temperature'][i], dataset['hour_bef_humidity'][i]))
    dataset['discomfort_index'] = discomfort_list
```


```python
train_df['discomfort_index']
```




    0       61
    1       65
    2       57
    3       49
    4       71
            ..
    1454    61
    1455    53
    1456    63
    1457    65
    1458    66
    Name: discomfort_index, Length: 1459, dtype: int64




```python
# # 불쾌지수 등급 별로 나누기 
# for dataset in train_and_test:
#     dataset.loc[dataset['discomfort_index'] < 68, 'discomfort_index'] = 0
#     dataset.loc[(dataset['discomfort_index'] >= 68) & (dataset['discomfort_index'] < 75), 'discomfort_index'] = 1
#     dataset.loc[(dataset['discomfort_index'] >= 75) & (dataset['discomfort_index'] < 80), 'discomfort_index'] = 2
#     dataset.loc[(dataset['discomfort_index'] >= 80), 'discomfort_index'] = 3
```


```python
train_df.info
```




    <bound method DataFrame.info of         id  hour  hour_bef_temperature  hour_bef_precipitation  \
    0        3    20                  16.3                     1.0   
    1        6    13                  20.1                     0.0   
    2        7     6                  13.9                     0.0   
    3        8    23                   8.1                     0.0   
    4        9    18                  29.5                     0.0   
    ...    ...   ...                   ...                     ...   
    1454  2174     4                  16.8                     0.0   
    1455  2175     3                  10.8                     0.0   
    1456  2176     5                  18.3                     0.0   
    1457  2178    21                  20.7                     0.0   
    1458  2179    17                  21.1                     0.0   
    
          hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  \
    0                    1.5               89.0                576.0   
    1                    1.4               48.0                916.0   
    2                    0.7               79.0               1382.0   
    3                    2.7               54.0                946.0   
    4                    4.8                7.0               2000.0   
    ...                  ...                ...                  ...   
    1454                 1.6               53.0               2000.0   
    1455                 3.8               45.0               2000.0   
    1456                 1.9               54.0               2000.0   
    1457                 3.7               37.0               1395.0   
    1458                 3.1               47.0               1973.0   
    
          hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count  discomfort_index  
    0            0.02934        25.1712       17.899240   49.0                61  
    1            0.03618        40.0292       28.464764  159.0                65  
    2            0.02502        60.3934       42.945747   26.0                57  
    3            0.01458        41.3402       29.397016   57.0                49  
    4            0.05310        87.4000       62.150140  431.0                71  
    ...              ...            ...             ...    ...               ...  
    1454         0.03024        87.4000       62.150140   21.0                61  
    1455         0.01944        87.4000       62.150140   20.0                53  
    1456         0.03294        87.4000       62.150140   22.0                63  
    1457         0.03726        60.9615       43.349723  216.0                65  
    1458         0.03798        86.2201       61.311113  170.0                66  
    
    [1459 rows x 12 columns]>



visibility(시정, 시계)를 등급 별로 나눠주기

500m씩 범위를 나눔





```python
# 시정거리 등급 별로 나누기 
for dataset in train_and_test:
    dataset.loc[dataset['hour_bef_visibility'] <= 500, 'hour_bef_visibility'] = 0
    dataset.loc[(dataset['hour_bef_visibility'] > 500) & (dataset['hour_bef_visibility'] <= 1000), 'hour_bef_visibility'] = 1
    dataset.loc[(dataset['hour_bef_visibility'] > 1000) & (dataset['hour_bef_visibility'] <= 1500), 'hour_bef_visibility'] = 2
    dataset.loc[(dataset['hour_bef_visibility'] > 1500) & (dataset['hour_bef_visibility'] <= 2000), 'hour_bef_visibility'] = 3
```


```python
plt.figure(figsize=(13,10))
g = sns.heatmap(train_df.corr(),annot=True,cmap="RdYlGn")
```


    
![png](../../../Downloads/ttareungyi/output_51_0.png)
    



```python
# 모든 데이터 타입을 int로 변환 

for dataset in train_and_test:
    for i in dataset.columns:
        dataset[i] = dataset[i].astype(int)
```


```python
train_df.info
```




    <bound method DataFrame.info of         id  hour  hour_bef_temperature  hour_bef_precipitation  \
    0        3    20                    16                       1   
    1        6    13                    20                       0   
    2        7     6                    13                       0   
    3        8    23                     8                       0   
    4        9    18                    29                       0   
    ...    ...   ...                   ...                     ...   
    1454  2174     4                    16                       0   
    1455  2175     3                    10                       0   
    1456  2176     5                    18                       0   
    1457  2178    21                    20                       0   
    1458  2179    17                    21                       0   
    
          hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  \
    0                      1                 89                  576   
    1                      1                 48                  916   
    2                      0                 79                 1382   
    3                      2                 54                  946   
    4                      4                  7                 2000   
    ...                  ...                ...                  ...   
    1454                   1                 53                 2000   
    1455                   3                 45                 2000   
    1456                   1                 54                 2000   
    1457                   3                 37                 1395   
    1458                   3                 47                 1973   
    
          hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count  discomfort_index  
    0                  0             25              17     49                61  
    1                  0             40              28    159                65  
    2                  0             60              42     26                57  
    3                  0             41              29     57                49  
    4                  0             87              62    431                71  
    ...              ...            ...             ...    ...               ...  
    1454               0             87              62     21                61  
    1455               0             87              62     20                53  
    1456               0             87              62     22                63  
    1457               0             60              43    216                65  
    1458               0             86              61    170                66  
    
    [1459 rows x 12 columns]>






```python
# 학습시 필요없는 feature drop 

train_df = train_df.drop(['id', 'hour_bef_visibility', 'hour_bef_precipitation'], axis=1)
test_df = test_df.drop(['id', 'hour_bef_visibility', 'hour_bef_precipitation'], axis=1)
```


```python
train_df = train_df.drop(['hour_bef_pm10', 'hour_bef_pm2.5', 'hour_bef_ozone'],axis=1)
test_df = test_df.drop(['hour_bef_pm10', 'hour_bef_pm2.5', 'hour_bef_ozone'],axis=1)
```


```python
train_df = train_df.drop(['hour_bef_humidity'],axis=1)
test_df = test_df.drop(['hour_bef_humidity'],axis=1)
```


```python
print(train_df)
```

            id  hour  hour_bef_temperature  hour_bef_precipitation  \
    0        3    20                    16                       1   
    1        6    13                    20                       0   
    2        7     6                    13                       0   
    3        8    23                     8                       0   
    4        9    18                    29                       0   
    ...    ...   ...                   ...                     ...   
    1454  2174     4                    16                       0   
    1455  2175     3                    10                       0   
    1456  2176     5                    18                       0   
    1457  2178    21                    20                       0   
    1458  2179    17                    21                       0   
    
          hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  count  \
    0                      1                 89                  576     49   
    1                      1                 48                  916    159   
    2                      0                 79                 1382     26   
    3                      2                 54                  946     57   
    4                      4                  7                 2000    431   
    ...                  ...                ...                  ...    ...   
    1454                   1                 53                 2000     21   
    1455                   3                 45                 2000     20   
    1456                   1                 54                 2000     22   
    1457                   3                 37                 1395    216   
    1458                   3                 47                 1973    170   
    
          discomfort_index  
    0                   61  
    1                   65  
    2                   57  
    3                   49  
    4                   71  
    ...                ...  
    1454                61  
    1455                53  
    1456                63  
    1457                65  
    1458                66  
    
    [1459 rows x 9 columns]


## 정규화


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_df)

train_scaled = scaler.fit_transform(train_df)

train_scaled_df = pd.DataFrame(train_scaled, columns = train_df.index)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-53-f20fc48eb9ef> in <module>
          6 train_scaled = scaler.fit_transform(train_df)
          7 
    ----> 8 train_scaled_df = pd.DataFrame(train_scaled, columns = train_df.index)
    

    /usr/local/lib/python3.8/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
        670                 )
        671             else:
    --> 672                 mgr = ndarray_to_mgr(
        673                     data,
        674                     index,


    /usr/local/lib/python3.8/dist-packages/pandas/core/internals/construction.py in ndarray_to_mgr(values, index, columns, dtype, copy, typ)
        322     )
        323 
    --> 324     _check_values_indices_shape_match(values, index, columns)
        325 
        326     if typ == "array":


    /usr/local/lib/python3.8/dist-packages/pandas/core/internals/construction.py in _check_values_indices_shape_match(values, index, columns)
        391         passed = values.shape
        392         implied = (len(index), len(columns))
    --> 393         raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
        394 
        395 


    ValueError: Shape of passed values is (1459, 6), indices imply (1459, 1459)



```python
print(train_df_scaled)
```

    [[0.86956522 0.49070632 0.1875     ... 0.2591051  0.11162791 0.59459459]
     [0.56521739 0.63197026 0.175      ... 0.43600416 0.36744186 0.7027027 ]
     [0.26086957 0.40148699 0.0875     ... 0.67845994 0.05813953 0.48648649]
     ...
     [0.2173913  0.56505576 0.2375     ... 1.         0.04883721 0.64864865]
     [0.91304348 0.65427509 0.4625     ... 0.68522373 0.5        0.7027027 ]
     [0.73913043 0.66914498 0.3875     ... 0.98595213 0.39302326 0.72972973]]


## 모델에 넣을 데이터 준비


```python
# 목적 변수 제거
X_train = train_df.drop(["count"], axis=1)
#목적 변수 역할
Y_train = train_df["count"]
#예측 대상 데이터 셋
X_test  = test_df
X_train.shape, Y_train.shape, X_test.shape
```




    ((1459, 5), (1459,), (715, 5))




```python
print(X_train)
print(Y_train)
print(X_test)
```

          hour  hour_bef_temperature  hour_bef_windspeed  hour_bef_humidity  \
    0       20                    16                   1                 89   
    1       13                    20                   1                 48   
    2        6                    13                   0                 79   
    3       23                     8                   2                 54   
    4       18                    29                   4                  7   
    ...    ...                   ...                 ...                ...   
    1454     4                    16                   1                 53   
    1455     3                    10                   3                 45   
    1456     5                    18                   1                 54   
    1457    21                    20                   3                 37   
    1458    17                    21                   3                 47   
    
          discomfort_index  
    0                   61  
    1                   65  
    2                   57  
    3                   49  
    4                   71  
    ...                ...  
    1454                61  
    1455                53  
    1456                63  
    1457                65  
    1458                66  
    
    [1459 rows x 5 columns]
    0        49
    1       159
    2        26
    3        57
    4       431
           ... 
    1454     21
    1455     20
    1456     22
    1457    216
    1458    170
    Name: count, Length: 1459, dtype: int64
         hour  hour_bef_temperature  hour_bef_windspeed  hour_bef_humidity  \
    0       7                    20                   1                 62   
    1      17                    30                   5                 33   
    2      13                    19                   2                 95   
    3       6                    22                   2                 60   
    4      22                    14                   3                 93   
    ..    ...                   ...                 ...                ...   
    710     1                    24                   2                 60   
    711     1                    18                   1                 55   
    712     9                    23                   2                 66   
    713    16                    27                   1                 46   
    714     8                    22                   1                 63   
    
         discomfort_index  
    0                  67  
    1                  76  
    2                  66  
    3                  69  
    4                  58  
    ..                ...  
    710                72  
    711                63  
    712                71  
    713                74  
    714                69  
    
    [715 rows x 5 columns]


## 모델 돌리기


```python
# model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
```


```python
# LinearRegression

model = LinearRegression()
model.fit(X_train, Y_train)
y_hat = model.predict(X_train)
nmae = np.mean(abs(y_hat - Y_train) / Y_train) # nmae 계산

print(f'모델 NMAE: {nmae}')
```

    모델 NMAE: 1.445731918699432



```python
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
```

    /usr/local/lib/python3.8/dist-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    4.59




```python
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
```





  <div id="df-65673992-5f1e-40bf-a607-cd7d00800e53">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>hour_bef_humidity</td>
      <td>0.277468</td>
    </tr>
    <tr>
      <th>3</th>
      <td>count</td>
      <td>0.151550</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hour_bef_windspeed</td>
      <td>0.081298</td>
    </tr>
    <tr>
      <th>0</th>
      <td>hour_bef_temperature</td>
      <td>-0.030161</td>
    </tr>
    <tr>
      <th>4</th>
      <td>discomfort_index</td>
      <td>-0.143713</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-65673992-5f1e-40bf-a607-cd7d00800e53')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-65673992-5f1e-40bf-a607-cd7d00800e53 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-65673992-5f1e-40bf-a607-cd7d00800e53');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```




    2.12




```python
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
```




    32.49




```python
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
```




    2.06




```python
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
```




    99.52




```python
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```




    99.52


