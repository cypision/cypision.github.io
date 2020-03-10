---
title:  "Recommendation Basic02"
excerpt: "Recommendation system using surprise library"

categories:
  - Machine-Learning
tags:
  - ML  
  - surprise  
  - Medium/susanli
  - Recommendation
last_modified_at: 2020-03-08T18:06:00-05:00
---

추천 시스템에서 유명한 library 는 surprise 이다. 워낙 유명하나, 개인적으로 한번 밖에 해본적이 없어서,  
기억이 가물거리는 관계로 남긴다. Basic 01 에 배경설명 등이 있고, 이어지는 post 이다
> 1. CF, latent matrix 를 활용한다.  
> 2. [Medium susanLi](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)  
> 3. [SusanLi github](https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Building%20Recommender%20System%20with%20Surprise.ipynb)
> 4. [나름 추천 잘하는 사람일것 같은 사람의 github](https://github.com/bigsnarfdude/guide-to-data-mining)
> 5. [Naber Lab Reseacher 초고수](http://sanghyukchun.github.io/31/)


```python
import pandas as pd
import numpy as np
```

[surprise 공식문서 링크](https://surprise.readthedocs.io/en/stable/)


```python
user = pd.read_csv('D:/★2020_ML_DL_Project/Alchemy/dataset/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
user.columns = ['userID', 'Location', 'Age']
rating = pd.read_csv('D:/★2020_ML_DL_Project/Alchemy/dataset/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
rating.columns = ['userID', 'ISBN', 'bookRating']
```


```python
df = pd.merge(user, rating, on='userID', how='inner') ## inner join 했지만, 딱히 줄어들거나, 변경된것은 없는 듯.
df.drop(['Location', 'Age'], axis=1, inplace=True)
```


```python
print(df.shape)
df.head(3) ## 별차이 없네..그냥 rating 데이터 하나만 써도 될듯 1149779
```

    (1149779, 3)
    




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
      <th>userID</th>
      <th>ISBN</th>
      <th>bookRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0195153448</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>034542252</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>0002005018</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



__본격 EDA__  
Rating Distribution


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style="white", context="talk")
```


```python
data = df['bookRating'].value_counts().sort_index(ascending=False)
print(data)
```

    10     78610
    9      67541
    8     103736
    7      76457
    6      36924
    5      50974
    4       8904
    3       5996
    2       2759
    1       1770
    0     716108
    Name: bookRating, dtype: int64
    


```python
(data/data.sum())*100
```




    10     6.836966
    9      5.874259
    8      9.022256
    7      6.649713
    6      3.211400
    5      4.433374
    4      0.774410
    3      0.521492
    2      0.239959
    1      0.153943
    0     62.282230
    Name: bookRating, dtype: float64



대략 62%에 달하는 user 들이 rating에 0 점을 부여했다. 이 0점은 과연 book 평점을 최악을 준걸까? 아니면, 평점 자체를 달지 않은걸까? 어떻게 봐야하나?

책별로, 평점이 가장 많이 달린걸 보자면...


```python
df.groupby(by=['ISBN'],as_index=False)['bookRating'].count().sort_values('bookRating', ascending=False)[:10]
```




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
      <th>ISBN</th>
      <th>bookRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>247408</th>
      <td>0971880107</td>
      <td>2502</td>
    </tr>
    <tr>
      <th>47371</th>
      <td>0316666343</td>
      <td>1295</td>
    </tr>
    <tr>
      <th>83359</th>
      <td>0385504209</td>
      <td>883</td>
    </tr>
    <tr>
      <th>9637</th>
      <td>0060928336</td>
      <td>732</td>
    </tr>
    <tr>
      <th>41007</th>
      <td>0312195516</td>
      <td>723</td>
    </tr>
    <tr>
      <th>101670</th>
      <td>044023722X</td>
      <td>647</td>
    </tr>
    <tr>
      <th>166705</th>
      <td>0679781587</td>
      <td>639</td>
    </tr>
    <tr>
      <th>28153</th>
      <td>0142001740</td>
      <td>615</td>
    </tr>
    <tr>
      <th>166434</th>
      <td>067976402X</td>
      <td>614</td>
    </tr>
    <tr>
      <th>153620</th>
      <td>0671027360</td>
      <td>586</td>
    </tr>
  </tbody>
</table>
</div>



유저별로, 어떤 유저가 평점을 가자 많이 달았는지 보자면....


```python
df.groupby(by=['userID'],as_index=False)['bookRating'].count().sort_values('bookRating', ascending=False)[:10]
```




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
      <th>userID</th>
      <th>bookRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4213</th>
      <td>11676</td>
      <td>13602</td>
    </tr>
    <tr>
      <th>74815</th>
      <td>198711</td>
      <td>7550</td>
    </tr>
    <tr>
      <th>58113</th>
      <td>153662</td>
      <td>6109</td>
    </tr>
    <tr>
      <th>37356</th>
      <td>98391</td>
      <td>5891</td>
    </tr>
    <tr>
      <th>13576</th>
      <td>35859</td>
      <td>5850</td>
    </tr>
    <tr>
      <th>80185</th>
      <td>212898</td>
      <td>4785</td>
    </tr>
    <tr>
      <th>105110</th>
      <td>278418</td>
      <td>4533</td>
    </tr>
    <tr>
      <th>28884</th>
      <td>76352</td>
      <td>3367</td>
    </tr>
    <tr>
      <th>42037</th>
      <td>110973</td>
      <td>3100</td>
    </tr>
    <tr>
      <th>88584</th>
      <td>235105</td>
      <td>3067</td>
    </tr>
  </tbody>
</table>
</div>




```python
min_book_ratings = 50
filter_books = df['ISBN'].value_counts() > min_book_ratings  ## 50보다 큰 ISBN 번호들만 남긴다. 
filter_books = filter_books[filter_books].index.tolist() ## 평점 50개 달린,(또는 그이상)되는 책들만, 리스트화 한다.

min_user_ratings = 50
filter_users = df['userID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

df_new = df[(df['ISBN'].isin(filter_books)) & (df['userID'].isin(filter_users))] ## 최소 충족요건을 만족한 data 들만, 추려서 들고온다.
print('The original data frame shape:\t{}'.format(df.shape))
print('The new data frame shape:\t{}'.format(df_new.shape))
```

    The original data frame shape:	(1149779, 3)
    The new data frame shape:	(140516, 3)
    


```python
df_new.head()
```




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
      <th>userID</th>
      <th>ISBN</th>
      <th>bookRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>394</th>
      <td>243</td>
      <td>0060915544</td>
      <td>10</td>
    </tr>
    <tr>
      <th>395</th>
      <td>243</td>
      <td>0060977493</td>
      <td>7</td>
    </tr>
    <tr>
      <th>397</th>
      <td>243</td>
      <td>0156006529</td>
      <td>0</td>
    </tr>
    <tr>
      <th>400</th>
      <td>243</td>
      <td>0316096199</td>
      <td>0</td>
    </tr>
    <tr>
      <th>401</th>
      <td>243</td>
      <td>0316601950</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



__진짜 Surprise library 사용하기__


```python
from surprise import Reader
from surprise import Dataset
```

We use rmse as our accuracy metric for the predictions.

https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html 상세한 설명은 여기에

여기서부터는 surprise의 여러 알고리즘이 등장하는데, Susan Li 의 설명을 보고도 몰라서, 그냥 내가 하나하나 정리하면서, 올린다.

# Basic algorithms

**NormalPredictor**
- NormalPredictor algorithm predicts a random rating based on the distribution of the training set, which is assumed to be normal. This is one of the most basic algorithms that do not do much work.  

Basic01 post 에 남겼다.

**BaselineOnly**
- BasiclineOnly algorithm predicts the baseline estimate for given user and item.
- Baselines estimates configuration

제일 위의 식이 Baselines 의 내용인데, bui = rui 즉 어떤 rating을 추정하는 텀. 맨 위가 Baseline 의 기본개념이다.  
뮤 는 전체 rating의 평균이고, bu 은 user의 bias, bi 는 item 의 bias이다. 이때, 핵심이 되는 bu, bi 를 구하는 Minimzie object 식이 밑에 그림이다.  
**Minimzie object 이 식의 찾는 과정에 fit(train) 과정이라 보면되는데 2가지 방식 'ALS' 랑 'SGD'방식이 있다.**

Basic01 post 에 남겼다.  
하기는 수식 notaion에 대한 설명

![image.png](/assets/images/Surprise_basic_02/notation.png)

갑자기 기억이 안나서 찾아본다....  원본 : https://goofcode.github.io/similarity-measure  
토막상식

![image.png](/assets/images/Surprise_basic_02/cosine.png)

# k-NN algorithms
**KNNBasic**  
- KNNBasic is a basic collaborative filtering algorithm.

![image.png](/assets/images/Surprise_basic_02/knnbasic.png)

상기 알고리즘이 현재 유명한 Callaborative Filtering 이다. 둘 중 위의식이 user-based , 밑의 식이 item-based 수식이다.


```python
df_new.bookRating.value_counts().sort_index(ascending=False)[0:5]
```




    10     8778
    9      7966
    8     10381
    7      6694
    6      2917
    Name: bookRating, dtype: int64




```python
## surprise 전용 dataset 으로 불러오기
from surprise import Reader
from surprise import Dataset
## Dataset 모듈에서, Reader class를 param으로 사용한다.
reader = Reader(rating_scale=(0, 9)) ## 실제 data 는 0~10 까지인데...왜 susan li 는 0~9 로 했을까? --> 이상하지만, 10점은 그래도 10점으로 변환된 값을 가지고 있따.
## (rating_scale=(0, 10) 으로 해도 크게 달라지는 건 없다.
s_data = Dataset.load_from_df(df_new[['userID', 'ISBN', 'bookRating']], reader)
```


```python
## 243 유저가, s_data 에서는 어떻게 변하는지 확인해보면, 
print(df_new.shape)
print(df_new.loc[(df_new.userID==243),:].shape)
df_new.loc[(df_new.userID==243)&(df_new.ISBN.isin(['0060915544','0060977493','0316776963'])),:]
```

    (140516, 3)
    (56, 3)
    




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
      <th>userID</th>
      <th>ISBN</th>
      <th>bookRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>394</th>
      <td>243</td>
      <td>0060915544</td>
      <td>10</td>
    </tr>
    <tr>
      <th>395</th>
      <td>243</td>
      <td>0060977493</td>
      <td>7</td>
    </tr>
    <tr>
      <th>405</th>
      <td>243</td>
      <td>0316776963</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



s_data 를 이제 trainn_set 과 test_set 으로 나누어야 하나, 상기 243 유저의 rating 값의 변화가 궁금하므로, 일단 모두 train set 으로 변화시켜서 확인해보기로 한다.


```python
print(s_data.build_full_trainset().n_users*s_data.build_full_trainset().n_items)
t_set = s_data.build_full_trainset()
## traint_set 내부의 inner_id 가 뭔지 먼저 확인하면,
inner_id_243 = t_set.to_inner_uid(243)
print(len(t_set.ur[inner_id_243]))
```

    6878625
    56
    

tuple 로 ISBN , rating 값을 return 하지만, 이 역시, ISBN 이 아닌 내부 id로 ISBN 도 변환시킨 값이다. 따라서, 다시 변환해줘야 한다.

일단, train_set 클래스로 변화시키면, 내부 inner_id 가 ISBN 대신 부여된다.


```python
# t_set.to_raw_iid(1284)
for inner_id,rating in t_set.ur[inner_id_243]:
    if t_set.to_raw_iid(inner_id) in ['0060915544','0060977493','0316776963']:
        print(t_set.to_raw_iid(inner_id),rating)
```

    0060915544 10.0
    0060977493 7.0
    0316776963 9.0
    

rating_scale=(0, 9) 나, rating_scale=(0, 10) 이나, 10점은 10점 그대로 값을 가지고 있다. rating_scale 은 그냥 지정하는 것 같은데 왜 하는지 잘 모르겠다.
옆길로 샜지만, 다시금 기본 가정에 충실하여, train / test 셋 분리부터 해보자.


```python
## train test 셋으로 나누어서 해보기
from surprise import model_selection
s_data_train, s_data_test = model_selection.train_test_split(data=s_data,test_size=0.2,random_state=42,shuffle=True)
```


```python
from surprise import KNNBasic
from surprise import accuracy
```


```python
print('Using KNNBasic_user_based')
sim_options_userbase = {'name': 'cosine',
                        'user_based': True  # compute  similarities between users
                       }
algo = KNNBasic(k=10, min_k=1, sim_options={}, verbose=True)
```

    Using KNNBasic_user_based
    


```python
algo.fit(s_data_train)
sd01_result = algo.test(s_data_test) ## sckit-learin 으로 치면, predict_proba (surprise 에서, predict 는 특정 id , item 으로 rating 예측값을 볼때 사용한다.)
```

    Computing the msd similarity matrix...
    Done computing similarity matrix.
    


```python
print(type(sd01_result[0]))
accuracy.rmse(sd01_result,verbose=True)
```

    <class 'surprise.prediction_algorithms.predictions.Prediction'>
    RMSE: 3.7634
    




    3.7634240148853655




```python
## surprise package 내에는 cross_validation 기능이 잘 되어있다.
from surprise.model_selection import cross_validate
## cross_validate 활용하기
cross_validate(algo, data=s_data, measures=['RMSE'], cv=5, verbose=False)
```

    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    




    {'test_rmse': array([3.75437251, 3.74949873, 3.76960853, 3.76278959, 3.73663112]),
     'fit_time': (0.539557933807373,
      0.5754616260528564,
      0.5734672546386719,
      0.5634937286376953,
      0.5764591693878174),
     'test_time': (1.845094919204712,
      1.9328339099884033,
      1.9368364810943604,
      1.9084382057189941,
      1.949164628982544)}



item_based 로 구해보면,


```python
print('Using KNNBasic_item_base')
sim_options_itembase = {'name': 'cosine',
                        'user_based': False  # compute  similarities between items
                       }
algo = KNNBasic(k=10, min_k=1, sim_options={}, verbose=True)
```

    Using KNNBasic_item_base
    


```python
algo.fit(s_data_train)
sd01_result = algo.test(s_data_test) ## sckit-learin 으로 치면, predict_proba (surprise 에서, predict 는 특정 id , item 으로 rating 예측값을 볼때 사용한다.)
```

    Computing the msd similarity matrix...
    Done computing similarity matrix.
    


```python
print(type(sd01_result[0]))
accuracy.rmse(sd01_result,verbose=True)
```

    <class 'surprise.prediction_algorithms.predictions.Prediction'>
    RMSE: 3.7634
    




    3.7634240148853655



User base 나, Item base 나 큰 차이는 없는걸로....

**KNNWithMeans**  
- KNNWithMeans is basic collaborative filtering algorithm, taking into account the mean ratings of each user.

[document KNNWithMeans](https://surprise.readthedocs.io/en/stable/knn_inspired.html)


```python
from surprise import KNNWithMeans
options_itembase = {'name': 'cosine',
                    'user_based': False  # compute  similarities between items
                    }
algo = KNNWithMeans(k=12, min_k=3, sim_options=options_itembase)
results = cross_validate(algo, s_data, measures=['RMSE'], cv=3, verbose=False)     ## RMSE : 평균제곱근편차
print(results['test_rmse'])
```

    Computing the cosine similarity matrix...
    

    C:\ProgramData\Anaconda3\envs\test\lib\site-packages\surprise\prediction_algorithms\algo_base.py:248: RuntimeWarning: invalid value encountered in double_scalars
      sim = construction_func[name](*args)
    

    Done computing similarity matrix.
    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    [3.60952312 3.62560815 3.61105939]
    

**KNNWithZScore**  
- KNNWithZScore is a basic collaborative filtering algorithm, taking into account the z-score normalization of each user.
> 활용은 비슷하니 생략한다.

**KNNBaseline**  
- KNNBaseline is a basic collaborative filtering algorithm taking into account a baseline rating.
> 하기 처럼, Basic 01 post 에서 다, 개인의 bias 를 뺀체 계산한다.

![image.png](/assets/images/Surprise_basic_02/knnbaseline.png)


```python
from surprise import KNNBaseline
options_itembase = {'name': 'cosine',
                    'user_based': False  # compute  similarities between items
                    }
bsl_option = {'method': 'als','n_epochs': 10,'reg_u': 12,'reg_i': 5}

algo = KNNBaseline(k=20, min_k=4, sim_options=options_itembase, bsl_options=bsl_option, verbose=True)
results = cross_validate(algo, s_data, measures=['RMSE'], cv=3, verbose=False)     ## RMSE : 평균제곱근편차
print(results['test_rmse'])
```

    Estimating biases using als...
    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    [3.45982023 3.45636238 3.46900723]
