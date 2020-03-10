---
title:  "Recommendation Basic03"
excerpt: "Recommendation system using surprise library"

categories:
  - Machine-Learning
tags:
  - ML  
  - surprise  
  - Medium/susanli
  - Recommendation
last_modified_at: 2020-03-10T18:18:00-21:00
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
이미 이전 포스팅에서 다루었으니, 생략한다.


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style="white", context="talk")
```

대략 62%에 달하는 user 들이 rating에 0 점을 부여했다. 이 0점은 과연 book 평점을 최악을 준걸까? 아니면, 평점 자체를 달지 않은걸까? 어떻게 봐야하나?

책별로, 평점이 가장 많이 달린걸 보자면...


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
## Dataset 모듈에서, Reader class를 param으로 사용한다.
reader = Reader(rating_scale=(0, 9)) ## 실제 data 는 0~10 까지인데...왜 susan li 는 0~9 로 했을까? --> 이상하지만, 10점은 그래도 10점으로 변환된 값을 가지고 있따.
## (rating_scale=(0, 10) 으로 해도 크게 달라지는 건 없다.
s_data = Dataset.load_from_df(df_new[['userID', 'ISBN', 'bookRating']], reader)
```


```python
## train test 셋으로 나누어서 해보기
from surprise import model_selection
s_data_train, s_data_test = model_selection.train_test_split(data=s_data,test_size=0.2,random_state=42,shuffle=True)
```


```python
from surprise import accuracy
```

We use rmse as our accuracy metric for the predictions.

https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html 상세한 설명은 여기에

여기서부터는 surprise의 여러 알고리즘이 등장하는데, Susan Li 의 설명을 보고도 몰라서, 그냥 내가 하나하나 정리하면서, 올린다.

Basic01 post 에 남겼다.  
하기는 수식 notaion에 대한 설명

![image.png](/assets/images/Surprise_basic_02/notation.png)

갑자기 기억이 안나서 찾아본다....  원본 : https://goofcode.github.io/similarity-measure  
토막상식


```python
df_new.bookRating.value_counts().sort_index(ascending=False)[0:5]
```




    10     8778
    9      7966
    8     10381
    7      6694
    6      2917
    Name: bookRating, dtype: int64



# Basic algorithms  
- NormalPredictor / BaselineOnly
- 이전 POST 참고

# k-NN algorithms  
- KNNBasic / KNNWithMeans /**KNNBaseline**
- 이전 POST 참고

[학문적 이해에 대해 탁월한 주소](http://sanghyukchun.github.io/31/)

# Matrix Factorization-based algorithms
**SVD**
- SVD algorithm is equivalent to Probabilistic Matrix Factorization (http://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)
- 제일 유명한 알고리즘이며, Netflix Prize에서 처음 선보였다. 
- 단일 알고리즘으로는 가장 우수한 성능을 낸다고 알려져있다.

![image.png](/assets/images/Surprise_basic_03/svd_01.png)

유명한 이론이기고 나에게는 어렵기에 상세히는 별도 문서를 찾아볼 것을 권한다. 일단 나를 위해 기록을 남기자면, 위의 공식은 surprise 에 document에 있는 내용이다.  
그런, 이것만으로는 어뜻 이해가 가지 않는다. 실제 SVD  = Sigular Value Decomposition 부터 이해하고, 접근하기게 나은 방법이다.  
[고유값의 이해 링크](https://darkpgmr.tistory.com/106)
- R = P.Q (원 matrix 가 R 이면, 이는 P.Q 로도 분해될 수 있다. 

**고유값 분해정의**

![image.png](/assets/images/Surprise_basic_03/eigint_decomposition.png)

![image.png](/assets/images/Surprise_basic_03/eigint_decomposition_02.png)

**SVD 수학적 정의**

![image.png](/assets/images/Surprise_basic_03/original_00.png)

비유하자면, A 가 m*n 행렬 (유저-아템-rating) 매트릭스라고 생각하면 된다.

이를 풀이하면, A는 직교행렬이기에 바로 eigen-vector-decomposition이 되지 못한다. 그래서, AAt 또는 AtA 로 정방행렬이 된 상태에서,  
고유값 분해를 하게 된다.  
(※ 정방행렬이 고유값 분해가 항상 가능한 것은 아니지만 대칭행렬은 항상 고유값 분해가 가능하며 더구나 직교행렬로 대각화가 가능함을 기억하자.)    
그렇게 되면, 상기처럼 고유값분해가 이루이고 그 와중에 U,V,∑ 이 정의된다.

U : m*m 직교행렬 A의 left singular vector [= AA(transform) eigenvector] 
> AAt 를 고유값 분해하면, 식 (7) 처럼 U는 고육벡터, (∑∑t) 는 고유치 대각행렬 Ut 는 U역행렬 이다. (U는 직교행렬이기에 U역행렬=Ut 이다.)
> P 는 AAt 의 고유벡터, ∧ 는 고유치 대각행렬이다.

V : m*n 직사각 대각행렬 A의 right sigular vector [=(ATA의 eigenvector))]


∑ : n*n 행렬 A의 고유값 대각행렬에 square root 를 씌운값  

SVD 를 완전히, 상기 처럼, 수학정 정의에 의해서 분해하는 것을 full SVD 라고 하는데, 실상 python surprise 에 적용된 것과 같이, 실전에서는 Full SVD 를 사용하지 않는다.  
어느정도 0이 아닌 고유값들만을 살린 행렬로 가져가는데,  
Full SVD -> thin SVD -> Compat SVD -> Truncate SVD 로 축소해서, 원 A가 아닌 A' 로 근사해서 값을 해를 찾는다.  
surprise에서도 엄밀히는 Truncate SVD 를 이용하고 있다.  그림으로 살펴보면

![image.png](/assets/images/Surprise_basic_03/svd_02.png)

다시, python surprise로 돌아와서, P,Q 의미를 생각하면,  
P = U∑ , R = ∑tVt 란 생각이 든다. 왜냐하면, 최초에 R=n*m / P=n*k / Q=m*k 이 적절한 해석일 것이다.  
아래의 SVD 클래스에서, 기본적으로, n_factors 의 요소가 k 를 의미한다. 얼마나 고유값 대각행렬을 결정할 (차원축소) 할 것이냐는 얘기이기도 하다


```python
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
```


```python
from surprise import SVD,SVDpp,NMF

algo = SVD(n_factors=150,
           n_epochs=20,
           biased=True, ## False 로 하면, Probabilistic Matrix Factorization 이론과 동일해진다. 즉, 정말 Sigular value decomposition 해 풀이로 들어간다.
                       ## 이 의미는 True 일경우, baseline 의 개념이 결합이 되어, 목적함수가 정해지는 의미이다.
           init_mean=0, ## The mean of the normal distribution for factor vectors initialization. Default is 0.
           init_std_dev=0.1, ## The standard deviation of the normal distribution for factor vectors initialization. Default is 0.1.
           lr_all=0.005, ## The learning rate for all parameters. Default is 0.005.
           reg_all=0.02, ## The regularization term for all parameters. Default is 0.02.
           lr_bu=None, ## The learning rate for bu. Takes precedence over lr_all if set. Default is None.
           lr_bi=None, ##  The learning rate for bi. Takes precedence over lr_all if set. Default is None.
           lr_pu=None, ## The learning rate for pu. Takes precedence over lr_all if set. Default is None.
           lr_qi=None, ## The learning rate for qi. Takes precedence over lr_all if set. Default is None.
           reg_bu=0.001, ##  The regularization term for bu. Takes precedence over reg_all if set. Default is None.
           reg_bi=0.001, ## The regularization term for bi. Takes precedence over reg_all if set.(reg_all설정보다 우선한다.) Default is None.
           reg_pu=0.001, ## The regularization term for pu. Takes precedence over reg_all if set. Default is None.
           reg_qi=0.001, ## The regularization term for qi. Takes precedence over reg_all if set. Default is None.
           random_state=None,verbose=False,)
```


```python
results = cross_validate(algo, s_data, measures=['RMSE'], cv=3, verbose=False)     ## RMSE : 평균제곱근편차
print(results['test_rmse'])
```

    [3.5485304  3.52974876 3.52158998]
    

상기 surprise 의 경우는 baseline 이론과도 결합되어 있다. 따라서, 조절해줘야 하는 hyper parameter 가 많다.  
완전히 수학적의미로, 활용하려면, biased = Fasle 로 하면되는데. 논문에 따르면, 과하게 overfitting 되기 때문에, 선호되는 방법은 아니라고 하니...따라하자.  
당연하겠지만, 정규화 term 을 넣은 상태로는 그닥 결과가 좋지 않다. default 로 했더니, 좀더 낫다....

**SVDpp**  
- The SVDpp algorithm is an extension of SVD that takes into account implicit ratings. 
- 수식이 이해가 잘 안간다....


```python
algo = SVDpp(n_factors=20,
             n_epochs=20,
             init_mean=0,
             init_std_dev=0.1,
             lr_all=0.007,
             reg_all=0.01,
             lr_bu=None,
             lr_bi=None,
             lr_pu=None,
             lr_qi=None,
             lr_yj=None,
             reg_bu=None,
             reg_bi=None,
             reg_pu=None,
             reg_qi=None,
             reg_yj=None, ## – The regularization term for yj. Takes precedence over reg_all if set. Default is None.
             random_state=None,verbose=False)
```


```python
results = cross_validate(algo, s_data, measures=['RMSE'], cv=3, verbose=False)     ## RMSE : 평균제곱근편차
print(results['test_rmse'])
```

    [3.83129776 3.83604476 3.84379468]
    

**NMF**  
- NMF is a collaborative filtering algorithm based on Non-negative Matrix Factorization. It is very similar with SVD.
- 이건 생략

# Slope One  
- Slope One is a straightforward implementation of the SlopeOne algorithm. (https://arxiv.org/abs/cs/0702144)
- CF 알고리즘의 가장 단순한 형태. simulatiry 를 평균으로 구하는 방법이다.


```python
from surprise import SlopeOne
from surprise import CoClustering
algo = SlopeOne()
results = cross_validate(algo, s_data, measures=['RMSE'], cv=3, verbose=False)     ## RMSE : 평균제곱근편차
print(results['test_rmse'])
```

    [3.47530464 3.47201827 3.46743209]
    

# Co-clustering
- Co-clustering is a collaborative filtering algorithm based on co-clustering (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.6458&rep=rep1&type=pdf) We use rmse as our accuracy metric for the predictions.
- neigbor 를 설정할때 clustering 을 이용하는 방법이다


```python
algo = CoClustering( n_cltr_u=5,n_cltr_i=5,
                    n_epochs=20,random_state=None,verbose=False)
results = cross_validate(algo, s_data, measures=['RMSE'], cv=3, verbose=False)     ## RMSE : 평균제곱근편차
print(results['test_rmse'])
```

    [3.53167786 3.53866095 3.53000255]
    

모든 방법을 몇개는 건너띄고 살펴보았다. 현 data 에서, 어떤 방안을 사용할까를 고민해서, 모두 활용하기 위해 for 문으로 아래와 같이 실행한다.


```python
## 모든 객체와 라이브러니는 surprise package 안에 있는 것을 사용한다.
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
benchmark = [] ## 무사통과 알고리즘 : 
# Iterate over all algorithms  ## 에러나는 알고리즘 NMF(),
%time
for algorithm in [ SVD(), SVDpp(), SlopeOne(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    print(algorithm)
    results = cross_validate(algorithm, s_data, measures=['RMSE'], cv=3, verbose=False)     ## RMSE : 평균제곱근편차
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
```

    Wall time: 0 ns
    <surprise.prediction_algorithms.matrix_factorization.SVD object at 0x0000017421E6C9E8>
    <surprise.prediction_algorithms.matrix_factorization.SVDpp object at 0x0000017421E6CAC8>
    <surprise.prediction_algorithms.slope_one.SlopeOne object at 0x0000017421E6CA90>
    <surprise.prediction_algorithms.random_pred.NormalPredictor object at 0x0000017421E6CB00>
    <surprise.prediction_algorithms.knns.KNNBaseline object at 0x0000017421E6CB38>
    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    <surprise.prediction_algorithms.knns.KNNBasic object at 0x0000017421E6CB70>
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    <surprise.prediction_algorithms.knns.KNNWithMeans object at 0x0000017421E6CBA8>
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    <surprise.prediction_algorithms.knns.KNNWithZScore object at 0x0000017421E6CBE0>
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    <surprise.prediction_algorithms.baseline_only.BaselineOnly object at 0x0000017421E6CC18>
    Estimating biases using als...
    Estimating biases using als...
    Estimating biases using als...
    <surprise.prediction_algorithms.co_clustering.CoClustering object at 0x0000017421E6CC50>
    

의외로 결과가, 심플한 알고리즘에 속하는 baseline 이 좋게 나왔다...  
latent matrix 쪽은 다소 baseline 보다는 약하게 나옴


```python
surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
surprise_results
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
      <th>test_rmse</th>
      <th>fit_time</th>
      <th>test_time</th>
    </tr>
    <tr>
      <th>Algorithm</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BaselineOnly</th>
      <td>3.374667</td>
      <td>0.192807</td>
      <td>0.196814</td>
    </tr>
    <tr>
      <th>CoClustering</th>
      <td>3.472809</td>
      <td>1.557530</td>
      <td>0.258480</td>
    </tr>
    <tr>
      <th>SlopeOne</th>
      <td>3.476510</td>
      <td>0.551533</td>
      <td>3.308080</td>
    </tr>
    <tr>
      <th>KNNWithMeans</th>
      <td>3.482859</td>
      <td>0.514802</td>
      <td>4.464338</td>
    </tr>
    <tr>
      <th>KNNBaseline</th>
      <td>3.496288</td>
      <td>0.654103</td>
      <td>5.382493</td>
    </tr>
    <tr>
      <th>KNNWithZScore</th>
      <td>3.505984</td>
      <td>0.609689</td>
      <td>4.832343</td>
    </tr>
    <tr>
      <th>SVD</th>
      <td>3.543556</td>
      <td>4.425603</td>
      <td>0.290065</td>
    </tr>
    <tr>
      <th>KNNBasic</th>
      <td>3.727995</td>
      <td>0.487108</td>
      <td>4.100230</td>
    </tr>
    <tr>
      <th>SVDpp</th>
      <td>3.798871</td>
      <td>105.043733</td>
      <td>4.350798</td>
    </tr>
    <tr>
      <th>NormalPredictor</th>
      <td>4.681728</td>
      <td>0.120685</td>
      <td>0.270594</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Using ALS')
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
algo = BaselineOnly(bsl_options=bsl_options)
cross_validate(algo, s_data, measures=['RMSE'], cv=3, verbose=False)

trainset, testset = train_test_split(s_data, test_size=0.25)
algo = BaselineOnly(bsl_options=bsl_options)
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)
```

    Using ALS
    Estimating biases using als...
    Estimating biases using als...
    Estimating biases using als...
    Estimating biases using als...
    RMSE: 3.3756
    




    3.3755915082844634




```python
trainset = algo.trainset ## fit 할때, 저장된다.
print(algo.__class__.__name__)
```

    BaselineOnly
    

데이터 결과분석을 하기 위해, Susan Li 와 동일하게 실행해보면,


```python
def get_II(uid): ## 실제 user id 가 trainset 데이터셋에서, 어떤 raiting을 준, 책 갯수 (평점을 줬는지에 대한)
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid): ## trainset 에서 아이템별 '책'별로 얼마다 평점이 달려있는지 보는 갯수
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
```


```python
print(type(predictions[0]),'\t',predictions[20])
## predictions[0] dictionary 처럼 사용할 수는 없다...그냥 tuple 처럼, 값이 나온다고 생각하고, DataFrame 으로 변경해야한다.
```

    <class 'surprise.prediction_algorithms.predictions.Prediction'> 	 user: 10819      item: 0671003755 r_ui = 0.00   est = 1.99   {'was_impossible': False}
    


```python
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Ii_cnt'] = df.uid.apply(get_II) ## 실제 user id 가 trainset 데이터셋에서, 어떤 raiting을 준, 책 갯수 (평점을 줬는지에 대한)
df['Ui_cnt'] = df.iid.apply(get_Ui) ## trainset 에서 아이템별 '책'별로 얼마다 평점이 달려있는지 보는 갯수
df['err'] = abs(df.est - df.rui)
```


```python
df.head()
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
      <th>uid</th>
      <th>iid</th>
      <th>rui</th>
      <th>est</th>
      <th>details</th>
      <th>Ii_cnt</th>
      <th>Ui_cnt</th>
      <th>err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31315</td>
      <td>0802139256</td>
      <td>0.0</td>
      <td>2.564510</td>
      <td>{'was_impossible': False}</td>
      <td>246</td>
      <td>48</td>
      <td>2.564510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>242073</td>
      <td>0446608890</td>
      <td>0.0</td>
      <td>1.405207</td>
      <td>{'was_impossible': False}</td>
      <td>45</td>
      <td>68</td>
      <td>1.405207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>227447</td>
      <td>0446609617</td>
      <td>0.0</td>
      <td>0.384353</td>
      <td>{'was_impossible': False}</td>
      <td>315</td>
      <td>38</td>
      <td>0.384353</td>
    </tr>
    <tr>
      <th>3</th>
      <td>196077</td>
      <td>0440204887</td>
      <td>0.0</td>
      <td>1.472857</td>
      <td>{'was_impossible': False}</td>
      <td>286</td>
      <td>31</td>
      <td>1.472857</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49842</td>
      <td>0553582747</td>
      <td>0.0</td>
      <td>2.458348</td>
      <td>{'was_impossible': False}</td>
      <td>19</td>
      <td>63</td>
      <td>2.458348</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df.rui!=0,:].head() ## 일단, test 셋이니, 실제 rating 값과, est 값은 모두 나오게 된다.
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
      <th>uid</th>
      <th>iid</th>
      <th>rui</th>
      <th>est</th>
      <th>details</th>
      <th>Ii_cnt</th>
      <th>Ui_cnt</th>
      <th>err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>116758</td>
      <td>0312195516</td>
      <td>10.0</td>
      <td>5.627643</td>
      <td>{'was_impossible': False}</td>
      <td>19</td>
      <td>264</td>
      <td>4.372357</td>
    </tr>
    <tr>
      <th>15</th>
      <td>18082</td>
      <td>0440221471</td>
      <td>8.0</td>
      <td>2.612378</td>
      <td>{'was_impossible': False}</td>
      <td>25</td>
      <td>171</td>
      <td>5.387622</td>
    </tr>
    <tr>
      <th>22</th>
      <td>123981</td>
      <td>0425121259</td>
      <td>8.0</td>
      <td>2.001490</td>
      <td>{'was_impossible': False}</td>
      <td>225</td>
      <td>42</td>
      <td>5.998510</td>
    </tr>
    <tr>
      <th>25</th>
      <td>165183</td>
      <td>0060976845</td>
      <td>10.0</td>
      <td>4.094378</td>
      <td>{'was_impossible': False}</td>
      <td>25</td>
      <td>187</td>
      <td>5.905622</td>
    </tr>
    <tr>
      <th>30</th>
      <td>100459</td>
      <td>037570504X</td>
      <td>8.0</td>
      <td>2.692422</td>
      <td>{'was_impossible': False}</td>
      <td>90</td>
      <td>70</td>
      <td>5.307578</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_predictions = df.sort_values(by='err')[:10] ## 제일 작은순으로, 앞에서부터 10 개
worst_predictions = df.sort_values(by='err')[-10:] ## 제일 작은순으로 했을때 뒤에서 부터 10개
```


```python
best_predictions
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
      <th>uid</th>
      <th>iid</th>
      <th>rui</th>
      <th>est</th>
      <th>details</th>
      <th>Ii_cnt</th>
      <th>Ui_cnt</th>
      <th>err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9837</th>
      <td>79942</td>
      <td>0971880107</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>23</td>
      <td>627</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7849</th>
      <td>225810</td>
      <td>0394742117</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>214</td>
      <td>30</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28966</th>
      <td>82926</td>
      <td>0345386108</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>40</td>
      <td>65</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28951</th>
      <td>102967</td>
      <td>0425172996</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>396</td>
      <td>45</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7923</th>
      <td>234623</td>
      <td>0553585118</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>242</td>
      <td>29</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28916</th>
      <td>35050</td>
      <td>0446607711</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>126</td>
      <td>81</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7945</th>
      <td>98741</td>
      <td>0064405842</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>239</td>
      <td>30</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8018</th>
      <td>78783</td>
      <td>0440224845</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>356</td>
      <td>38</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8074</th>
      <td>73394</td>
      <td>0345447840</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>234</td>
      <td>58</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8091</th>
      <td>102967</td>
      <td>0440201926</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>{'was_impossible': False}</td>
      <td>396</td>
      <td>47</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
worst_predictions
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
      <th>uid</th>
      <th>iid</th>
      <th>rui</th>
      <th>est</th>
      <th>details</th>
      <th>Ii_cnt</th>
      <th>Ui_cnt</th>
      <th>err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33276</th>
      <td>245827</td>
      <td>0451183665</td>
      <td>10.0</td>
      <td>0.304337</td>
      <td>{'was_impossible': False}</td>
      <td>122</td>
      <td>90</td>
      <td>9.695663</td>
    </tr>
    <tr>
      <th>12093</th>
      <td>245864</td>
      <td>0345409876</td>
      <td>10.0</td>
      <td>0.293552</td>
      <td>{'was_impossible': False}</td>
      <td>28</td>
      <td>34</td>
      <td>9.706448</td>
    </tr>
    <tr>
      <th>31016</th>
      <td>69697</td>
      <td>0425183971</td>
      <td>10.0</td>
      <td>0.269747</td>
      <td>{'was_impossible': False}</td>
      <td>149</td>
      <td>45</td>
      <td>9.730253</td>
    </tr>
    <tr>
      <th>10868</th>
      <td>245963</td>
      <td>0425170349</td>
      <td>10.0</td>
      <td>0.267484</td>
      <td>{'was_impossible': False}</td>
      <td>128</td>
      <td>49</td>
      <td>9.732516</td>
    </tr>
    <tr>
      <th>6584</th>
      <td>77940</td>
      <td>0671027581</td>
      <td>10.0</td>
      <td>0.259088</td>
      <td>{'was_impossible': False}</td>
      <td>52</td>
      <td>43</td>
      <td>9.740912</td>
    </tr>
    <tr>
      <th>17718</th>
      <td>14521</td>
      <td>0553269631</td>
      <td>10.0</td>
      <td>0.219905</td>
      <td>{'was_impossible': False}</td>
      <td>174</td>
      <td>27</td>
      <td>9.780095</td>
    </tr>
    <tr>
      <th>31077</th>
      <td>227447</td>
      <td>0515132268</td>
      <td>10.0</td>
      <td>0.163739</td>
      <td>{'was_impossible': False}</td>
      <td>315</td>
      <td>32</td>
      <td>9.836261</td>
    </tr>
    <tr>
      <th>12162</th>
      <td>238120</td>
      <td>0385413041</td>
      <td>10.0</td>
      <td>0.000000</td>
      <td>{'was_impossible': False}</td>
      <td>327</td>
      <td>31</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>8109</th>
      <td>227447</td>
      <td>055356773X</td>
      <td>10.0</td>
      <td>0.000000</td>
      <td>{'was_impossible': False}</td>
      <td>315</td>
      <td>48</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>20402</th>
      <td>55548</td>
      <td>0553278398</td>
      <td>10.0</td>
      <td>0.000000</td>
      <td>{'was_impossible': False}</td>
      <td>134</td>
      <td>24</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>



전체 성능을 3.374667 로 알고 있기에 이를 기준으로 판단해야 한다.

완전히 예측 rating 이 틀린 경우에 대해서 모델이 잘못되었다고 할 수 있을까? 예를 들어, 특정 책 0515132268 은 월등히 0 점 맞은 비율이 높은데,  
실제로 227447 유저가 10점을 주었다는건....이 사람이 더 특이한 취양이 아닐까...약간 이상치 느낌이 아닐까란 생각을 하게 된다.


```python
print(df.loc[df.iid=='0515132268',:].shape)
df.loc[df.iid=='0515132268',:].rui.value_counts()
```

    (15, 8)
    




    0.0     11
    10.0     1
    5.0      1
    8.0      1
    9.0      1
    Name: rui, dtype: int64




```python
df.loc[df.iid=='0515132268',:]
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
      <th>uid</th>
      <th>iid</th>
      <th>rui</th>
      <th>est</th>
      <th>details</th>
      <th>Ii_cnt</th>
      <th>Ui_cnt</th>
      <th>err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>350</th>
      <td>11676</td>
      <td>0515132268</td>
      <td>9.0</td>
      <td>4.399784</td>
      <td>{'was_impossible': False}</td>
      <td>1132</td>
      <td>32</td>
      <td>4.600216</td>
    </tr>
    <tr>
      <th>4498</th>
      <td>94923</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>0.881753</td>
      <td>{'was_impossible': False}</td>
      <td>45</td>
      <td>32</td>
      <td>0.881753</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>159033</td>
      <td>0515132268</td>
      <td>8.0</td>
      <td>0.631298</td>
      <td>{'was_impossible': False}</td>
      <td>155</td>
      <td>32</td>
      <td>7.368702</td>
    </tr>
    <tr>
      <th>6053</th>
      <td>70065</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>1.186498</td>
      <td>{'was_impossible': False}</td>
      <td>28</td>
      <td>32</td>
      <td>1.186498</td>
    </tr>
    <tr>
      <th>7656</th>
      <td>136382</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>1.875635</td>
      <td>{'was_impossible': False}</td>
      <td>83</td>
      <td>32</td>
      <td>1.875635</td>
    </tr>
    <tr>
      <th>7930</th>
      <td>165319</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>1.732530</td>
      <td>{'was_impossible': False}</td>
      <td>74</td>
      <td>32</td>
      <td>1.732530</td>
    </tr>
    <tr>
      <th>12488</th>
      <td>234597</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>1.992210</td>
      <td>{'was_impossible': False}</td>
      <td>48</td>
      <td>32</td>
      <td>1.992210</td>
    </tr>
    <tr>
      <th>18132</th>
      <td>123095</td>
      <td>0515132268</td>
      <td>5.0</td>
      <td>3.708091</td>
      <td>{'was_impossible': False}</td>
      <td>33</td>
      <td>32</td>
      <td>1.291909</td>
    </tr>
    <tr>
      <th>21573</th>
      <td>135351</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>3.109227</td>
      <td>{'was_impossible': False}</td>
      <td>22</td>
      <td>32</td>
      <td>3.109227</td>
    </tr>
    <tr>
      <th>22122</th>
      <td>55492</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>0.175138</td>
      <td>{'was_impossible': False}</td>
      <td>374</td>
      <td>32</td>
      <td>0.175138</td>
    </tr>
    <tr>
      <th>25194</th>
      <td>51450</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>1.163447</td>
      <td>{'was_impossible': False}</td>
      <td>82</td>
      <td>32</td>
      <td>1.163447</td>
    </tr>
    <tr>
      <th>26347</th>
      <td>151589</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>0.381658</td>
      <td>{'was_impossible': False}</td>
      <td>39</td>
      <td>32</td>
      <td>0.381658</td>
    </tr>
    <tr>
      <th>26768</th>
      <td>8936</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>0.311826</td>
      <td>{'was_impossible': False}</td>
      <td>89</td>
      <td>32</td>
      <td>0.311826</td>
    </tr>
    <tr>
      <th>31077</th>
      <td>227447</td>
      <td>0515132268</td>
      <td>10.0</td>
      <td>0.163739</td>
      <td>{'was_impossible': False}</td>
      <td>315</td>
      <td>32</td>
      <td>9.836261</td>
    </tr>
    <tr>
      <th>32133</th>
      <td>243930</td>
      <td>0515132268</td>
      <td>0.0</td>
      <td>2.573211</td>
      <td>{'was_impossible': False}</td>
      <td>35</td>
      <td>32</td>
      <td>2.573211</td>
    </tr>
  </tbody>
</table>
</div>




```python
inner_uid = trainset.to_inner_uid(11676)
len(trainset.ur[inner_uid]) ## return (item_inner_id, rating)
```




    1132



틈틈이 update 예정  
End
