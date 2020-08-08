---
title:  "Text Analyis Basic using Scikit-Learn and KoNLPy (python 텍스트 분석 03)"
excerpt: "Sckit-lean library 과 Keras를 사용한 Text 분석 비교"

categories:
  - Deep-Learning
tags:
  - KoNLPy
  - text analysis
  - 머신러닝
  - linux docker
last_modified_at: 2020-03-21T16:13:00-05:00
---

## 이 분석은 개인적인 local 사정상, Docker 환경에서 실습했음을 알린다.  
### Docker 환경을 만든 이유는 local OS : Window 10 Home edition 이기 때문이다.  
 - docker toolbox 로 진행했는데, 꽤나 힘들었다.
KoNLPy 순서로 진행한다.

이 외에도, KoNLPy (형태소 분석기 = 한국어 전용 어간분석기) 를 사용할 수 있고, 실제로도 이를 많이 사용한다.  
그러나, 현재 실습 환경이 Window 인 관계로, 생략한다. 이는 추후 Collab 에서 활용하도록 한다  
"KoNLPy의 Mecab() 클래스는 윈도우에서 지원되지 않습니다." (http://konlpy.org/ko/latest/install/)


```python
## 라이브러리 로드

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

%matplotlib inline
# 시각화 결과가 선명하게 표시되도록
%config InlineBackend.figure_format = 'retina'
```

## 시각화를 위한 한글폰트 설정


```python
# Window 한글폰트 설정
# plt.rc("font", family="Malgun Gothic")
# Mac 한글폰트 설정
plt.rc("font", family="AppleGothic")
plt.rc('axes', unicode_minus=False)
```

## Naver Movie Review 가져오기  
[이미 github 에 txt 파일로 정제된것을 활용했다.](https://github.com/e9t/nsmc)


```python
df_train = pd.read_csv('/home/cypision/Alchemy/dataset/naver_movie_sample/ratings_train.txt',delimiter='\t',keep_default_na=False)
df_train.head(2)
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
      <th>id</th>
      <th>document</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9976970</td>
      <td>아 더빙.. 진짜 짜증나네요 목소리</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3819312</td>
      <td>흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



lable 이 0 이면 부정적인 리뷰. 1이면 긍정적인 리뷰다  
numpy 배열로 바꿔보면


```python
text_train = df_train.document.values
y_train = df_train.label.values
```


```python
print(text_train.shape,y_train.shape)
```

    (150000,) (150000,)



```python
print(type(text_train),text_train.ndim)
```

    <class 'numpy.ndarray'> 1



```python
## text data 불러와서 가공하기
df_test = pd.read_csv('/home/cypision/Alchemy/dataset/naver_movie_sample/ratings_test.txt',delimiter='\t',keep_default_na=False)
text_test = df_test['document'].values
y_test = df_test.label.values
```

## 데이터 탐색하기


```python
len(text_train), np.bincount(y_train)
```




    (150000, array([75173, 74827]))




```python
len(text_test), np.bincount(y_test)
```




    (50000, array([24827, 25173]))



## KoNLPy 를 tokenizer 로 활용하기  
KoNLPy 는 앞선 post 에서 언급했던, 형태소 분석기이고 기본적으로 5개 정도가 있다.  
여기서는 2개 정도만 실습해보기로 한다.

__Okt 이른바 Twitter 형태소 분석기 활용__


```python
from konlpy.tag import Okt ## Twitter --> Okt 로 버전업하면서 명칭이 바뀌었다.
twitter_tag = Okt()
```


```python
def twitter_tokenizer(text):
    return twitter_tag.morphs(text)
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
```


```python
twit_param_grid = {'tfidfvectorizer__min_df':[3,5,7],
                  'tfidfvectorizer__ngram_range':[(1,1),(1,2),(1,3)],
                  'logisticregression__C':[0.1,1,10]}
## 여기가 핵심이다.
## 기본 TfidfVectorizer() 는 정규식을 활용하자만, 여기서는 KoNLPy 에 해당하는 어간(형태소) tokenizer 를 사용한다.
t_pipe = make_pipeline(TfidfVectorizer(tokenizer=twitter_tokenizer),LogisticRegression())
```


```python
t_grid = GridSearchCV(t_pipe,twit_param_grid)
```


```python
t_grid.fit(text_train[0:1000],y_train[0:1000])
```


```python
print(t_grid.best_score_)
print(t_grid.best_params_)
```

    0.718
    {'logisticregression__C': 1, 'tfidfvectorizer__min_df': 3, 'tfidfvectorizer__ngram_range': (1, 3)}


이제 최적의 조합을 찾았으니, text set를 변환시키고 (KoNLPy tokenzier 를 사용하고) 실제 test 결과를 구해본다


```python
t_grid.best_estimator_
```




    Pipeline(memory=None,
             steps=[('tfidfvectorizer',
                     TfidfVectorizer(analyzer='word', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.float64'>,
                                     encoding='utf-8', input='content',
                                     lowercase=True, max_df=1.0, max_features=None,
                                     min_df=3, ngram_range=(1, 3), norm='l2',
                                     preprocessor=None, smooth_idf=True,
                                     stop_words=None, strip_accents=None,
                                     sublinear_tf=False,
                                     token...
                                     tokenizer=<function twitter_tokenizer at 0x7f0b5545f268>,
                                     use_idf=True, vocabulary=None)),
                    ('logisticregression',
                     LogisticRegression(C=1, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=100,
                                        multi_class='auto', n_jobs=None,
                                        penalty='l2', random_state=None,
                                        solver='lbfgs', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)




```python
x_test_konlypy = t_grid.best_estimator_.named_steps['tfidfvectorizer'].transform(text_test)
```


```python
score = t_grid.best_estimator_.named_steps['logisticregression'].score(x_test_konlypy,y_test)
```


```python
print(score)
```

    0.70698


딱히 비교할 건 없지만, 꽤나 잘 맞는다

__Mecab 형태소 분석기 활용__


```python
from konlpy.tag import Mecab
mecab = Mecab()
```


```python
def mecab_tokenizer(text):
    return mecab.morphs(text)
```


```python
mecab_param_grid = {'tfidfvectorizer__min_df':[3,5,7],
                  'tfidfvectorizer__ngram_range':[(1,1),(1,2),(1,3)],
                  'logisticregression__C':[0.1,1,10]}
## 여기가 핵심이다.
## 기본 TfidfVectorizer() 는 정규식을 활용하자만, 여기서는 KoNLPy 에 해당하는 어간(형태소) tokenizer 를 사용한다.
m_pipe = make_pipeline(TfidfVectorizer(tokenizer=mecab_tokenizer),LogisticRegression())
```


```python
m_grid = GridSearchCV(m_pipe,mecab_param_grid)
```


```python
m_grid.fit(text_train[0:1000],y_train[0:1000])
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('tfidfvectorizer',
                                            TfidfVectorizer(analyzer='word',
                                                            binary=False,
                                                            decode_error='strict',
                                                            dtype=<class 'numpy.float64'>,
                                                            encoding='utf-8',
                                                            input='content',
                                                            lowercase=True,
                                                            max_df=1.0,
                                                            max_features=None,
                                                            min_df=1,
                                                            ngram_range=(1, 1),
                                                            norm='l2',
                                                            preprocessor=None,
                                                            smooth_idf=True,
                                                            stop_words=N...
                                                               n_jobs=None,
                                                               penalty='l2',
                                                               random_state=None,
                                                               solver='lbfgs',
                                                               tol=0.0001,
                                                               verbose=0,
                                                               warm_start=False))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'logisticregression__C': [0.1, 1, 10],
                             'tfidfvectorizer__min_df': [3, 5, 7],
                             'tfidfvectorizer__ngram_range': [(1, 1), (1, 2),
                                                              (1, 3)]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
print(m_grid.best_score_)
print(m_grid.best_params_)
```

    0.7529999999999999
    {'logisticregression__C': 1, 'tfidfvectorizer__min_df': 3, 'tfidfvectorizer__ngram_range': (1, 2)}



```python
m_grid.best_estimator_
```




    Pipeline(memory=None,
             steps=[('tfidfvectorizer',
                     TfidfVectorizer(analyzer='word', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.float64'>,
                                     encoding='utf-8', input='content',
                                     lowercase=True, max_df=1.0, max_features=None,
                                     min_df=3, ngram_range=(1, 2), norm='l2',
                                     preprocessor=None, smooth_idf=True,
                                     stop_words=None, strip_accents=None,
                                     sublinear_tf=False,
                                     token...
                                     tokenizer=<function mecab_tokenizer at 0x7f0b2f7a3378>,
                                     use_idf=True, vocabulary=None)),
                    ('logisticregression',
                     LogisticRegression(C=1, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=100,
                                        multi_class='auto', n_jobs=None,
                                        penalty='l2', random_state=None,
                                        solver='lbfgs', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)




```python
x_test_konlypy = m_grid.best_estimator_.named_steps['tfidfvectorizer'].transform(text_test)
```


```python
score = m_grid.best_estimator_.named_steps['logisticregression'].score(x_test_konlypy,y_test)
```


```python
print(score)
```

    0.74632


그렇다...이렇다 할 특별한 부분은 없지만...결과는 훨씬 좋은 듯. ㅋㅋ
