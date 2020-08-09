---
title:  "20 News Group Basic"  
excerpt: "Text Analysis"  

categories:  
  - Deep-Learning  
tags:  
  - Text Analysis
  - fetch_20newsgroups  
  - csr-matrix
  - CountVectorizer  
  - TfidfVectorizer
last_modified_at: 2020-08-09T14:13:00-05:00
---

## Reference  
* 파이썬 머신러닝 완벽가이드 - 권철민
* NCIA shkim.hi@gmail.com

## 20-뉴스그룹 분류

### 데이터 로딩과 데이터 구성 확인


```python
from sklearn.datasets import fetch_20newsgroups

news_data = fetch_20newsgroups(subset='all',random_state=156)
## 기본제공해주는 파라미터 
```


```python
print(type(news_data))
```

    <class 'sklearn.utils.Bunch'>
    

Bunch type : scikit-learn 쪽에서 주로 사용하는 Bunch type. dict 와 유사한 객체이다.


```python
print(news_data.keys())
```

    dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
    


```python
print(type(news_data.data), type(news_data.target_names), type(news_data.target))
```

    <class 'list'> <class 'list'> <class 'numpy.ndarray'>
    


```python
import numpy as np
print(np.unique(news_data.target))
print(news_data.target_names)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    


```python
for i,val in zip(np.unique(news_data.target),news_data.target_names):
    print("index ({}) : topic {} ".format(i,val))
```

    index (0) : topic alt.atheism 
    index (1) : topic comp.graphics 
    index (2) : topic comp.os.ms-windows.misc 
    index (3) : topic comp.sys.ibm.pc.hardware 
    index (4) : topic comp.sys.mac.hardware 
    index (5) : topic comp.windows.x 
    index (6) : topic misc.forsale 
    index (7) : topic rec.autos 
    index (8) : topic rec.motorcycles 
    index (9) : topic rec.sport.baseball 
    index (10) : topic rec.sport.hockey 
    index (11) : topic sci.crypt 
    index (12) : topic sci.electronics 
    index (13) : topic sci.med 
    index (14) : topic sci.space 
    index (15) : topic soc.religion.christian 
    index (16) : topic talk.politics.guns 
    index (17) : topic talk.politics.mideast 
    index (18) : topic talk.politics.misc 
    index (19) : topic talk.religion.misc 
    


```python
print(len(news_data.data), len(news_data.data[0]),len(news_data.data[1]))
print(len(news_data.target_names))
print(news_data.target.shape)
```

    18846 1303 2944
    20
    (18846,)
    


```python
print(news_data.data[0][:500])
```

    From: egreen@east.sun.com (Ed Green - Pixel Cruncher)
    Subject: Re: Observation re: helmets
    Organization: Sun Microsystems, RTP, NC
    Lines: 21
    Distribution: world
    Reply-To: egreen@east.sun.com
    NNTP-Posting-Host: laser.east.sun.com
    
    In article 211353@mavenry.altcit.eskimo.com, maven@mavenry.altcit.eskimo.com (Norman Hamer) writes:
    > 
    > The question for the day is re: passenger helmets, if you don't know for 
    >certain who's gonna ride with you (like say you meet them at a .... church 
    >meeting, yeah
    

## 데이터 설명  
- scikit-learn 내장 data  
- 영문 20개 topic data로, target 0 ~ 19 총 20개 topic으로 된 정답지가 있음  
- 18846 의 data(문단) 이 있으며, 각 문단별 길이는 모두 다름


```python
import pandas as pd

print('target 클래스의 값과 분포도 \n', pd.Series(news_data.target).value_counts().sort_index())
print('target 클래스의 이름들 \n',news_data.target_names)
len(news_data.target_names), pd.Series(news_data.target).shape
```

    target 클래스의 값과 분포도 
     0     799
    1     973
    2     985
    3     982
    4     963
    5     988
    6     975
    7     990
    8     996
    9     994
    10    999
    11    991
    12    984
    13    990
    14    987
    15    997
    16    910
    17    940
    18    775
    19    628
    dtype: int64
    target 클래스의 이름들 
     ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    




    (20, (18846,))



### 학습 & 테스트용 데이터 생성


```python
from sklearn.datasets import fetch_20newsgroups

# subset='train'으로 학습용(Train) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
# body 만 활용하기 위해 제거함
train_news= fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=156)
X_train = train_news.data
y_train = train_news.target
```


```python
print(type(X_train))
print(len(X_train))
print(len(y_train))
```

    <class 'list'>
    11314
    11314
    


```python
print(train_news.data[0][:500])
```

    
    
    What I did NOT get with my drive (CD300i) is the System Install CD you
    listed as #1.  Any ideas about how I can get one?  I bought my IIvx 8/120
    from Direct Express in Chicago (no complaints at all -- good price & good
    service).
    
    BTW, I've heard that the System Install CD can be used to boot the mac;
    however, my drive will NOT accept a CD caddy is the machine is off.  How can
    you boot with it then?
    
    --Dave
    
    


```python
# subset='test'으로 테스트(Test) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
test_news= fetch_20newsgroups(subset='test',remove=('headers', 'footers','quotes'),random_state=156)
X_test = test_news.data
y_test = test_news.target
print('학습 데이터 크기 {0} , 테스트 데이터 크기 {1}'.format(len(train_news.data) , len(test_news.data)))
```

    학습 데이터 크기 11314 , 테스트 데이터 크기 7532
    

### Count 피처 벡터화 변환과 머신러닝 모델 학습/예측/평가
* 주의: 학습 데이터에 대해 fit( )된 CountVectorizer를 이용해서 테스트 데이터를 피처 벡터화 해야함.   
* 테스트 데이터에서 다시 CountVectorizer의 fit_transform()을 수행하거나 fit()을 수행 하면 안됨.   
* 이는 이렇게 테스트 데이터에서 fit()을 수행하게 되면 기존 학습된 모델에서 가지는 feature의 갯수가 달라지기 때문임.

#### sklearn.feature_extraction.text.CountVectorizer  
[__CountVectorizer 설명__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)  
- parameter 간단요약  
> stop_words : 불용어 사전을 거치게 하는 것. english 중심이다.  
> tokenizer : 내가 만든 tokenizer 객체를 불러올 수 있다. default = None  
> ngram_range : n-gram 사용시 활용한다.  
> max_df : ex> max_df=100 일 경우, 특정 단어가 100 번 이상 (전체어휘사전기준) 등장하면, 그 단어는 수식어,관용어(a,an,the) 일 수 있으니, 강제로 특정 값이상이면 어휘사전 구성시 제외  
> min_df : 상기 max_df 컨셉의 반대  


```python
from sklearn.feature_extraction.text import CountVectorizer

# Count Vectorization으로 feature extraction 변환 수행. 
cnt_vect = CountVectorizer() ## max_df=100 . 100개 이상이 되는 단어는 무시한다... min_df : 반대의 의미
cnt_vect.fit(X_train)
X_train_cnt_vect = cnt_vect.transform(X_train) ## vector화 명령어

# 학습 데이터로 fit( )된 CountVectorizer를 이용하여 테스트 데이터를 feature extraction 변환 수행. 
X_test_cnt_vect = cnt_vect.transform(X_test) ## X_train 으로 완성한 어휘사전 기준으로 vectorize

print('학습 & 테스트 데이터 Text의 CountVectorizer Shape:',X_train_cnt_vect.shape, X_test_cnt_vect.shape)
```

    학습 & 테스트 데이터 Text의 CountVectorizer Shape: (11314, 101631) (7532, 101631)
    


```python
print(type(X_train_cnt_vect))
print(X_train_cnt_vect.todense().shape) ## csr_matrix 를 numpy matrix로 변환해준 이후, shape 확인
print(len(X_train_cnt_vect.todense()[0])) ## 메모리용량때문에 모두는 표현 불가
print(len(X_train_cnt_vect.todense()[1]))
```

    <class 'scipy.sparse.csr.csr_matrix'>
    (11314, 101631)
    1
    1
    

기본적으로 CSR_MATRIX 구조로 return한다.  


```python
print(X_train_cnt_vect[0].todense().shape) ## toarray() 도 가능
print(X_train_cnt_vect[1].todense().shape)
print(X_train_cnt_vect[21].todense().shape)
print(X_train_cnt_vect[0].toarray().shape) 
```

    (1, 101631)
    (1, 101631)
    (1, 101631)
    (1, 101631)
    

각 문단 크기는 columns : 101631 로 고정된것을 알 수 있다.


```python
print(type(cnt_vect), type(X_train_cnt_vect))
```

    <class 'sklearn.feature_extraction.text.CountVectorizer'> <class 'scipy.sparse.csr.csr_matrix'>
    


```python
print([i for i in cnt_vect.vocabulary_.items()][:5])
```

    [('what', 96391), ('did', 33551), ('not', 66511), ('get', 43217), ('with', 96917)]
    


```python
print(X_train_cnt_vect[0].shape)
```

    (1, 101631)
    


```python
print(X_train_cnt_vect.ndim)
print(X_train_cnt_vect.shape)
print(type(X_train_cnt_vect[0]))
print(X_train_cnt_vect[0]) ## tuple 로 구성된 csr_matrix 를 보여준다.
```

    2
    (11314, 101631)
    <class 'scipy.sparse.csr.csr_matrix'>
      (0, 2223)	1
      (0, 16251)	1
      (0, 16406)	1
      (0, 17936)	1
      (0, 18903)	1
      (0, 19756)	1
      (0, 20123)	1
      (0, 21987)	1
      (0, 23663)	2
      (0, 23790)	1
      (0, 24444)	1
      (0, 25370)	1
      (0, 25590)	3
      (0, 26271)	3
      (0, 26277)	1
      (0, 26992)	1
      (0, 28805)	1
      (0, 31939)	1
      (0, 33551)	1
      (0, 33799)	1
      (0, 35147)	2
      (0, 38824)	1
      (0, 41715)	1
      (0, 43217)	2
      (0, 43961)	2
      :	:
      (0, 49447)	1
      (0, 50300)	2
      (0, 51136)	3
      (0, 51326)	1
      (0, 56936)	1
      (0, 58921)	1
      (0, 58962)	1
      (0, 64435)	3
      (0, 66242)	1
      (0, 66511)	2
      (0, 67683)	1
      (0, 68102)	1
      (0, 73151)	1
      (0, 81774)	1
      (0, 87099)	2
      (0, 88519)	1
      (0, 88532)	4
      (0, 88587)	1
      (0, 89360)	1
      (0, 92875)	1
      (0, 93870)	1
      (0, 96391)	1
      (0, 96683)	1
      (0, 96917)	2
      (0, 100208)	2
    


```python
print(X_train_cnt_vect[0:10].shape) ## 여전히 2차원임  
print(X_train_cnt_vect[0].ndim) ## 여전히 2차원임  
## X_train_cnt_vect 은 csr_matrix 구조로, array slice와는 다르다.
```

    (10, 101631)
    2
    


```python
print(type(cnt_vect.inverse_transform(X_train_cnt_vect[0])))
cnt_vect.inverse_transform(X_train_cnt_vect[0])[0]
```

    <class 'list'>
    




    array(['120', 'about', 'accept', 'all', 'any', 'as', 'at', 'be', 'boot',
           'bought', 'btw', 'caddy', 'can', 'cd', 'cd300i', 'chicago',
           'complaints', 'dave', 'did', 'direct', 'drive', 'express', 'from',
           'get', 'good', 'heard', 'how', 'however', 'ideas', 'iivx', 'in',
           'install', 'is', 'it', 'listed', 'mac', 'machine', 'my', 'no',
           'not', 'off', 'one', 'price', 'service', 'system', 'that', 'the',
           'then', 'to', 'used', 've', 'what', 'will', 'with', 'you'],
          dtype='<U81')



---  
**Test numpy array slice 와 비교**


```python
data = np.array([[1, 0, 2],[0, 0, 3],[4, 5, 6]])
```


```python
data.shape
```




    (3, 3)




```python
print(data[0].ndim)
data[0]
```

    1
    




    array([1, 0, 2])



---


```python
print(y_train.shape)
y_train[0:10]
```

    (11314,)
    




    array([ 4, 15, 10,  2,  0,  0,  6, 18, 13, 17])




```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LogisticRegression을 이용하여 학습/예측/평가 수행. 
lr_clf = LogisticRegression(solver='liblinear')  # default:lbfgs
lr_clf.fit(X_train_cnt_vect , y_train)
pred = lr_clf.predict(X_test_cnt_vect)
print('CountVectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test,pred)))
```

    CountVectorized Logistic Regression 의 예측 정확도는 0.617
    

### TF-IDF 피처 변환과 머신러닝 학습/예측/평가


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization 적용하여 학습 데이터셋과 테스트 데이터 셋 변환. 
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

print('학습 & 테스트 데이터 Text의 TfidfVectorizer Shape:',X_train_tfidf_vect.shape, X_test_tfidf_vect.shape)
```

    학습 & 테스트 데이터 Text의 TfidfVectorizer Shape: (11314, 101631) (7532, 101631)
    


```python
X_train_tfidf_vect[0]
```




    <1x101631 sparse matrix of type '<class 'numpy.float64'>'
    	with 55 stored elements in Compressed Sparse Row format>




```python
# LogisticRegression을 이용하여 학습/예측/평가 수행. 
lr_clf = LogisticRegression(solver='liblinear')  # default:lbfgs
lr_clf.fit(X_train_tfidf_vect , y_train)
pred = lr_clf.predict(X_test_tfidf_vect)
print('TF-IDF Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))
```

    TF-IDF Logistic Regression 의 예측 정확도는 0.678
    

### stop words 필터링을 추가하고 ngram을 기본(1,1)에서 (1,2)로 변경하여 피처 벡터화


```python
# stop words 필터링을 추가하고 ngram을 기본(1,1)에서 (1,2)로 변경하여 Feature Vectorization 적용.
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300 )
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test)

lr_clf = LogisticRegression(solver='liblinear')  # default:lbfgs
lr_clf.fit(X_train_tfidf_vect , y_train)
pred = lr_clf.predict(X_test_tfidf_vect)
print('TF-IDF Vectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))
```

    TF-IDF Vectorized Logistic Regression 의 예측 정확도는 0.690
    

### GridSearchCV로 LogisticRegression C 하이퍼 파라미터 튜닝
* C(float, default=1.0) : Inverse of regularization strength(alpha); C=1/alpha, must be a positive float. Like in support vector machines, smaller values specify stronger regularization.


```python
from sklearn.model_selection import GridSearchCV

# 최적 C 값 도출 튜닝 수행. CV는 3 Fold셋으로 설정. 
params = { 'C':[0.01, 0.1, 1, 5, 10]}
grid_cv_lr = GridSearchCV( lr_clf , param_grid= params, cv=3 , scoring='accuracy' , verbose=0 )
grid_cv_lr.fit(X_train_tfidf_vect , y_train)
print('Logistic Regression best C parameter :',grid_cv_lr.best_params_ )
```

    Logistic Regression best C parameter : {'C': 10}
    


```python
# 최적 C 값으로 학습된 grid_cv로 예측 수행하고 정확도 평가. 
pred = grid_cv_lr.predict( X_test_tfidf_vect)
print('TF-IDF Vectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))
```

    TF-IDF Vectorized Logistic Regression 의 예측 정확도는 0.704
    


### 사이킷런 파이프라인(Pipeline) 사용 및 GridSearchCV와의 결합


```python
from sklearn.pipeline import Pipeline

# TfidfVectorizer 객체를 tfidf_vect 객체명으로, LogisticRegression객체를 lr_clf 객체명으로 생성하는 Pipeline생성
pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)),
    ('lr_clf', LogisticRegression(solver='liblinear', C=10))
])

# 별도의 TfidfVectorizer객체의 fit_transform( )과 LogisticRegression의 fit(), predict( )가 필요 없음. 
# pipeline의 fit( ) 과 predict( ) 만으로 한꺼번에 Feature Vectorization과 ML 학습/예측이 가능. 
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print('Pipeline을 통한 Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))
```

    Pipeline을 통한 Logistic Regression 의 예측 정확도는 0.704
    


```python
print(len(pred), pred[:10])
```

    7532 [ 3 11  1  7  8  1 16  6  4 18]
    


```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english')),
    ('lr_clf', LogisticRegression(solver='liblinear'))
])

# Pipeline에 기술된 각각의 객체 변수에 언더바(_)2개를 연달아 붙여 GridSearchCV에 사용될 
# 파라미터/하이퍼 파라미터 이름과 값을 설정. . 
params = { 'tfidf_vect__ngram_range': [(1,1), (1,2), (1,3)],
           'tfidf_vect__max_df': [100, 300, 700],
           'lr_clf__C': [1,5,10]
}

# GridSearchCV의 생성자에 Estimator가 아닌 Pipeline 객체 입력
grid_cv_pipe = GridSearchCV( pipeline,param_grid =params, cv=3 , scoring='accuracy',verbose=1)
grid_cv_pipe.fit(X_train , y_train)
print(grid_cv_pipe.best_score_ , grid_cv_pipe.best_params_)

pred = grid_cv_pipe.predict(X_test)
print('Pipeline을 통한 Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))
```

    Fitting 3 folds for each of 27 candidates, totalling 81 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  81 out of  81 | elapsed: 18.1min finished
    

    0.7550828826229531 {'lr_clf__C': 10, 'tfidf_vect__max_df': 700, 'tfidf_vect__ngram_range': (1, 2)}
    Pipeline을 통한 Logistic Regression 의 예측 정확도는 0.702
    

## CSR-Matrix


```python
from scipy import sparse
import numpy as np
```


```python
row_ind = np.array([0, 1, 2])
col_ind = np.array([1, 2, 1])
```


```python
k = np.ones(3)
k
```




    array([1., 1., 1.])




```python
y_sparse = sparse.csr_matrix((k, (row_ind, col_ind)))
```


```python
print(y_sparse)
```

      (0, 1)	1.0
      (1, 2)	1.0
      (2, 1)	1.0
    


```python
k[0]
```




    1.0




```python
row_ind[0],col_ind[0]
```




    (0, 1)




```python
row_ind[1],col_ind[1]
```




    (1, 2)




```python
row_ind[2],col_ind[2]
```




    (2, 1)


