---
title:  "20 News Group Basic 02"
excerpt: "Text Analysis"  

categories:  
  - Deep-Learning  
tags:  
  - Text Analysis
  - fetch_20newsgroups  
  - csr-matrix
  - CountVectorizer  
  - TfidfVectorizer
last_modified_at: 2020-08-09T15:20:00-05:00
---

## Reference  
* 파이썬 머신러닝 완벽가이드 - 권철민
* NCIA shkim.hi@gmail.com  
* [디리클레 분포](https://datascienceschool.net/view-notebook/e0508d3b7dd6427eba2d35e1f629d3de/)

## 20 Newsgroup 토픽 모델링

* 20개 중 8개의 주제 데이터 로드 및 Count기반 피처 벡터화 
* LDA는 Count기반 Vectorizer만 적용함

## Topic Modeling  
#### 머신러닝 기반의 토픽 모델링은 숨겨진 주제를 효과적으로 표현할 수 있는 중심단어를 함축적으로 추출함  
#### 머신 러닝 기반의 토픽 모델링 알고리즘  
> LSA(Latent Semantic Analysis), pLSA  
> LDA(Latent dirichlet Allocation) - 이번 포스팅 주제임  
> NMF(Non-Negative Matrix Factorization)  

__행렬분해 기반 토픽 모델링 : LSA,NMG__  
__확률 기반 토픽 모델링 : LDA,pLSA__  

어떤 토픽 모델링 알고리즘이든 아래 __2가지의 가정__ 을 기반으로 하고 있다.  
<span style='color:green'>**개별 문서(Document)는 혼합된 여러개의 주제로 구성되어 있음.**</span>  
<span style='color:green'>**개별 주제는 겨러개의 단어로 구성되어 있음.**</span>  


```python
from sklearn.datasets import fetch_20newsgroups

# 모토사이클, 야구, 그래픽스, 윈도우즈, 중동, 기독교, 전자공학, 의학 등 8개 주제를 추출. 
cats = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'comp.windows.x',
        'talk.politics.mideast', 'soc.religion.christian', 'sci.electronics', 'sci.med'  ]
```


```python
# 위에서 cats 변수로 기재된 category만 추출. featch_20newsgroups( )의 categories에 cats 입력
news_df= fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'), categories=cats, random_state=0)
```


```python
print(type(news_df))
print(news_df.keys())
print(type(news_df.data), type(news_df.target))
print(news_df.target.shape)
print(news_df.target[0])
print(news_df.data[0])
```

    <class 'sklearn.utils.Bunch'>
    dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
    <class 'list'> <class 'numpy.ndarray'>
    (7862,)
    6
    I appreciate if anyone can point out some good books about the dead sea
    scrolls of Qumran. Thanks in advance.
    


```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# LDA 는 Count기반의 Vectorizer만 적용합니다.  
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
feat_vect = count_vect.fit_transform(news_df.data)
print('CountVectorizer Shape:', feat_vect.shape)
```

    CountVectorizer Shape: (7862, 1000)
    

max_features=1000 하면서, 자연스럽게 단어 1000 개만 사용하는 것으로 정함  
7862 중 1개만 뜯어서 보면, 해당 어휘사전의 값으로 구성되어 있다. 단, CSR-matrix 형태이니, 변환시 데이터 type을 고려해서 봐야한다.


```python
print(feat_vect[0]) ## csr_matrix 라서, 나온 것임.
```

      (0, 93)	1
      (0, 669)	1
      (0, 390)	1
      (0, 148)	1
      (0, 251)	1
      (0, 876)	1
      (0, 70)	1
      (0, 877)	1
    


```python
count_vect.inverse_transform(feat_vect[0])
```




    [array(['appreciate', 'point', 'good', 'books', 'dead', 'thanks',
            'advance', 'thanks advance'], dtype='<U14')]




```python
[key for key,val in count_vect.vocabulary_.items() if val in feat_vect[0].indices]
```




    ['appreciate',
     'point',
     'good',
     'books',
     'dead',
     'thanks',
     'advance',
     'thanks advance']




```python
print(feat_vect[0].toarray())
```

    [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    

### LDA 객체 생성 후 Count 피처 벡터화 객체로 LDA수행  
#### 디리클레분포(Dirichlet distribution)  


```python
lda = LatentDirichletAllocation(n_components=8, random_state=0) ## 주제가 8개임을 임의로 정한 것이다.  
## feat_vect 를 fit 하면, 초기 lda n_components 수에 따라, 초기 분포가 결정된다.
lda.fit(feat_vect) ## feat_vect : CountVectorizer로 vector와 된 값. BOW 계열 vectorize 되었다.  
```




    LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                              evaluate_every=-1, learning_decay=0.7,
                              learning_method='batch', learning_offset=10.0,
                              max_doc_update_iter=100, max_iter=10,
                              mean_change_tol=0.001, n_components=8, n_jobs=None,
                              perp_tol=0.1, random_state=0, topic_word_prior=None,
                              total_samples=1000000.0, verbose=0)



### 각 토픽 모델링 주제별 단어들의 연관도 확인
* lda객체의 components_ 속성은 주제별로 개별 단어들의 연관도 정규화 숫자가 들어있음
* shape는 주제 개수 X 피처 단어 개수  
* components_ 에 들어 있는 숫자값은 각 주제별로 단어가 나타난 횟수를 정규화 하여 나타냄.   
* 숫자가 클 수록 토픽에서 단어가 차지하는 비중이 높음  


```python
print(lda.components_.shape)
lda.components_
```

    (8, 1000)
    




    array([[3.60992018e+01, 1.35626798e+02, 2.15751867e+01, ...,
            3.02911688e+01, 8.66830093e+01, 6.79285199e+01],
           [1.25199920e-01, 1.44401815e+01, 1.25045596e-01, ...,
            1.81506995e+02, 1.25097844e-01, 9.39593286e+01],
           [3.34762663e+02, 1.25176265e-01, 1.46743299e+02, ...,
            1.25105772e-01, 3.63689741e+01, 1.25025218e-01],
           ...,
           [3.60204965e+01, 2.08640688e+01, 4.29606813e+00, ...,
            1.45056650e+01, 8.33854413e+00, 1.55690009e+01],
           [1.25128711e-01, 1.25247756e-01, 1.25005143e-01, ...,
            9.17278769e+01, 1.25177668e-01, 3.74575887e+01],
           [5.49258690e+01, 4.47009532e+00, 9.88524814e+00, ...,
            4.87048440e+01, 1.25034678e-01, 1.25074632e-01]])



(8 by 1000) 으로 나온다.
8개의 topic 에 대해서, 결과를 보여준다.


```python
# for test by SOO
lda.components_.argsort() ## index sort 열기준으로 앞부분에 있을수록..가장 작은 값이다.
```




    array([[959, 484, 990, ..., 374,   7, 994],
           [296, 433, 690, ..., 517, 485, 291],
           [123, 124, 106, ..., 484, 353, 451],
           ...,
           [ 86,  69, 940, ..., 119, 205, 312],
           [296, 988, 295, ..., 479, 655, 386],
           [478, 496,  69, ..., 876, 295, 921]], dtype=int64)




```python
lda.components_.argsort().shape
```




    (8, 1000)




```python
# for test by JJH
print(lda.components_[0][297], lda.components_[0][485], lda.components_[0][769], lda.components_[0][994])
print(lda.components_[7][297], lda.components_[7][691], lda.components_[7][486], lda.components_[7][518])
```

    110.41185581607033 59.891798905153955 87.4359751747397 703.2389928959205
    19.163149265944565 47.2505161189943 239.93631363605678 16.22916033549396
    

결과값을 보면 상관성있는 값을 return 하는 것으로 보인다. 연관관계? 상관계수같은 값은 아닌것 같다.


```python
# for test by JJH
print(lda.components_[0].argsort()[::][:5], lda.components_[0].argsort()[::-1][:5])
print(lda.components_[7].argsort()[::][:5], lda.components_[7].argsort()[::-1][:5])
```

    [959 484 990 960 361] [994   7 374 563 420]
    [478 496  69 906 433] [921 295 876 966 928]
    


```python
# for test by JJH
' + '.join([str('aaa')+'*'+str(10.33) for i in range(5)]) 
```




    'aaa*10.33 + aaa*10.33 + aaa*10.33 + aaa*10.33 + aaa*10.33'



### 각 토픽별 중심 단어 확인


```python
print(len(count_vect.get_feature_names()))
count_vect.get_feature_names()[210:220]
```

    1000
    




    ['comments',
     'commercial',
     'common',
     'community',
     'comp',
     'company',
     'complete',
     'completely',
     'computer',
     'conference']




```python
## no_top_words : 보고 싶은 상위 단어 갯수를 정하는 param
## feature_names : 학습한 어휘사전의 단어 이름
def display_topic_words(lda_model, feature_names, no_top_words):
    for topic_index, topic in enumerate(lda_model.components_): ## 8 by 1000 : 8개의 주제이니, 1개 주제씩 꺼낸다.
        print('\nTopic #',topic_index)

        # components_ array에서 가장 값이 큰 순으로 정렬했을 때, 그 값의 array index를 반환. 
        topic_word_indexes = topic.argsort()[::-1]
        top_indexes=topic_word_indexes[:no_top_words]
        
        # top_indexes대상인 index별로 feature_names에 해당하는 word feature 추출 후 join으로 concat
        feature_concat = ' + '.join([str(feature_names[i])+'*'+str(round(topic[i],1)) for i in top_indexes])                
        print(feature_concat)
```


```python
# CountVectorizer객체내의 전체 word들의 명칭을 get_features_names( )를 통해 추출
feature_names = count_vect.get_feature_names()
```


```python
# Topic별 가장 연관도가 높은 word를 15개만 추출
display_topic_words(lda, feature_names, 15)

# 모토사이클, 야구, 그래픽스, 윈도우즈, 중동, 기독교, 전자공학, 의학 등 8개 주제를 추출. 
```

    
    Topic # 0
    year*703.2 + 10*563.6 + game*476.3 + medical*413.2 + health*377.4 + team*346.8 + 12*343.9 + 20*340.9 + disease*332.1 + cancer*319.9 + 1993*318.3 + games*317.0 + years*306.5 + patients*299.8 + good*286.3
    
    Topic # 1
    don*1454.3 + just*1392.8 + like*1190.8 + know*1178.1 + people*836.9 + said*802.5 + think*799.7 + time*754.2 + ve*676.3 + didn*675.9 + right*636.3 + going*625.4 + say*620.7 + ll*583.9 + way*570.3
    
    Topic # 2
    image*1047.7 + file*999.1 + jpeg*799.1 + program*495.6 + gif*466.0 + images*443.7 + output*442.3 + format*442.3 + files*438.5 + color*406.3 + entry*387.6 + 00*334.8 + use*308.5 + bit*308.4 + 03*258.7
    
    Topic # 3
    like*620.7 + know*591.7 + don*543.7 + think*528.4 + use*514.3 + does*510.2 + just*509.1 + good*425.8 + time*417.4 + book*410.7 + read*402.9 + information*395.2 + people*393.5 + used*388.2 + post*368.4
    
    Topic # 4
    armenian*960.6 + israel*815.9 + armenians*699.7 + jews*690.9 + turkish*686.1 + people*653.0 + israeli*476.1 + jewish*467.0 + government*464.4 + war*417.8 + dos dos*401.1 + turkey*393.5 + arab*386.1 + armenia*346.3 + 000*345.2
    
    Topic # 5
    edu*1613.5 + com*841.4 + available*761.5 + graphics*708.0 + ftp*668.1 + data*517.9 + pub*508.2 + motif*460.4 + mail*453.3 + widget*447.4 + software*427.6 + mit*421.5 + information*417.3 + version*413.7 + sun*402.4
    
    Topic # 6
    god*2013.0 + people*721.0 + jesus*688.7 + church*663.0 + believe*563.0 + christ*553.1 + does*500.1 + christian*474.8 + say*468.6 + think*446.0 + christians*443.5 + bible*422.9 + faith*420.1 + sin*396.5 + life*371.2
    
    Topic # 7
    use*685.8 + dos*635.0 + thanks*596.0 + windows*548.7 + using*486.5 + window*483.1 + does*456.2 + display*389.1 + help*385.2 + like*382.8 + problem*375.7 + server*370.2 + need*366.3 + know*355.5 + run*315.3
    

### 개별 문서별 토픽 분포 확인
* lda객체의 transform()을 수행하면 개별 문서별 토픽 분포를 반환함


```python
doc_topics = lda.transform(feat_vect)
print(doc_topics.shape)
print(doc_topics[:3])
```

    (7862, 8)
    [[0.01389701 0.01394362 0.01389104 0.48221844 0.01397882 0.01389205
      0.01393501 0.43424401]
     [0.27750436 0.18151826 0.0021208  0.53037189 0.00212129 0.00212102
      0.00212113 0.00212125]
     [0.00544459 0.22166575 0.00544539 0.00544528 0.00544039 0.00544168
      0.00544182 0.74567512]]
    

### 개별 문서별 토픽 분포도를 출력
* 20-newsgroup으로 만들어진 문서명을 출력
* fetch_20newsgroups()으로 만들어진 데이터의 filename속성은 모든 문서의 문서명을 가지고 있음
* filename속성은 절대 디렉토리를 가지는 문서명을 가지고 있으므로 '\\'로 분할하여 맨 마지막 두번째 부터 파일명으로 가져옴


```python
# for test by SOO
print(news_df.filenames)
```

    ['C:\\Users\\정진환\\scikit_learn_data\\20news_home\\20news-bydate-train\\soc.religion.christian\\20630'
     'C:\\Users\\정진환\\scikit_learn_data\\20news_home\\20news-bydate-test\\sci.med\\59422'
     'C:\\Users\\정진환\\scikit_learn_data\\20news_home\\20news-bydate-test\\comp.graphics\\38765'
     ...
     'C:\\Users\\정진환\\scikit_learn_data\\20news_home\\20news-bydate-train\\rec.sport.baseball\\102656'
     'C:\\Users\\정진환\\scikit_learn_data\\20news_home\\20news-bydate-train\\sci.electronics\\53606'
     'C:\\Users\\정진환\\scikit_learn_data\\20news_home\\20news-bydate-train\\talk.politics.mideast\\76505']
    


```python
def get_filename_list(newsdata):
    filename_list=[]

    for file in newsdata.filenames:
            filename_temp = file.split('\\')[-2:]
            filename = '.'.join(filename_temp)
            filename_list.append(filename)
    
    return filename_list

filename_list = get_filename_list(news_df)
print("filename 개수:",len(filename_list), "filename list 10개만:",filename_list[:10])
```

    filename 개수: 7862 filename list 10개만: ['soc.religion.christian.20630', 'sci.med.59422', 'comp.graphics.38765', 'comp.graphics.38810', 'sci.med.59449', 'comp.graphics.38461', 'comp.windows.x.66959', 'rec.motorcycles.104487', 'sci.electronics.53875', 'sci.electronics.53617']
    

### DataFrame으로 생성하여 문서별 토픽 분포도 확인


```python
import pandas as pd 

topic_names = ['Topic #'+ str(i) for i in range(0, 8)]
doc_topic_df = pd.DataFrame(data=doc_topics, columns=topic_names, index=filename_list)
doc_topic_df.head(20)
# 모토사이클, 야구, 그래픽스, 윈도우즈, 중동, 기독교, 전자공학, 의학 등 8개 주제
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
      <th>Topic #0</th>
      <th>Topic #1</th>
      <th>Topic #2</th>
      <th>Topic #3</th>
      <th>Topic #4</th>
      <th>Topic #5</th>
      <th>Topic #6</th>
      <th>Topic #7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>soc.religion.christian.20630</th>
      <td>0.013897</td>
      <td>0.013944</td>
      <td>0.013891</td>
      <td>0.482218</td>
      <td>0.013979</td>
      <td>0.013892</td>
      <td>0.013935</td>
      <td>0.434244</td>
    </tr>
    <tr>
      <th>sci.med.59422</th>
      <td>0.277504</td>
      <td>0.181518</td>
      <td>0.002121</td>
      <td>0.530372</td>
      <td>0.002121</td>
      <td>0.002121</td>
      <td>0.002121</td>
      <td>0.002121</td>
    </tr>
    <tr>
      <th>comp.graphics.38765</th>
      <td>0.005445</td>
      <td>0.221666</td>
      <td>0.005445</td>
      <td>0.005445</td>
      <td>0.005440</td>
      <td>0.005442</td>
      <td>0.005442</td>
      <td>0.745675</td>
    </tr>
    <tr>
      <th>comp.graphics.38810</th>
      <td>0.005439</td>
      <td>0.005441</td>
      <td>0.005449</td>
      <td>0.578959</td>
      <td>0.005440</td>
      <td>0.388387</td>
      <td>0.005442</td>
      <td>0.005442</td>
    </tr>
    <tr>
      <th>sci.med.59449</th>
      <td>0.006584</td>
      <td>0.552000</td>
      <td>0.006587</td>
      <td>0.408485</td>
      <td>0.006585</td>
      <td>0.006585</td>
      <td>0.006588</td>
      <td>0.006585</td>
    </tr>
    <tr>
      <th>comp.graphics.38461</th>
      <td>0.008342</td>
      <td>0.008352</td>
      <td>0.182622</td>
      <td>0.767314</td>
      <td>0.008335</td>
      <td>0.008341</td>
      <td>0.008343</td>
      <td>0.008351</td>
    </tr>
    <tr>
      <th>comp.windows.x.66959</th>
      <td>0.372861</td>
      <td>0.041667</td>
      <td>0.377020</td>
      <td>0.041668</td>
      <td>0.041703</td>
      <td>0.041703</td>
      <td>0.041667</td>
      <td>0.041711</td>
    </tr>
    <tr>
      <th>rec.motorcycles.104487</th>
      <td>0.225351</td>
      <td>0.674669</td>
      <td>0.004814</td>
      <td>0.075920</td>
      <td>0.004812</td>
      <td>0.004812</td>
      <td>0.004812</td>
      <td>0.004810</td>
    </tr>
    <tr>
      <th>sci.electronics.53875</th>
      <td>0.008944</td>
      <td>0.836686</td>
      <td>0.008932</td>
      <td>0.008941</td>
      <td>0.008935</td>
      <td>0.109691</td>
      <td>0.008932</td>
      <td>0.008938</td>
    </tr>
    <tr>
      <th>sci.electronics.53617</th>
      <td>0.041733</td>
      <td>0.041720</td>
      <td>0.708081</td>
      <td>0.041742</td>
      <td>0.041671</td>
      <td>0.041669</td>
      <td>0.041699</td>
      <td>0.041686</td>
    </tr>
    <tr>
      <th>sci.electronics.54089</th>
      <td>0.001647</td>
      <td>0.512634</td>
      <td>0.001647</td>
      <td>0.152375</td>
      <td>0.001645</td>
      <td>0.001649</td>
      <td>0.001647</td>
      <td>0.326757</td>
    </tr>
    <tr>
      <th>rec.sport.baseball.102713</th>
      <td>0.982653</td>
      <td>0.000649</td>
      <td>0.013455</td>
      <td>0.000649</td>
      <td>0.000648</td>
      <td>0.000648</td>
      <td>0.000649</td>
      <td>0.000649</td>
    </tr>
    <tr>
      <th>rec.sport.baseball.104711</th>
      <td>0.288554</td>
      <td>0.007358</td>
      <td>0.007364</td>
      <td>0.596561</td>
      <td>0.078082</td>
      <td>0.007363</td>
      <td>0.007360</td>
      <td>0.007358</td>
    </tr>
    <tr>
      <th>comp.graphics.38232</th>
      <td>0.044939</td>
      <td>0.138461</td>
      <td>0.375098</td>
      <td>0.003914</td>
      <td>0.003909</td>
      <td>0.003911</td>
      <td>0.003912</td>
      <td>0.425856</td>
    </tr>
    <tr>
      <th>sci.electronics.52732</th>
      <td>0.017944</td>
      <td>0.874782</td>
      <td>0.017869</td>
      <td>0.017904</td>
      <td>0.017867</td>
      <td>0.017866</td>
      <td>0.017884</td>
      <td>0.017885</td>
    </tr>
    <tr>
      <th>talk.politics.mideast.76440</th>
      <td>0.003381</td>
      <td>0.003385</td>
      <td>0.003381</td>
      <td>0.843991</td>
      <td>0.135716</td>
      <td>0.003380</td>
      <td>0.003384</td>
      <td>0.003382</td>
    </tr>
    <tr>
      <th>sci.med.59243</th>
      <td>0.491684</td>
      <td>0.486865</td>
      <td>0.003574</td>
      <td>0.003577</td>
      <td>0.003578</td>
      <td>0.003574</td>
      <td>0.003574</td>
      <td>0.003574</td>
    </tr>
    <tr>
      <th>talk.politics.mideast.75888</th>
      <td>0.015639</td>
      <td>0.499140</td>
      <td>0.015641</td>
      <td>0.015683</td>
      <td>0.015640</td>
      <td>0.406977</td>
      <td>0.015644</td>
      <td>0.015636</td>
    </tr>
    <tr>
      <th>soc.religion.christian.21526</th>
      <td>0.002455</td>
      <td>0.164735</td>
      <td>0.002455</td>
      <td>0.002456</td>
      <td>0.208655</td>
      <td>0.002454</td>
      <td>0.614333</td>
      <td>0.002458</td>
    </tr>
    <tr>
      <th>comp.windows.x.66408</th>
      <td>0.000080</td>
      <td>0.000080</td>
      <td>0.809449</td>
      <td>0.163054</td>
      <td>0.000080</td>
      <td>0.027097</td>
      <td>0.000080</td>
      <td>0.000080</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
