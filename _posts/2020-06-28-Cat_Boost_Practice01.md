---
title:  "CatBoost Practice 01"  
excerpt: "CatBoost, Ensemble, Gradient Descent"  

categories:  
  - Machine-Learning  
tags:  
  - Stacking  
  - Ensemble  
  - Medium  
  - Kaggle
last_modified_at: 2020-06-28T15:00:00-05:00
---

## Reference  
- [Medium Daniel Chepenko](https://towardsdatascience.com/introduction-to-gradient-boosting-on-decision-trees-with-catboost-d511a9ccbd14)
- 상기 Blog 역자의 Collab 코드
>  Kaggle competition 에 있는 data를 활용한다. 
>> 보험회사에 공개한, 심각한 회사차원의 보험손실 여부를 추정한 regression 모델


```python
import pandas as pd

df_train = pd.read_csv('../dataset/allstate-claims-severity/train.csv')
df_test = pd.read_csv('../dataset/allstate-claims-severity/test.csv')
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots
```


```python
print(df_train.shape) ## loss 값 있고,
print(df_test.shape)  ## loss 값 없음.
```

    (188318, 132)
    (125546, 131)
    


```python
df_train.head()
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.718367</td>
      <td>0.335060</td>
      <td>0.30260</td>
      <td>0.67135</td>
      <td>0.83510</td>
      <td>0.569745</td>
      <td>0.594646</td>
      <td>0.822493</td>
      <td>0.714843</td>
      <td>2213.18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.438917</td>
      <td>0.436585</td>
      <td>0.60087</td>
      <td>0.35127</td>
      <td>0.43919</td>
      <td>0.338312</td>
      <td>0.366307</td>
      <td>0.611431</td>
      <td>0.304496</td>
      <td>1283.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.289648</td>
      <td>0.315545</td>
      <td>0.27320</td>
      <td>0.26076</td>
      <td>0.32446</td>
      <td>0.381398</td>
      <td>0.373424</td>
      <td>0.195709</td>
      <td>0.774425</td>
      <td>3005.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.440945</td>
      <td>0.391128</td>
      <td>0.31796</td>
      <td>0.32128</td>
      <td>0.44467</td>
      <td>0.327915</td>
      <td>0.321570</td>
      <td>0.605077</td>
      <td>0.602642</td>
      <td>939.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.178193</td>
      <td>0.247408</td>
      <td>0.24564</td>
      <td>0.22089</td>
      <td>0.21230</td>
      <td>0.204687</td>
      <td>0.202213</td>
      <td>0.246011</td>
      <td>0.432606</td>
      <td>2763.85</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 132 columns</p>
</div>




```python
df_test.head(2)
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.281143</td>
      <td>0.466591</td>
      <td>0.317681</td>
      <td>0.61229</td>
      <td>0.34365</td>
      <td>0.38016</td>
      <td>0.377724</td>
      <td>0.369858</td>
      <td>0.704052</td>
      <td>0.392562</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.836443</td>
      <td>0.482425</td>
      <td>0.443760</td>
      <td>0.71330</td>
      <td>0.51890</td>
      <td>0.60401</td>
      <td>0.689039</td>
      <td>0.675759</td>
      <td>0.453468</td>
      <td>0.208045</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 131 columns</p>
</div>




```python
print("Number of missing values in train",df_train.isnull().sum().sum())
print("Number of missing values in test",df_test.isnull().sum().sum())
```

    Number of missing values in train 0
    Number of missing values in test 0
    

* 한꺼번에 전처리 하기 위해 묶은다음에 수행한다.


```python
df_train_idx = df_train.index
df_test_idx = df_test.index
df_train['isTrain'] = True
df_test['isTrain'] = False
traintest = pd.concat([df_train, df_test], axis = 0)
```


```python
set(df_train.columns)-set(df_test.columns)
```
    {'loss'}




```python
print(traintest.shape) ## 왜 133 이 되었지? -> isTrain 컬럼이 추가됨
```

    (313864, 133)
    

* 컬럼이름 정리


```python
import re
cat_pattern = re.compile("^cat([1-9]|[1-9][0-9]|[1-9][0-9][0-9])$")
cont_pattern = re.compile("^cont([1-9]|[1-9][0-9]|[1-9][0-9][0-9])$")

## cat이 들어간 컬럼. catogory 컬럼을 고르고, 뒤 3자리 숫자에 대해서, sorting
cat_col = sorted([cat for cat in traintest.columns if 'cat' in cat], key = lambda s: int(s[3:]))
## 정규식 패턴에 매칭되는 (위에서 구한 cat_col)들의 index 나열
cat_index = [i for i in range(0,len(traintest.columns)-1) if cat_pattern.match(traintest.columns[i])]

cont_col = sorted([cont for cont in traintest.columns if 'cont' in cont], key = lambda s: int(s[4:]))
cont_index = [i for i in range(0,len(traintest.columns)-1) if cont_pattern.match(traintest.columns[i])]
features = cat_col + cont_col
```


```python
print(len(features)) ## 3개가 줄음 : isTrain,loss,id 컬럼
feats_counts = traintest[cat_col].nunique(dropna = False)
```

    130
    

## 1-1. EDA columns check

[datafram.nunique](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.nunique.html)


```python
len(cat_col),len(cont_col)
```
    (116, 14)


```python
feats_counts.sort_values() ## 대략 categorical 컬럼의 cardinality 를 확인하기 위해 자주 사용한다.
```

    cat1        2
    cat52       2
    cat51       2
    cat50       2
    cat49       2
             ... 
    cat112     51
    cat113     63
    cat109     85
    cat110    134
    cat116    349
    Length: 116, dtype: int64




```python
## .nunique : return pandas.core.series.Series / 각컬럼값들의 unique한 갯수를 확인할 수 있음.
nunique = df_train[cat_col].nunique(dropna = False)
```


```python
print(len(nunique))
nunique[100:110]
```

    116
    
    cat101     19
    cat102      9
    cat103     13
    cat104     17
    cat105     20
    cat106     17
    cat107     20
    cat108     11
    cat109     84
    cat110    131
    dtype: int64




```python
plt.figure(figsize=(14,6))
_ = plt.hist(nunique, bins=100)
```


![png](/assets/images/cat_boost_practice/output_21_0.png)


대부분의 cardinality가 10 개 이하의, 2개 짜리의 컬럼이 매우 많고. cardinality 300 이 넘는 것도 있다.


```python
mask = (nunique > 100) ## cardinality 100 개 넘는 column 들만, 따로 dataframe 으로 빼서 확인하기
df_train[cat_col].loc[:, mask] 
## cat110,cat116 컬럼의 cardinality가 100 이 넘음
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
      <th>cat110</th>
      <th>cat116</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BC</td>
      <td>LB</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CQ</td>
      <td>DP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DK</td>
      <td>GK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS</td>
      <td>DJ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C</td>
      <td>CK</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>188313</th>
      <td>EG</td>
      <td>CK</td>
    </tr>
    <tr>
      <th>188314</th>
      <td>BT</td>
      <td>DF</td>
    </tr>
    <tr>
      <th>188315</th>
      <td>DM</td>
      <td>DJ</td>
    </tr>
    <tr>
      <th>188316</th>
      <td>AI</td>
      <td>MD</td>
    </tr>
    <tr>
      <th>188317</th>
      <td>EG</td>
      <td>MJ</td>
    </tr>
  </tbody>
</table>
<p>188318 rows × 2 columns</p>
</div>

```python
df_train.cat110.value_counts()
```

    CL    25305
    EG    24654
    CS    24592
    EB    21396
    CO    17495
          ...  
    BI        1
    BM        1
    S         1
    DV        1
    CB        1
    Name: cat110, Length: 131, dtype: int64



상기 같이 cardinality 높은 컬럼들은 뉴메릭으로 변환하거나, zero variance groups 으로 이니 제거하는게 나을 수 있다.


```python
df_train.id.is_unique
```
    True

```python
## cat110 컬럼에 대해 각 분류값별로, 얼마나 row 가 있는지 확인함 = df_train.cat110.value_counts()
## df_train.groupby('cat110')['id'].size().sort_values() 과 동일
cat110_nunique = df_train.groupby('cat110')['id'].nunique().sort_values(ascending=False) 
```
```python
cat110_nunique
```
    cat110
    CL    25305
    EG    24654
    CS    24592
    EB    21396
    CO    17495
          ...  
    DV        1
    BD        1
    BI        1
    EH        1
    BK        1
    Name: id, Length: 131, dtype: int64




```python
plt.figure(figsize=(14,6))
_ = plt.hist(cat110_nunique, bins=50)
```

![png](/assets/images/cat_boost_practice/output_29_0.png)

대부분 1~2개씩 있는 것으로 보인다.

```python
cat116_nunique = df_train.groupby('cat116')['id'].nunique().sort_values()
```


```python
plt.figure(figsize=(14,6))
_ = plt.hist(cat116_nunique, bins=50)
```


![png](/assets/images/cat_boost_practice/output_32_0.png)


The values are not float, they are integer, so these features are likely to be even counts. Let's look at another pack of features.  
**값은 부동하지 않고 정수이므로 이러한 특성은 짝수 카운트일 가능성이 높다. 다른 특징들을 살펴봅시다.**  


```python
from tqdm.notebook import tqdm
```


```python
df_train[cat_col].head()
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>...</th>
      <th>cat107</th>
      <th>cat108</th>
      <th>cat109</th>
      <th>cat110</th>
      <th>cat111</th>
      <th>cat112</th>
      <th>cat113</th>
      <th>cat114</th>
      <th>cat115</th>
      <th>cat116</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>...</td>
      <td>J</td>
      <td>G</td>
      <td>BU</td>
      <td>BC</td>
      <td>C</td>
      <td>AS</td>
      <td>S</td>
      <td>A</td>
      <td>O</td>
      <td>LB</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>...</td>
      <td>K</td>
      <td>K</td>
      <td>BI</td>
      <td>CQ</td>
      <td>A</td>
      <td>AV</td>
      <td>BM</td>
      <td>A</td>
      <td>O</td>
      <td>DP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>...</td>
      <td>F</td>
      <td>A</td>
      <td>AB</td>
      <td>DK</td>
      <td>A</td>
      <td>C</td>
      <td>AF</td>
      <td>A</td>
      <td>I</td>
      <td>GK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>...</td>
      <td>K</td>
      <td>K</td>
      <td>BI</td>
      <td>CS</td>
      <td>C</td>
      <td>N</td>
      <td>AE</td>
      <td>A</td>
      <td>O</td>
      <td>DJ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>...</td>
      <td>G</td>
      <td>B</td>
      <td>H</td>
      <td>C</td>
      <td>C</td>
      <td>Y</td>
      <td>BM</td>
      <td>A</td>
      <td>K</td>
      <td>CK</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 116 columns</p>
</div>

```python
df_train.cat1.unique().tolist()
```
    ['A', 'B']

```python
df_train['cat112'].loc[df_train['cat112'].isin(df_train.cat1.unique().tolist())]
```
    17        A
    34        A
    77        A
    107       A
    292       B
             ..
    187955    A
    188037    A
    188266    A
    188281    A
    188303    B
    Name: cat112, Length: 2834, dtype: object

```python
## 예시
print(len(set(traintest['cat116'].factorize()[0])))
```
    349

```python
#Encoding categorical features to find duplicates
train_enc =  pd.DataFrame(index = traintest.index)

## categorical = cat_col 에 대해서, 모두 숫자화로 변형시킨다. traintest = train+test concat
for col in tqdm(traintest[cat_col].columns): 
    train_enc[col] = traintest[col].factorize()[0]
```
    HBox(children=(FloatProgress(value=0.0, max=116.0), HTML(value='')))


```python
train_enc.head(3)
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>...</th>
      <th>cat107</th>
      <th>cat108</th>
      <th>cat109</th>
      <th>cat110</th>
      <th>cat111</th>
      <th>cat112</th>
      <th>cat113</th>
      <th>cat114</th>
      <th>cat115</th>
      <th>cat116</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 116 columns</p>
</div>




```python
dup_cols = {} ## 중복되는 컬럼을 제거하는 과정
import numpy as np

for i, c1 in enumerate(tqdm(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]: ## i번째 컬럼이후 다른 모든 컬럼
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]): ## 바로 next 컬럼이, 지금 컬럼과 통으로 일치하면~
            dup_cols[c2] = c1
```
    HBox(children=(FloatProgress(value=0.0, max=116.0), HTML(value='')))

```python
## 다행히 중복되는 컬럼은 없다.
dup_cols
```
    {}



여기서, cat_col 에 대해 보면,, 모든 값이 알파벳의 조합으로만 되어있다. 따라서,  
Factorize 하면, 결국 같은 value 들은 같은 숫자를 가지게 된다.  
<span style='color:red'>원작자는 이를 통해, 각 cat_columns 값이 같은면 1, 아니면 0으로 하여 평균값을 구하였고  
이를 통해 컬럼들의 유사성을 파악했다.</span>


```python
test_list = ['a','b','c','d','e']
```


```python
alst = []
for i,c1 in enumerate(test_list):
    blst = [] 
    for j,c2 in enumerate(test_list):
        if i>=j: ##
            blst.append(c1+c2)            
#             blst.append((train.loc[mask,c1].values>=train.loc[mask,c2].values).mean())
        else: ## i < j 인 경우
            blst.append(c2+c1)
#             blst.append((train.loc[mask,c1].values>train.loc[mask,c2].values).mean())
    alst.append(blst)
```


```python
alst
```




    [['aa', 'ba', 'ca', 'da', 'ea'],
     ['ba', 'bb', 'cb', 'db', 'eb'],
     ['ca', 'cb', 'cc', 'dc', 'ec'],
     ['da', 'db', 'dc', 'dd', 'ed'],
     ['ea', 'eb', 'ec', 'ed', 'ee']]



## 1-2. EDA 컬럼별 연관성 분석

cat_col들의 연관성 보는 함수 (위의 설명 참조)


```python
def autolabel(arrayA):
    ''' label each colored square with the corresponding data value. 
    If value > 20, the text is in black, else in white.
    '''
    arrayA = np.array(arrayA)
    for i in range(arrayA.shape[0]):
        for j in range(arrayA.shape[1]):
                plt.text(j,i, "%.2f"%arrayA[i,j], ha='center', va='bottom',color='w')


def gt_matrix(train, feats,sz=18): ## feats 컬럼명
    a = []
    for i,c1 in enumerate(feats):
        b = [] 
        for j,c2 in enumerate(feats):
            mask = (~train[c1].isnull()) & (~train[c2].isnull()) ## c1,c2 컬럼 둘다 null 아닌 row 만 표시
            if i>=j: ##
                b.append((train.loc[mask,c1].values>=train.loc[mask,c2].values).mean())
            else: ## i < j 인 경우
                b.append((train.loc[mask,c1].values>train.loc[mask,c2].values).mean())
        a.append(b)
    plt.figure(figsize = (sz,sz))
    plt.imshow(a, interpolation = 'None')
    _ = plt.xticks(range(len(feats)),feats,rotation = 90)
    _ = plt.yticks(range(len(feats)),feats,rotation = 0)
    autolabel(a)
```


```python
mask = (~train_enc['cat3'].isnull()) & (~train_enc['cat1'].isnull()) 
train_enc.loc[mask,'cat3'].values 
```
    array([0, 0, 0, ..., 0, 0, 0], dtype=int64)

```python
## 대각 아래 영역 : 같거나, 큰거...
print(train_enc.loc[mask,'cat2'].values >= train_enc.loc[mask,'cat1'].values)
print((train_enc.loc[mask,'cat2'].values >= train_enc.loc[mask,'cat1'].values).mean())
```

    [ True  True  True ... False  True  True]
    0.9280388958274922
    

```python
## 대각 상단 영역
print(train_enc.loc[mask,'cat1'].values > train_enc.loc[mask,'cat3'].values)
print((train_enc.loc[mask,'cat1'].values > train_enc.loc[mask,'cat3'].values).mean())
```

    [False False False ...  True False False]
    0.24539609512400276
    


```python
#from cat13 to cat30
gt_matrix(train_enc, cat_col[:30])
```

![png](/assets/images/cat_boost_practice/output_53_0.png)

```python
#everything except cat36, cat 37, cat50
gt_matrix(train_enc, cat_col[29:60])
```


![png](/assets/images/cat_boost_practice/output_54_0.png)

```python
#from cat60 to cat71, cat74, from cat76 to cat78
gt_matrix(train_enc, cat_col[59:90])
```
![png](/assets/images/cat_boost_practice/output_55_0.png)


```python
cum_cat_1 = []
for i in range(12,23):
    cum_cat_1.append('cat'+str(i))

cum_cat_2 = []
for i in range(24,50):
    cum_cat_2.append('cat'+str(i))

cum_cat_3 = []
for i in range(51,72):
    cum_cat_3.append('cat'+str(i)) 
```

From the graph above we can see that some of the features are cummulative, as in some cases the one is strictly greater than the other.
 Finally lets look at correlation between continious features. Probably dimension reduction techniques can help us

## 1-3. EDA -Numeric 변수들 상관계수 확인하기


```python
threshold = 0.5
data_corr = traintest[cont_col].corr()
# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,len(cont_col)): #for 'size' features
    for j in range(i+1,len(cont_col)): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: abs(x[0]),reverse=True)

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cont_col[i],cont_col[j],v))
```
    cont11 and cont12 = 0.99
    cont1 and cont9 = 0.93
    cont6 and cont10 = 0.88
    cont6 and cont13 = 0.81
    cont1 and cont10 = 0.81
    cont6 and cont9 = 0.80
    cont6 and cont12 = 0.79
    cont9 and cont10 = 0.79
    cont6 and cont11 = 0.77
    cont1 and cont6 = 0.76
    cont7 and cont11 = 0.75
    cont7 and cont12 = 0.74
    cont10 and cont12 = 0.72
    cont10 and cont13 = 0.71
    cont10 and cont11 = 0.70
    cont6 and cont7 = 0.66
    cont9 and cont13 = 0.64
    cont9 and cont12 = 0.63
    cont1 and cont12 = 0.61
    cont9 and cont11 = 0.61
    cont1 and cont11 = 0.60
    cont1 and cont13 = 0.53
    cont4 and cont8 = 0.53
    

## 1-4. EDA - PCA dimension reduction


```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```


```python
traintest.shape[0] == len(df_train_idx.to_list())+len(df_test_idx.to_list())
```
    True

If 0 < n_components < 1 and svd_solver == 'full'  
> select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.  
> diag 고유값행렬 (eigen value (lambda)) 의  0.95(95%)에 해당하는 변수들을 모두 본다는 의미임  
> 만약 0.95 -> 0.7 -> 0.5 로 될 수록, 백분위수 컨셉으로 총량의 variance 가 작아지니, len(pca.explained_variance_) 이 작아진다.


```python
## numeric 변수들로만 시행
scaler = StandardScaler()
traintest_scaled = scaler.fit_transform(traintest[cont_col]) ## modify scale in numeric columns
```


```python
print(traintest_scaled.shape)
```
    (313864, 14)
    


```python
pca = PCA(n_components=0.95, svd_solver='full').fit(traintest_scaled)
traintest_pca = pca.transform(traintest_scaled)
```


```python
print(len(pca.explained_variance_)),print(pca.explained_variance_)
```
    9
    [6.15481809 2.00696816 1.49785047 0.99611156 0.88174836 0.75109768
     0.56965726 0.39493657 0.30094088]

    (None, None)




```python
traintest_pca_df = pd.DataFrame(data = traintest_pca)
traintest_pca_df['id'] = traintest['id'].values
```


```python
print(traintest[cont_col].shape)
```
    (313864, 14)
    


```python
np.dot(traintest_scaled,pca.components_.T)[0]
```
    array([ 2.70453183, -2.10166972, -1.51128802,  0.61746921, -0.54985965,
           -0.21919071,  0.94667346,  1.10674658, -1.36947719])

```python
print(traintest_pca_df.shape)
traintest_pca_df.head()
```
    (313864, 10)
    

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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.704532</td>
      <td>-2.101670</td>
      <td>-1.511288</td>
      <td>0.617469</td>
      <td>-0.549860</td>
      <td>-0.219191</td>
      <td>0.946673</td>
      <td>1.106747</td>
      <td>-1.369477</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.132381</td>
      <td>-0.395118</td>
      <td>2.283496</td>
      <td>-0.762693</td>
      <td>-0.352918</td>
      <td>-0.593252</td>
      <td>-1.064844</td>
      <td>0.443155</td>
      <td>0.220732</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.948887</td>
      <td>0.339230</td>
      <td>-1.375522</td>
      <td>1.256290</td>
      <td>-0.212120</td>
      <td>0.107400</td>
      <td>0.040797</td>
      <td>0.128152</td>
      <td>-0.309722</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.597690</td>
      <td>0.072182</td>
      <td>-0.424028</td>
      <td>0.392106</td>
      <td>-0.004357</td>
      <td>-1.252742</td>
      <td>-0.050669</td>
      <td>0.638594</td>
      <td>-0.293227</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-4.068934</td>
      <td>-1.052458</td>
      <td>-0.660599</td>
      <td>-0.245261</td>
      <td>-0.841898</td>
      <td>0.906329</td>
      <td>-0.748438</td>
      <td>0.017867</td>
      <td>-0.517443</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>

```python
traintest[cont_col].describe()
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
      <th>cont1</th>
      <th>cont2</th>
      <th>cont3</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.00000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
      <td>313864.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.494096</td>
      <td>0.507089</td>
      <td>0.498653</td>
      <td>0.492021</td>
      <td>0.487513</td>
      <td>0.491442</td>
      <td>0.485360</td>
      <td>0.486823</td>
      <td>0.48571</td>
      <td>0.498403</td>
      <td>0.493850</td>
      <td>0.493503</td>
      <td>0.493917</td>
      <td>0.495665</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.187768</td>
      <td>0.207056</td>
      <td>0.201961</td>
      <td>0.211101</td>
      <td>0.209063</td>
      <td>0.205394</td>
      <td>0.178531</td>
      <td>0.199442</td>
      <td>0.18185</td>
      <td>0.185906</td>
      <td>0.210002</td>
      <td>0.209716</td>
      <td>0.212911</td>
      <td>0.222537</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000016</td>
      <td>0.001149</td>
      <td>0.002634</td>
      <td>0.176921</td>
      <td>0.281143</td>
      <td>0.012683</td>
      <td>0.069503</td>
      <td>0.236880</td>
      <td>0.00008</td>
      <td>0.000000</td>
      <td>0.035321</td>
      <td>0.036232</td>
      <td>0.000228</td>
      <td>0.178568</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.347403</td>
      <td>0.358319</td>
      <td>0.336963</td>
      <td>0.327354</td>
      <td>0.281143</td>
      <td>0.336105</td>
      <td>0.351299</td>
      <td>0.317960</td>
      <td>0.35897</td>
      <td>0.364580</td>
      <td>0.310961</td>
      <td>0.314945</td>
      <td>0.315758</td>
      <td>0.294657</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.475784</td>
      <td>0.555782</td>
      <td>0.527991</td>
      <td>0.452887</td>
      <td>0.422268</td>
      <td>0.440945</td>
      <td>0.438650</td>
      <td>0.441060</td>
      <td>0.44145</td>
      <td>0.461190</td>
      <td>0.457203</td>
      <td>0.462286</td>
      <td>0.363547</td>
      <td>0.407020</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.625272</td>
      <td>0.681761</td>
      <td>0.634224</td>
      <td>0.652072</td>
      <td>0.643315</td>
      <td>0.655818</td>
      <td>0.591165</td>
      <td>0.623580</td>
      <td>0.56889</td>
      <td>0.619840</td>
      <td>0.678924</td>
      <td>0.679096</td>
      <td>0.689974</td>
      <td>0.724707</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.984975</td>
      <td>0.862654</td>
      <td>0.944251</td>
      <td>0.956046</td>
      <td>0.983674</td>
      <td>0.997162</td>
      <td>1.000000</td>
      <td>0.982800</td>
      <td>0.99540</td>
      <td>0.994980</td>
      <td>0.998742</td>
      <td>0.998484</td>
      <td>0.988494</td>
      <td>0.844848</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(18,9))
ax = sns.boxplot(data=traintest[cont_col], orient="v", palette="Set2")
```


![png](/assets/images/cat_boost_practice/output_73_0.png)


## 1-5. EDA - 데이터 왜도(skewness) 확인

For **normally distributed data, the skewness should be about zero.** For unimodal continuous distributions, a skewness value greater than zero means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to zero, statistically speaking.


```python
from scipy.stats import skew ## Compute the sample skewness of a data set.

skewed_cols = []
for col in cont_col:
    if skew(traintest[col]) > 0.75:
        skewed_cols.append(col)
```


```python
## 왜도가 0.75 넘는 컬럼 찾아내기
skewed_cols
```
    ['cont7', 'cont9']


```python
for col in skewed_cols:
    plt.figure()
    plt.title(col)
    plt.hist(traintest[col], bins = 50)
```
![png](/assets/images/cat_boost_practice/output_78_0.png)



![png](/assets/images/cat_boost_practice/output_78_1.png)


As we see the features cont7 and cont 9 are left skewed. So we need to apply transformation. BoxCox could be a good way to deal with this problem


```python
from scipy import stats
stats.boxcox(traintest[col])[0] ## unnormal distribution data -> normal distribution data  (일종의 Skill)
```
    array([-0.37071516, -0.86886216, -1.06116485, ..., -0.05663025,
           -0.92868009, -0.94375053])




```python
for col in skewed_cols:
    col_name = col+"_"
    traintest[col_name] = stats.boxcox(traintest[col])[0]
    plt.figure()
    plt.title(col_name)
    plt.hist(traintest[col_name], bins = 50)
```

![png](/assets/images/cat_boost_practice/output_81_0.png)



![png](/assets/images/cat_boost_practice/output_81_1.png)


## 1-6. EDA - Y 컬럼(=loss) 컬럼의 밀도를 보고, left skewness 인것에 대해 log 변환한다.


```python
## log 를 씌워서, loss 컬럼의 밀도 살펴보기
plt.figure(figsize=(13,9))
ax = sns.distplot(df_train["loss"])
ax.set_title('NO-LOG-LOSS column')
```
    Text(0.5, 1.0, 'NO-LOG-LOSS column')

![png](/assets/images/cat_boost_practice/output_83_1.png)



```python
## log 를 씌워서, loss 컬럼의 밀도 살펴보기
plt.figure(figsize=(13,9))
sns.distplot(np.log(df_train["loss"])).set_title('LOG-LOSS column')
```
    Text(0.5, 1.0, 'LOG-LOSS column')

![png](/assets/images/cat_boost_practice/output_84_1.png)


## 2-1. Modeling Without tunning, apply Catboost - Base
- y 에 대해서는 np.log(y) 로 변형시켜서 사용했다.  
- Firslty try to run model without any EDA transformation


```python
from catboost import CatBoostRegressor
```


```python
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
      <th>isTrain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.335060</td>
      <td>0.30260</td>
      <td>0.67135</td>
      <td>0.83510</td>
      <td>0.569745</td>
      <td>0.594646</td>
      <td>0.822493</td>
      <td>0.714843</td>
      <td>2213.18</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.436585</td>
      <td>0.60087</td>
      <td>0.35127</td>
      <td>0.43919</td>
      <td>0.338312</td>
      <td>0.366307</td>
      <td>0.611431</td>
      <td>0.304496</td>
      <td>1283.60</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 133 columns</p>
</div>




```python
df_train.isTrain.value_counts()
```




    True    188318
    Name: isTrain, dtype: int64




```python
## features : loss,isTrain 컬럼을 제외한 순수 Feature 컬럼
X = df_train.drop('id', axis = 1)[features]
y = np.log(df_train['loss'])

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)
```


```python
## cat이 들어간 컬럼. catogory 컬럼을 고르고, 뒤 3자리 숫자에 대해서, sorting
cat_col = sorted([cat for cat in X_train.columns if 'cat' in cat], key = lambda s: int(s[3:]))
## 정규식 패턴에 매칭되는 (위에서 구한 cat_col)들의 index 나열
cat_index = [i for i in range(0,len(X_train.columns)-1) if cat_pattern.match(X_train.columns[i])]

cont_col = sorted([cont for cont in X_train.columns if 'cont' in cont], key = lambda s: int(s[4:]))
cont_index = [i for i in range(0,len(X_train.columns)-1) if cont_pattern.match(X_train.columns[i])]
features = cat_col + cont_col
```

```python
params = {'iterations':100, 'learning_rate':0.1, 'eval_metric':"MAE"}
```

```python
model_0 = CatBoostRegressor(**params)
```

```python
model_0.fit(X_train, y_train, cat_index, eval_set=(X_validation, y_validation) ,plot=True,verbose=False,use_best_model=True)
```
![png](/assets/images/cat_boost_practice/cat_boo_plot_01.png)

```python
print(model_0.evals_result_.keys())
print(model_0.learning_rate_,model_0.tree_count_)
mae_min_idx = np.argmin(model_0.evals_result_['validation']['MAE'])
print("Minimize MAE value is {:.6f} in Validation Set".format(model_0.evals_result_['validation']['MAE'][mae_min_idx]))
```

    dict_keys(['learn', 'validation'])
    0.10000000149011612 100
    Minimize MAE value is 0.427776 in Validation Set

```python
## y 결과값에 log 변환한 값이 나올테니, 이를 다시 원본값으로 돌리기 위해 np.exp()를 씌웠다.
df_test['loss'] = np.exp(model_0.predict(df_test[features]))
```

```python
df_test.head()
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>isTrain</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.317681</td>
      <td>0.61229</td>
      <td>0.34365</td>
      <td>0.38016</td>
      <td>0.377724</td>
      <td>0.369858</td>
      <td>0.704052</td>
      <td>0.392562</td>
      <td>False</td>
      <td>1597.451430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.443760</td>
      <td>0.71330</td>
      <td>0.51890</td>
      <td>0.60401</td>
      <td>0.689039</td>
      <td>0.675759</td>
      <td>0.453468</td>
      <td>0.208045</td>
      <td>False</td>
      <td>1960.685945</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.325779</td>
      <td>0.29758</td>
      <td>0.34365</td>
      <td>0.30529</td>
      <td>0.245410</td>
      <td>0.241676</td>
      <td>0.258586</td>
      <td>0.297232</td>
      <td>False</td>
      <td>8500.643058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>...</td>
      <td>0.342355</td>
      <td>0.40028</td>
      <td>0.33237</td>
      <td>0.31480</td>
      <td>0.348867</td>
      <td>0.341872</td>
      <td>0.592264</td>
      <td>0.555955</td>
      <td>False</td>
      <td>5131.988609</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>...</td>
      <td>0.391833</td>
      <td>0.23688</td>
      <td>0.43731</td>
      <td>0.50556</td>
      <td>0.359572</td>
      <td>0.352251</td>
      <td>0.301535</td>
      <td>0.825823</td>
      <td>False</td>
      <td>915.043904</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 133 columns</p>
</div>




```python
plt.figure(figsize=(13,9))
sns.distplot(df_test["loss"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x151826c5448>

![png](/assets/images/cat_boost_practice/output_97_1.png)



```python
df_test.to_csv('../dataset/kaggle_submission/acs_submission_0.csv', sep = ',', columns = ['id', 'loss'], index=False)
```

## 2-2. Modiling With PCA, apply Catboost
 - cont_col 만 PCA decomposition 시 사용했음. 주의할 것!  
 - Try to reduce dimenstions with PCA


```python
traintest_pca_df.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.704532</td>
      <td>-2.101670</td>
      <td>-1.511288</td>
      <td>0.617469</td>
      <td>-0.549860</td>
      <td>-0.219191</td>
      <td>0.946673</td>
      <td>1.106747</td>
      <td>-1.369477</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.132381</td>
      <td>-0.395118</td>
      <td>2.283496</td>
      <td>-0.762693</td>
      <td>-0.352918</td>
      <td>-0.593252</td>
      <td>-1.064844</td>
      <td>0.443155</td>
      <td>0.220732</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.948887</td>
      <td>0.339230</td>
      <td>-1.375522</td>
      <td>1.256290</td>
      <td>-0.212120</td>
      <td>0.107400</td>
      <td>0.040797</td>
      <td>0.128152</td>
      <td>-0.309722</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.597690</td>
      <td>0.072182</td>
      <td>-0.424028</td>
      <td>0.392106</td>
      <td>-0.004357</td>
      <td>-1.252742</td>
      <td>-0.050669</td>
      <td>0.638594</td>
      <td>-0.293227</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-4.068934</td>
      <td>-1.052458</td>
      <td>-0.660599</td>
      <td>-0.245261</td>
      <td>-0.841898</td>
      <td>0.906329</td>
      <td>-0.748438</td>
      <td>0.017867</td>
      <td>-0.517443</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("traintest_pca_df.shape :{}".format(traintest_pca_df.shape))
print("traintest.shape :{}".format(traintest.shape))
```

    traintest_pca_df.shape :(313864, 10)
    traintest.shape :(313864, 135)
    

df_train_idx,df_test_idx 각각에 해당하는 데이터를 골라냄 (index 를 이용해서)  
traintest_pca_df 는 이미 PCA 를 이용해서,원래의 row,columns 등 shape가 축소된 형태임


```python
type(df_train_idx)
```
    pandas.core.indexes.range.RangeIndex




```python
traintest_pca_df.loc[df_train_idx,:].shape
```
    (188318, 10)




```python
# df_train_pca = traintest_pca_df[traintest_pca_df['id'].isin(df_train_idx)]
# df_test_pca = traintest_pca_df[traintest_pca_df['id'].isin(df_test_idx)]
## 원본코드가 이상해서 바꿈

df_train_pca = traintest_pca_df.loc[df_train_idx,:]
df_test_pca = traintest_pca_df.loc[df_test_idx,:]
```


```python
len(df_train_pca)
```
    188318




```python
print("df_train.shape :{}".format(df_train.shape))
print("df_test.shape :{}".format(df_test.shape))
print()
print("df_train_pca.shape :{}".format(df_train_pca.shape))
print("df_test_pca.shape :{}".format(df_test_pca.shape))
```

    df_train.shape :(188318, 133)
    df_test.shape :(125546, 133)
    
    df_train_pca.shape :(188318, 10)
    df_test_pca.shape :(125546, 10)
    

<span style="color:blue"> **df_train_pca(pca 축소된 주요인 컬럼)을 id 기준으로 merge 함.</span>  
이때, cont_col 컬럼들은 제외함**(pca 가 cont_col 컬럼에 해당함으로)  

정리하면, train_set 에서, cont 컬럼(뉴메릭컬럼) 은, id 값을 맞추어서, pca 로 추려진 요인들의 값으로 대체함


```python
df_train_ = df_train.drop(cont_col, axis = 1).set_index('id').join(df_train_pca.set_index('id'))
df_test_ = df_test.drop(cont_col, axis = 1).set_index('id').join(df_test_pca.set_index('id'))
```


```python
df_train_.head(2)
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>...</th>
      <th>isTrain</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>...</td>
      <td>True</td>
      <td>2.704532</td>
      <td>-2.101670</td>
      <td>-1.511288</td>
      <td>0.617469</td>
      <td>-0.549860</td>
      <td>-0.219191</td>
      <td>0.946673</td>
      <td>1.106747</td>
      <td>-1.369477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>...</td>
      <td>True</td>
      <td>-1.132381</td>
      <td>-0.395118</td>
      <td>2.283496</td>
      <td>-0.762693</td>
      <td>-0.352918</td>
      <td>-0.593252</td>
      <td>-1.064844</td>
      <td>0.443155</td>
      <td>0.220732</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 127 columns</p>
</div>




```python
df_test_.head(2)
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>...</th>
      <th>loss</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>...</td>
      <td>1597.451430</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>...</td>
      <td>1960.685945</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 127 columns</p>
</div>




```python
print("df_train_.shape :{}".format(df_train_.shape))
print("df_test_.shape :{}".format(df_test_.shape))
```

    df_train_.shape :(188318, 127)
    df_test_.shape :(125546, 127)
    
```python
X = df_train_.drop('loss', axis = 1)
y = np.log(df_train_['loss'])

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)
```


```python
model_1 = CatBoostRegressor(**params)
model_1.fit(X_train, y_train, cat_index, eval_set=(X_validation, y_validation) ,plot=True,verbose=False,use_best_model=True)
```
![png](/assets/images/cat_boost_practice/cat_boo_plot_02.png)


```python
print(model_1.evals_result_.keys())
print(model_1.learning_rate_,model_1.tree_count_)
mae_min_idx = np.argmin(model_1.evals_result_['validation']['MAE'])
print("Minimize MAE value is {:.6f} in Validation Set".format(model_1.evals_result_['validation']['MAE'][mae_min_idx]))
```

    dict_keys(['learn', 'validation'])
    0.10000000149011612 100
    Minimize MAE value is 0.431609 in Validation Set
    


```python
## y 결과값에 log 변환한 값이 나올테니, 이를 다시 원본값으로 돌리기 위해 np.exp()를 씌웠다.
df_test_['loss'] = np.exp(model_1.predict(df_test_.drop('loss', axis=1)))
```


```python
df_test_.to_csv('../dataset/kaggle_submission/acs_submission_1.csv', sep = ',', columns = ['loss'], index=False)
```

## PCA를 활용한 model_1 을 tunning 하면,


```python
# model_1.fit(X_train, y_train, cat_index, eval_set=(X_validation, y_validation) ,plot=True,verbose=False,use_best_model=True)
```

```python
params
```
    {'iterations': 100, 'learning_rate': 0.1, 'eval_metric': 'MAE'}

```python
model_1_tuned = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth =8, task_type = "GPU", eval_metric = "MAE", l2_leaf_reg=3, bagging_temperature=1,one_hot_max_size=0)
model_1_tuned.fit(X_train, y_train, cat_index, eval_set=(X_validation, y_validation) ,plot=True,verbose=False,use_best_model=True)
```
![png](/assets/images/cat_boost_practice/cat_boo_plot_03.png)
    <catboost.core.CatBoostRegressor at 0x151826f1cc8>

확실히 더 좋은 성능을 보여준다.


```python
print(model_1_tuned.evals_result_.keys())
print(model_1_tuned.learning_rate_,model_1_tuned.tree_count_)
mae_min_idx = np.argmin(model_1_tuned.evals_result_['validation']['MAE'])
print("Minimize MAE value is {:.6f} in Validation Set".format(model_1_tuned.evals_result_['validation']['MAE'][mae_min_idx]))
```

    dict_keys(['learn', 'validation'])
    0.05000000074505806 1496
    Minimize MAE value is 0.415938 in Validation Set
    


```python
df_test_['loss'] = np.exp(model_1_tuned.predict(df_test_.drop('loss', axis=1)))
```


```python
df_test_.to_csv('kaggle_submission/submission_1_1.csv', sep = ',', columns = ['loss'], index='id')
```

**kaggle collab 에서, 바로 적용시 활용하는 Code**  
!kaggle competitions submit -c allstate-claims-severity -f submission_1_1.csv -m "Catboost second submission (tuned)"  
<span style='color:red'>MAE 값이 basic PCA 했을때보다, param 수정한 버전이 좀 더 나은 결과를 보여준다.</span>

## skewed_cols를 제외한, 모델링


```python
skewed_cols
```
    ['cont7', 'cont9']


```python
df_train_ = traintest[traintest['isTrain']==True].drop(skewed_cols, axis = 1).set_index('id').drop('isTrain', axis = 1)
df_test_ = traintest[traintest['isTrain']==False].drop(skewed_cols, axis = 1).set_index('id').drop('isTrain', axis = 1)

cat_index = [i for i in range(0,len(df_train_.columns)-1) if 'cat' in df_train_.columns[i]]
cont_index = [i for i in range(0,len(df_train_.columns)-1) if 'cont' in df_train_.columns[i]]
```


```python
X = df_train_.drop('loss', axis = 1)
y = np.log(df_train['loss'])

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)
```


```python
model_2 = CatBoostRegressor(iterations=600, learning_rate=0.05, depth =8, task_type = "GPU", eval_metric = "MAE", l2_leaf_reg=3, bagging_temperature=1,one_hot_max_size=0,use_best_model=True)
model_2.fit(X_train, y_train, cat_index, eval_set=(X_validation, y_validation),verbose=False ,plot=True)
```
![png](/assets/images/cat_boost_practice/cat_boo_plot_04.png)
MetricVisualizer(layout=Layout(align_self='stretch', height='500px'))

```python
print(model_2.evals_result_.keys())
print(model_2.learning_rate_,model_2.tree_count_)
mae_min_idx = np.argmin(model_2.evals_result_['validation']['MAE'])
print("Minimize MAE value is {:.6f} in Validation Set".format(model_2.evals_result_['validation']['MAE'][mae_min_idx]))
```

    dict_keys(['learn', 'validation'])
    0.05000000074505806 600
    Minimize MAE value is 0.412394 in Validation Set
    


```python
df_test_['loss'] = np.exp(model_2.predict(df_test_.drop('loss', axis=1)))
df_test_.to_csv('kaggle_submission/allstate_submission_2.csv', sep = ',', columns = ['loss'], index=True)
```

## 3. Combine categorical variables  
* train_enc : all categoried columns transformed to factorize_colum


```python
train_enc.head(2)
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>...</th>
      <th>cat107</th>
      <th>cat108</th>
      <th>cat109</th>
      <th>cat110</th>
      <th>cat111</th>
      <th>cat112</th>
      <th>cat113</th>
      <th>cat114</th>
      <th>cat115</th>
      <th>cat116</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 116 columns</p>
</div>



cum_cat_1 = []  
for i in range(12,23):  
    cum_cat_1.append('cat'+str(i))

cum_cat_2 = []  
for i in range(24,50):  
    cum_cat_2.append('cat'+str(i))

cum_cat_3 = []  
for i in range(51,72):  
    cum_cat_3.append('cat'+str(i)) 


```python
#create function - new traintest and new encoding
traintest = pd.concat([df_train, df_test], axis = 0)
```


```python
cum_cat_1
```
    ['cat12',
     'cat13',
     'cat14',
     'cat15',
     'cat16',
     'cat17',
     'cat18',
     'cat19',
     'cat20',
     'cat21',
     'cat22']




```python
## origin_code
cum_df_1 = train_enc[cum_cat_1].diff(axis=1).dropna(axis=1)
cum_df_1.columns = [col+'_cum' for col in cum_df_1.columns]
traintest = pd.concat([traintest, cum_df_1], axis=1).drop(cum_cat_1[1:], axis=1)
traintest[traintest['isTrain']==True].isnull().sum().sum()==0
```

dataframe 의 diff 함수는  row, columns 방향에 따라서, 전의 값과의 '차' 를 return 한다.  
[dataframe.diff()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html)  

무슨 의도로, 이렇게 접근하는지 알 수 없기 때문에 종료한다.

## Catboost tunning


```python
from catboost import Pool as pool
```


```python
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
```


```python
D_train = pool(X_train, y_train, cat_features = cat_index)
D_val = pool(X_validation, y_validation, cat_features = cat_index)
```


```python
# number of random sampled hyperparameters
N_HYPEROPT_PROBES = 15

# the sampling aplgorithm 
HYPEROPT_ALGO = tpe.suggest 

def get_catboost_params(space):
    params = dict()
    params['learning_rate'] = space['learning_rate']
    params['depth'] = int(space['depth'])
    params['l2_leaf_reg'] = space['l2_leaf_reg']
    params['one_hot_max_size'] = space['one_hot_max_size']
    return params

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( 'catboost-hyperopt-log.txt', 'w' )


def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    params = get_catboost_params(space)

#     sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
#     params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
#     print('Params: {}'.format(params_str) )
    
    model = CatBoostRegressor(iterations=2000, 
                              learning_rate=params['learning_rate'], 
                              depth =int(params['depth']), 
                              task_type = "GPU",
                              eval_metric = "MAE",
                              l2_leaf_reg=params['l2_leaf_reg'],
                              bagging_temperature=1,
                              one_hot_max_size=params['one_hot_max_size'],
                              use_best_model=True)

    model.fit(D_train, eval_set=D_val, silent=True)
    #y_pred = model.predict(df_test_.drop('loss', axis=1))
    val_loss = model.best_score_['validation_0']['MAE']
    
    if val_loss<cur_best_loss:
        cur_best_loss = val_loss

    return{'loss':cur_best_loss, 'status': STATUS_OK }
```


```python
# --------------------------------------------------------------------------------
space ={
        'depth': hp.quniform("depth", 4, 12, 1),
        'learning_rate': hp.loguniform('learning_rate', -3.0, -0.7),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
        'one_hot_max_size': hp.quniform("one_hot_max_size", 0, 15, 1)
       }
```


```python
trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')
```
hyperropt 라이브러리르 활용하는 방법이나, 현재 Local PC 의 한계로, Collab에서 수행하는 것을 권한다.  
여기까지로 포스팅을 마친다.