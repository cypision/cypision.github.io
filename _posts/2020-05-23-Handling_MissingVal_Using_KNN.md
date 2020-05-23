---
title:  "Handing Missing Value usin KNN"
excerpt: "using sckit-learn or handmade Function Test"

categories:
  - Useful_Fuction
tags:
  - KNN
  - OrdinalEncoder
  - sklearn.impute import KNNImputer  
last_modified_at: 2020-05-22T23:06:00-05:00
---

Missing Value 를 처리하는 방식은 늘 DS 직군의 사람들에게는 숙제와 같다.  
이를 간단히도 처리할수 있지만, 좀더 나은 방법으로 처리하기위해 여러 도전을 했고, 오늘은 그 중 KNN 방법에 대해서, 알아본다.

## Reference  
- [Medium article 01](https://towardsdatascience.com/the-use-of-knn-for-missing-values-cf33d935c637)  
- [Medium article 02](https://medium.com/@amrwrites/knn-based-missing-value-imputation-using-scikit-learn-802fceb5b2ea)


```python
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../dataset/titanic_train.csv')
test = pd.read_csv('../dataset/titanic_test.csv')
# train = pd.read_csv('../ML_Area/data_source/titanic_train.csv')
# test = pd.read_csv('../ML_Area/data_source/titanic_test.csv')
# ML_Area\data_source
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots
```


```python
pd.concat([train.isnull().sum(),train.dtypes],axis=1,names=[['null','dtype']])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>177</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>Cabin</th>
      <td>687</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>



'Age'  : numeric 변수  
'Cabin','Embarked : categorical 변수


```python
# !python -m pip install scikit-learn==0.23.1
# !pip show version scikit-learn
```

## 1. Scikit-Learn KNNImputer


```python
import sklearn
sklearn.show_versions()
```

    
    System:
        python: 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
    executable: C:\ProgramData\Anaconda3\envs\test\python.exe
       machine: Windows-10-10.0.18362-SP0
    
    Python dependencies:
              pip: 19.0.3
       setuptools: 41.0.0
          sklearn: 0.23.1
            numpy: 1.16.4
            scipy: 1.2.1
           Cython: None
           pandas: 0.24.2
       matplotlib: 3.0.3
           joblib: 0.15.1
    threadpoolctl: 2.0.0
    
    Built with OpenMP: True
    


```python
## scikit-learn : 0.23.1 이상부터 가능
from sklearn.impute import KNNImputer
```


```python
## 'Age','Cabin'
train.iloc[10:20]
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0</td>
      <td>3</td>
      <td>Saundercock, Mr. William Henry</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>A/5. 2151</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0</td>
      <td>3</td>
      <td>Andersson, Mr. Anders Johan</td>
      <td>male</td>
      <td>39.0</td>
      <td>1</td>
      <td>5</td>
      <td>347082</td>
      <td>31.2750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0</td>
      <td>3</td>
      <td>Vestrom, Miss. Hulda Amanda Adolfina</td>
      <td>female</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>350406</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>1</td>
      <td>2</td>
      <td>Hewlett, Mrs. (Mary D Kingcome)</td>
      <td>female</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
      <td>248706</td>
      <td>16.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Master. Eugene</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>244373</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Planke, Mrs. Julius (Emelia Maria Vande...</td>
      <td>female</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>345763</td>
      <td>18.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2649</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



`KNNImputer` 는 오로지 뉴메릭변수에서만, 활용가능하다.


```python
not_cat_df = train[[col for col in train.columns if col not in ['Cabin','Embarked']]].copy() ## Cabin 제외한 df
```


```python
int_col = train.columns[train.dtypes != 'object'].to_list()
cat_col = train.columns[train.dtypes == 'object'].to_list()
```


```python
imputer = KNNImputer(n_neighbors=5)
# df_filled = imputer.fit_transform(not_cat_df) ## parameter df 에 카테고리변수가 있으면 무조건 에러난다. 즉. 카테고리칼 변수는 KNN 으로 missing value 처리할 수 없다.
df_filled = imputer.fit_transform(train[int_col])
```


```python
df_filled[10:20,:]
```




    array([[11.    ,  1.    ,  3.    ,  4.    ,  1.    ,  1.    , 16.7   ],
           [12.    ,  1.    ,  1.    , 58.    ,  0.    ,  0.    , 26.55  ],
           [13.    ,  0.    ,  3.    , 20.    ,  0.    ,  0.    ,  8.05  ],
           [14.    ,  0.    ,  3.    , 39.    ,  1.    ,  5.    , 31.275 ],
           [15.    ,  0.    ,  3.    , 14.    ,  0.    ,  0.    ,  7.8542],
           [16.    ,  1.    ,  2.    , 55.    ,  0.    ,  0.    , 16.    ],
           [17.    ,  0.    ,  3.    ,  2.    ,  4.    ,  1.    , 29.125 ],
           [18.    ,  1.    ,  2.    , 29.8   ,  0.    ,  0.    , 13.    ],
           [19.    ,  0.    ,  3.    , 31.    ,  1.    ,  0.    , 18.    ],
           [20.    ,  1.    ,  3.    , 27.6   ,  0.    ,  0.    ,  7.225 ]])



비교해보면, 승객 18,20번이 na 였는데  
`18 : NaN -> 29.8`  
`20 : NaN -> 27.6` 으로 바뀐것을 알 수 있다.

## 2. fancyimpute KNNImputer

[Medium Post Preprocessing: Encode and KNN Impute All Categorical Features Fast](https://towardsdatascience.com/preprocessing-encode-and-knn-impute-all-categorical-features-fast-b05f50b4dfaa)


```python
# !python -m pip install fancyimpute
```


```python
from fancyimpute import KNN
```

    Using TensorFlow backend.
    

#### ################```small talk abount OrdinalEncoder()``` Start################
- parameter 를 받는 형식과 return 값이 다른다.
 > 1) OrdinalEncoder :(n_samples, n_features)   2D를 받고, 2D `categories_` 를 사용한다. 또한, 이는 다른 인코더(OHE)와 같다고 할 수 있다.  
 > 2) LabelEncoder: (n_samples,) 1D를 받고 `classes_` param을 사용한다.  LabelEncoder 는 loop형으로 변환를 해서, 시간이 오래 걸린다.  
- OrdinalEncoder를 순서가 있는 특성에 적용하는 클래스라고 혼동하지 마세요. OrdinalEncoder는 순서가 없는 범주형 특성을 정수로 변환하는 클래스입니다.


```python
enc = OrdinalEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 10]]
enc.fit(X)
```




    OrdinalEncoder()




```python
enc.categories_
```




    [array(['Female', 'Male'], dtype=object), array([1, 3, 10], dtype=object)]




```python
enc.transform([['Female', 1],['Female', 3],['Female', 10],['Male', 10],['Male', 3],['Male', 1]])
```




    array([[0., 0.],
           [0., 1.],
           [0., 2.],
           [1., 2.],
           [1., 1.],
           [1., 0.]])



위 내용을 보면, Male,Feamale , 1,3,10 을 연계해서 Encoding 시킨것을 볼수 있다. 더욱이, 1,3,10 을 각각의 범주형으로 바꾸어 버린점을 흥미롭다.  
위 그림처럼, 성별구분(2개) * (숫자 3개 1,3,10) 해서, 총 6개의 조합을 인코딩변환할 수 있다.
#### ################ small talk abount OrdinalEncoder() End################ 


```python
from sklearn.preprocessing import OrdinalEncoder 
pd.options.display.max_columns = None
```


```python
## 예제에서 잼있게도 titanic data 를 사용하지만, kaggle 과 구성이 달라서, 일단 따라간다.
impute_data = sns.load_dataset('titanic')
impute_data.head()
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([impute_data.isnull().sum(),impute_data.dtypes],axis=1,names=[['null','dtype']])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>survived</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>pclass</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>age</th>
      <td>177</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>sibsp</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>parch</th>
      <td>0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>fare</th>
      <td>0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>embarked</th>
      <td>2</td>
      <td>object</td>
    </tr>
    <tr>
      <th>class</th>
      <td>0</td>
      <td>category</td>
    </tr>
    <tr>
      <th>who</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>adult_male</th>
      <td>0</td>
      <td>bool</td>
    </tr>
    <tr>
      <th>deck</th>
      <td>688</td>
      <td>category</td>
    </tr>
    <tr>
      <th>embark_town</th>
      <td>2</td>
      <td>object</td>
    </tr>
    <tr>
      <th>alive</th>
      <td>0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>alone</th>
      <td>0</td>
      <td>bool</td>
    </tr>
  </tbody>
</table>
</div>



pandas type 에서, category 로 되어 있는 것은 사실상 큰 의미가 없다. 따라서, 이를 제거해주고 목적에 충실하기로 한다.

```python
impute_data['deck1'] = impute_data['deck'].astype(object,axit=0)
impute_data['class1'] = impute_data['class'].astype(object,axit=0)
impute_data = impute_data.drop(columns=['deck','class'],axis=1)
```

```python
print(impute_data.shape)
impute_data.isnull().sum()
```

    (891, 15)
    survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         2
    who              0
    adult_male       0
    embark_town      2
    alive            0
    alone            0
    deck1          688
    class1           0
    dtype: int64



embarked,embark_town,deck1 이 카테고리컬 변수이면서, null 이 존재하는걸 알 수 있다. 추후 아래과정을 통해서, 이것이 어떻게 변경되는지 알아보자


```python
impute_data[impute_data['embarked'].isnull()]
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>who</th>
      <th>adult_male</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
      <th>deck1</th>
      <th>class1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>yes</td>
      <td>True</td>
      <td>B</td>
      <td>First</td>
    </tr>
    <tr>
      <th>829</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>yes</td>
      <td>True</td>
      <td>B</td>
      <td>First</td>
    </tr>
  </tbody>
</table>
</div>




```python
impute_data[impute_data['embark_town'].isnull()]
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>who</th>
      <th>adult_male</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
      <th>deck1</th>
      <th>class1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>yes</td>
      <td>True</td>
      <td>B</td>
      <td>First</td>
    </tr>
    <tr>
      <th>829</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>yes</td>
      <td>True</td>
      <td>B</td>
      <td>First</td>
    </tr>
  </tbody>
</table>
</div>




```python
impute_data[['embarked','embark_town','deck1']].iloc[60:65]
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
      <th>embarked</th>
      <th>embark_town</th>
      <th>deck1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>60</th>
      <td>C</td>
      <td>Cherbourg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>61</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>B</td>
    </tr>
    <tr>
      <th>62</th>
      <td>S</td>
      <td>Southampton</td>
      <td>C</td>
    </tr>
    <tr>
      <th>63</th>
      <td>S</td>
      <td>Southampton</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>64</th>
      <td>C</td>
      <td>Cherbourg</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#instantiate both packages to use
encoder = OrdinalEncoder()
imputer = KNN()
# create a list of categorical columns to iterate over
cat_cols = ['embarked','class1','deck1','who','embark_town','sex','adult_male','alive','alone']

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
#     data.loc[data.notnull(),:] = np.squeeze(impute_ordinal)
    return data

#create a for loop to iterate through each column in the data
for columns in cat_cols:
    encode(impute_data[columns])
```


```python
#create a for loop to iterate through each column in the data
for columns in cat_cols:
    encode(impute_data[columns])
```

상기식을 하나씩 분석해보면, 

```
encoder = OrdinalEncoder()
imputer = KNN()
cat_cols = ['embarked','class1','deck1','who','embark_town','sex','adult_male','alive','alone']
```

`embarked` 를 예를 들어서, 살펴보면, S,C,Q 로 이루어져있다.


```python
impute_data.embarked.value_counts()
```




    S    644
    C    168
    Q     77
    Name: embarked, dtype: int64




```python
## na의 항목을 제거하고, 
nonulls = np.array(impute_data.embarked.dropna())
print(len(nonulls),nonulls.shape,impute_data.shape)
## Ordinal Encoder 적용을 위해, 2차원으로 만든다.
impute_reshape = nonulls.reshape(-1,1)
print(impute_reshape.shape,'\n',impute_reshape[0:5])
```

    889 (889,) (891, 15)
    (889, 1) 
     [['S']
     ['C']
     ['S']
     ['S']
     ['S']]
    


```python
## null 을 제거한 이후, 2차원으로 만들고, 이를 숫자형으로 바꿔주면 (이때, 숫자값으로 들어가게 된다.)
impute_ordinal = encoder.fit_transform(impute_reshape)
print(impute_ordinal.shape,'\n',impute_ordinal[0:5])
```

    (889, 1) 
     [[2.]
     [0.]
     [2.]
     [2.]
     [2.]]
    


```python
# #Assign back encoded values to non-null values 숫자형으로 바꾼값을 원래값 대신 넣어준다.
impute_data.loc[impute_data.embarked.notnull(),:].loc[:,'embarked'] = np.squeeze(impute_ordinal) ## squeeze 는 dim 을 축소하는 함수
```


```python
impute_data.iloc[60:65] ## embarked값이 숫자형으로 바뀐것을 확인할 수 있다.
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>who</th>
      <th>adult_male</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
      <th>deck1</th>
      <th>class1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>60</th>
      <td>0</td>
      <td>3</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.2292</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>80.0000</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>45.0</td>
      <td>1</td>
      <td>0</td>
      <td>83.4750</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0</td>
      <td>3</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>2</td>
      <td>27.9000</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>27.7208</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



상기 과정은 결국, 각 카테고리컬 컬럼들을 null 을 제외한 이후, 인코딩하여 바꾸어주는 1번째 과정이다.
```for columns in cat_cols:
    encode(impute_data[columns])```


```python
## 여기서, 실제로 null 인 값들을 KNN 방식으로 impute 한다.
# impute data and convert 
encode_data = pd.DataFrame(np.round(imputer.fit_transform(impute_data)),columns = impute_data.columns)
```

    Imputing row 1/891 with 1 missing, elapsed time: 0.128
    Imputing row 101/891 with 1 missing, elapsed time: 0.130
    Imputing row 201/891 with 1 missing, elapsed time: 0.131
    Imputing row 301/891 with 2 missing, elapsed time: 0.133
    Imputing row 401/891 with 1 missing, elapsed time: 0.135
    Imputing row 501/891 with 1 missing, elapsed time: 0.136
    Imputing row 601/891 with 1 missing, elapsed time: 0.138
    Imputing row 701/891 with 0 missing, elapsed time: 0.140
    Imputing row 801/891 with 1 missing, elapsed time: 0.141
    


```python
encode_data.iloc[60:65]
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>who</th>
      <th>adult_male</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
      <th>deck1</th>
      <th>class1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>60</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>80.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>45.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>83.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>28.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



과정을 요약해보면,  
1. null 인 얘들을 제외하고, ordinal_encoder (이때, 숫자형으로 변하면서, 카테코리컬 컬럼이, order 순서를 가지게 되는것은 피할 수 없음)  
2. 이후, 이를 KNN Imputer로 null 처리한다.  
카테고리컬 컬럼이, order 파워를 가지게 되는것은 딱히 좋아보이지 않는다.  
