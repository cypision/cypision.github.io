---
title:  "K-Fold Target Encoding"
excerpt: "Encoding"

categories:
  - Useful_Fuction
tags:
  - K-fold
  - Target Encoding
  - Categorical Variable Encoding
last_modified_at: 2020-05-23T08:19:00-05:00
---

이번 포스팅은 2014년 Catboosting과 더불어, 등장했던, Target Encoding 에 대해서 알아보고자 한다.  특히 K-fold TargetEncoding 인데, 개념정리는 간단히 하고  
실제로 구현된 코드를 가지고 활용하는데 의의를 둔다.  
실제 이론내용은 하기 링크를 참조하기 바란다.

## Reference  
- [Pourya Medium Blog](https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b)  
- [Pourya GitHub](https://github.com/pourya-ir/Medium/blob/master/K-fold-target-enc/K-fold-Target-Encoding.ipynb)
- [자주가는 한국어 Blog 설명](https://dailyheumsi.tistory.com/120?category=877153)

## K-Fold Target Encoding 

- 기존의 Target Encoding 이, Target 값을 활용하기에, data Leackage 현상이 있고, Overfitting 이 심했다.  
- 이를 보완하기 위해, Fold를 구현, Validation 폴드에 해당하는 값들을 당시 Train 폴드의 값을 활용해서, Encoding 시킨다는 컨셉  
(상세내용은 Reference)


```python
import pandas as pd
import numpy as np
from sklearn import base
from sklearn.model_selection import KFold
```

그냥 아무 Data 만들기


```python
def getRandomDataFrame(data, numCol):
    if data== 'train':
        key = ["A" if x ==0  else 'B' for x in np.random.randint(2, size=(numCol,))]
        value = np.random.randint(2, size=(numCol,))
        df = pd.DataFrame({'Feature':key, 'Target':value})
        return df
    
    elif data=='test':
        key = ["A" if x ==0  else 'B' for x in np.random.randint(2, size=(numCol,))]
        df = pd.DataFrame({'Feature':key})

        return df
    else:
        print(';)')
```


```python
train = getRandomDataFrame('train',20)
test = getRandomDataFrame('test',5)
```


```python
train
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
      <th>Feature</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>A</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test
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
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>




```python
class KFoldTargetEncoderTrain(base.BaseEstimator,base.TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold,
                   shuffle = False, random_state=2019)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind],X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace = True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {}is {}.'.format(col_mean_name,self.targetName,np.corrcoef(X[self.targetName].values,\
                                                                                                                           encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X
```


```python
targetc = KFoldTargetEncoderTrain('Feature','Target',n_fold=5)
new_train = targetc.fit_transform(train)
```

    Correlation between the new feature, Feature_Kfold_Target_Enc and, Targetis -0.42682437741737905.
    

    C:\ProgramData\Anaconda3\envs\test\lib\site-packages\sklearn\model_selection\_split.py:297: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.
      FutureWarning
    


```python
new_train
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
      <th>Feature</th>
      <th>Target</th>
      <th>Feature_Kfold_Target_Enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>1</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>1</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>1</td>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A</td>
      <td>0</td>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A</td>
      <td>0</td>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B</td>
      <td>0</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>8</th>
      <td>B</td>
      <td>0</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A</td>
      <td>1</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>10</th>
      <td>B</td>
      <td>0</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B</td>
      <td>1</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>B</td>
      <td>0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>B</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>A</td>
      <td>0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>A</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>A</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A</td>
      <td>0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>A</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>A</td>
      <td>0</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>


