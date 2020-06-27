---
title:  "Principal Component Analysis Using Python"
excerpt: "python 으로 하는 주성분 분석(PCA)"

categories:
  - Machine-Learning
tags:
  - PCA
  - EigenDecomposition
  - Dimension reduction
  - Feature extraction
last_modified_at: 2020-04-26T21:15:00-05:00
---

## Principal component analysis  
### reference  
- 코드관련 : StatQuest youtube - "PCA in Python" 을 참고하여 진행했다
- 포아송분포관련 blog : https://blog.naver.com/PostView.nhn?blogId=mykepzzang&logNo=220840724901

## Loding Library


```python
import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt # NOTE: This was tested with matplotlib v. 2.1.0
```

## Data Generation Code


```python
## In this example, the data is in a data frame called data.
## Columns are individual samples (i.e. cells)
## Rows are measurements taken for all the samples (i.e. genes)
## Just for the sake of the example, we'll use made up data...
genes = ['gene' + str(i) for i in range(1,101)]
 
wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]
 
data = pd.DataFrame(columns=[*wt, *ko], index=genes)
 
for gene in data.index:
    data.loc[gene,'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    data.loc[gene,'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
```

## Special Poisson Distribution
random.poisson 은 lam 다 값을 통하여, 랜덤하게 값을 뽑는데, 이때 lambda의 범위를 10 ~ 1000 으로 한다.  
이걸 활용하여, 이 lambda 값을 활용하여, 포아송 분포에서, 랜덤하게 5(size=5)개를 추출한다. 


```python
# ![img](../dataset/PCA/1280px-Poisson_pmf.svg.PNG)
```


```python
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)
%matplotlib inline
```

#### 포아송분포 정의  
1) 모수 : 모집단의 특성을 나타내는 수치  ex> 한 시간동안 사무실에 걸려온 전화의 수, 일주일동안 기병들이 낙마하는 횟수  
2) 포아송분포의 모수 : "단위시간 또는 단위공간에서 평균발생횟수"  

정의하면, __"단위 시간, 단위 공간에 어떤 사건이 몇 번 발생할 것인가를 표현하는 이산 확률분포"__


```python
# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
sns.despine(left=True)

ax1=sns.distplot(np.random.poisson(lam=1, size=100), color="b", kde=True, ax=axes[0, 0],label="lambda:1")
ax1.legend()
ax2=sns.distplot(np.random.poisson(lam=5, size=100), color="r", kde=True, ax=axes[0, 1],label="lambda:5")
ax2.legend()
ax3=sns.distplot(np.random.poisson(lam=10, size=100), color="g", kde=True, ax=axes[1, 0],label="lambda:10")
ax3.legend()
ax4=sns.distplot(np.random.poisson(lam=20, size=100), color="m", kde=True, ax=axes[1, 1],label="lambda:20")
ax4.legend()

plt.setp(axes, yticks=[])
plt.tight_layout()
```


![png](/assets/images/PCA/output_10_0.png)

```python
print(data.head()) ## gene 이 feature고, wt1~5,ko1~5 가 sample이다.
print()
print(data.shape)
```

            wt1  wt2  wt3  wt4  wt5  ko1  ko2  ko3  ko4  ko5
    gene1   757  753  765  705  792  359  379  378  395  396
    gene2  1005  940  946  909  992  861  896  884  942  880
    gene3   682  679  681  669  741  116   94   87   91   96
    gene4   281  266  286  307  301  882  833  848  849  842
    gene5    51   48   48   30   42  856  788  823  882  849
    
    (100, 10)
    


```python
## 데이터 형식을 익숙하게 row * feature 구조로 바꾼다
data01 = data.T
print(data01.shape)
```

    (10, 100)
    


```python
import warnings
warnings.filterwarnings(action='ignore')
```

## Perform PCA on the data

센터화 + scaling 하는 부분은 어떤 library를 사용해도 상관없다.


```python
# First center and scale the data
scaled_data00 = preprocessing.scale(data01)
sc = preprocessing.StandardScaler()
scaled_data01 = sc.fit_transform(data01)
```


```python
scaled_data00[0:2,0:4]
```




    array([[ 1.00688443,  1.74565139,  0.96960136, -1.02401425],
           [ 0.98558598,  0.31838925,  0.95951536, -1.07725589]])




```python
scaled_data01[0:2,0:4]
```




    array([[ 1.00688443,  1.74565139,  0.96960136, -1.02401425],
           [ 0.98558598,  0.31838925,  0.95951536, -1.07725589]])



sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)  
- n_components : None일경우, min(n_samples, n_features) 이며, n_components 라이브러기가 자동으로 min(n_samples, n_features) 골라서, 수행하게 된다. 저 조건은 변경불가하다  
- svd_solver  
>default `auto`: The solver is selected by a default policy based on X.shape and n_components: if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient ‘randomized’ method is enabled. Otherwise the exact full SVD is computed and optionally truncated afterwards.  
>`full`: run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components by postprocessing  
>`arpack`: run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires strictly 0 < n_components < min(X.shape)  
>`randomized` : run randomized SVD by the method of Halko et al.


```python
pca = PCA(n_components=None,svd_solver='full') # create a PCA object ## n_components == min(n_samples, n_features)
pca.fit(scaled_data01) # do the math => eigen decomposition 이 이루어진다.
pca_data = pca.transform(scaled_data01) # get PCA coordinates for scaled_data
```


```python
pca_data.shape
```




    (10, 10)



`여기서 얻어지는 pca_data 는 각각의 주성분사용하여, 원래 데이터를 선형변환한 값이다.`

![image.png](/assets/images/PCA/pca01.PNG)

pca_data 는 상기 그림에서 e 매트릭스들을 곱한값이라고 여기면 된다.
실제 e 에 해당하는 값은 pca.components_ 다.  
* components_ : array, shape (n_components, n_features)  
> Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.  
`공분산의 eigenvector 를 뜻한다.`
* explained_variance_ : array, shape (n_components,)
> The amount of variance explained by each of the selected components.  
Equal to n_components largest eigenvalues of the covariance matrix of X.  
`공분산의  eigenvalue, lambda 로 이루어진 함수를 뜻한다.`

상기 식처럼 표현하면,  
`feature들의 공분산행렬`: __pca.get_covariance()__  
`공분산행렬의 eigen vector`: __pca.components___    
`공분산행렬의 eigen value`: __pca.explained_variance___    


```python
## feature들의 공분산행렬
print("feature 100개를 공분산행렬A이라 할때\n A Covariance(): shape{}".format(pca.get_covariance().shape))
print(" A행렬의 고유벡터(eigen vactor) 행렬 P: {}".format(pca.components_.shape))
print(" A행렬의 고유치(eigen value) 대각행렬 P: {}".format(np.diag(pca.explained_variance_).shape))
```

    feature 100개를 공분산행렬A이라 할때
     A Covariance(): shape(100, 100)
     A행렬의 고유벡터(eigen vactor) 행렬 P: (10, 100)
     A행렬의 고유치(eigen value) 대각행렬 P: (10, 10)
    


```python
## 실제결과 비교하기
print(pca.get_covariance()[0][0:5])

tmp = np.dot(pca.components_.T,np.diag(pca.explained_variance_))
rslt = np.dot(tmp,pca.components_)

print(rslt[0][0:5])
```

    [ 1.11111111  0.86121875  1.10571855 -1.10431886 -1.09973187]
    [ 1.11111111  0.86121875  1.10571855 -1.10431886 -1.09973187]
    

## Draw a scree plot and a PCA plot

`여기서 얻어지는 pca_data 는 각각의 주성분사용하여, 원래 데이터를 선형변환한 값이다.`


```python
pca_df
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
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>wt1</th>
      <td>-9.317911</td>
      <td>0.103542</td>
      <td>-2.396421</td>
      <td>0.557520</td>
      <td>1.392658</td>
      <td>-0.527304</td>
      <td>0.364037</td>
      <td>1.336152</td>
      <td>-0.394318</td>
      <td>3.207748e-15</td>
    </tr>
    <tr>
      <th>wt2</th>
      <td>-9.421881</td>
      <td>0.266279</td>
      <td>-0.385886</td>
      <td>-0.128603</td>
      <td>-0.008204</td>
      <td>1.414610</td>
      <td>-1.136689</td>
      <td>0.029911</td>
      <td>1.405821</td>
      <td>-7.493186e-16</td>
    </tr>
    <tr>
      <th>wt3</th>
      <td>-9.234666</td>
      <td>0.723066</td>
      <td>3.408363</td>
      <td>0.369731</td>
      <td>1.239508</td>
      <td>-1.286638</td>
      <td>0.133903</td>
      <td>-0.094057</td>
      <td>0.089323</td>
      <td>3.872959e-15</td>
    </tr>
    <tr>
      <th>wt4</th>
      <td>-9.505680</td>
      <td>-1.758335</td>
      <td>1.025360</td>
      <td>-0.827711</td>
      <td>-1.032711</td>
      <td>1.644661</td>
      <td>-0.023192</td>
      <td>0.074406</td>
      <td>-1.131210</td>
      <td>1.100753e-15</td>
    </tr>
    <tr>
      <th>wt5</th>
      <td>-9.641283</td>
      <td>0.802679</td>
      <td>-1.608738</td>
      <td>-0.011221</td>
      <td>-1.576811</td>
      <td>-1.255885</td>
      <td>0.636638</td>
      <td>-1.304464</td>
      <td>0.021058</td>
      <td>1.001444e-15</td>
    </tr>
    <tr>
      <th>ko1</th>
      <td>9.261086</td>
      <td>-0.634707</td>
      <td>-0.891183</td>
      <td>-0.474136</td>
      <td>2.264253</td>
      <td>0.442054</td>
      <td>-0.461911</td>
      <td>-1.304797</td>
      <td>-0.381146</td>
      <td>-2.791638e-15</td>
    </tr>
    <tr>
      <th>ko2</th>
      <td>9.644877</td>
      <td>-0.439244</td>
      <td>-0.000261</td>
      <td>-1.497199</td>
      <td>-0.985670</td>
      <td>-1.593969</td>
      <td>-1.565809</td>
      <td>0.602234</td>
      <td>-0.136152</td>
      <td>-7.438973e-16</td>
    </tr>
    <tr>
      <th>ko3</th>
      <td>9.832206</td>
      <td>4.213431</td>
      <td>0.310651</td>
      <td>-0.128132</td>
      <td>-0.427786</td>
      <td>1.022025</td>
      <td>0.438142</td>
      <td>0.291788</td>
      <td>-0.276545</td>
      <td>1.249172e-15</td>
    </tr>
    <tr>
      <th>ko4</th>
      <td>9.158681</td>
      <td>-1.351882</td>
      <td>0.247421</td>
      <td>3.425707</td>
      <td>-0.780595</td>
      <td>0.092005</td>
      <td>-0.178542</td>
      <td>0.028724</td>
      <td>0.009418</td>
      <td>-8.802470e-16</td>
    </tr>
    <tr>
      <th>ko5</th>
      <td>9.224571</td>
      <td>-1.924829</td>
      <td>0.290694</td>
      <td>-1.285956</td>
      <td>-0.084641</td>
      <td>0.048441</td>
      <td>1.793423</td>
      <td>0.340103</td>
      <td>0.793751</td>
      <td>-1.416208e-15</td>
    </tr>
  </tbody>
</table>
</div>




```python
#the following code makes a fancy looking plot using PC1 and PC2
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)
 
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
 
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
```


![png](/assets/images/PCA/output_31_0.png)


## 총분산으로 비율 확인하기


```python
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
```


```python
print(per_var.sum())
per_var ## 총분산값을 보여준다.
## 순서데로, PC1의 요소의 Var 값이다.
```

    100.00000000000001
    




    array([88.9,  2.8,  2.2,  1.7,  1.4,  1.2,  0.8,  0.6,  0.4,  0. ])




```python
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
 
```


![png](/assets/images/PCA/output_35_0.png)


## Determine which genes had the biggest influence on PC1


```python
pca.components_.shape
```




    (10, 100)




```python
## get the name of the top 10 measurements (genes) that contribute
## most to pc1.
## first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index=genes) ## pca.components_[0] 첫번째 고유벡터
## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values
 
## print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes])
```

    gene96   -0.106024
    gene97    0.106020
    gene10    0.106018
    gene79    0.106002
    gene49   -0.105999
    gene51    0.105978
    gene73    0.105965
    gene18   -0.105957
    gene59   -0.105953
    gene38   -0.105941
    dtype: float64
    

위 결과에서 보면, 특정한 cell 이 PC1의 구성에 큰 영향을 주는 상태가 아니라, 모두 일정하게 영향을 주고 있는 것으로 보인다. 즉, 특별한 cell 이 영향을 미친다고 할 수 없다.

## Special PCA
random.poisson 은 lam 다 값을 통하여, 랜덤하게 값을 뽑는데, 이때 lambda의 범위를 10 ~ 1000 으로 한다.  
이걸 활용하여, 이 lambda 값을 활용하여, 포아송 분포에서, 랜덤하게 5(size=5)개를 추출한다. 


```python
x1=np.array([1,2,1]).reshape(3,1)
x2=np.array([4,2,13]).reshape(3,1)
x3=np.array([7,8,1]).reshape(3,1)
x4=np.array([8,4,5]).reshape(3,1)
X=np.c_[x1,x2,x3,x4];X
```




    array([[ 1,  4,  7,  8],
           [ 2,  2,  8,  4],
           [ 1, 13,  1,  5]])


