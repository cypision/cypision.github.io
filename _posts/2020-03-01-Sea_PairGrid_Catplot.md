---
title:  'Seaborn Basic 01'
excerpt: 'Seaborn pariplot, catplot, FacetGrid'

categories:
  - Visual
tags:
  - Seaborn
  - pariplot
  - catplot
  - FacetGrid
last_modified_at: 2020-03-01T17:06:00-05:00
---

Seaborn Gallery 에 살펴보면, 
pariplot, catplot, FacetGrid 가 등장한다

link : https://seaborn.pydata.org/examples/index.html

더욱이 pariplot 파트는 4개, catplot은 2개, facetGrid 도 3개 이상 존재한다. 이들 관계가 무엇인지 살펴보는것이 이번 수업의 목표이다.

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
%matplotlib inline
```

## PairPlot


```python
iris = sns.load_dataset("iris")
g = sns.pairplot(iris, hue="species")
# g.map(sns.pointplot)
```


![png](/assets/images/output_3_0.png)



```python
print(type(g))
```

    <class 'seaborn.axisgrid.PairGrid'>
    
type 형태를 봐서는 Pairplot 은 diagos선을 중심으로 위.아래가 같다. 그리고, FacetGrid 와 연관되어 있다.
결론부터 말하자만, 
PairGrid, Catplot, Pairplot 은 모두 FacetGrid 객체를 좀 더 편리하게 사용하기 위해 만들어진 하위 클래스 개념이다
## CatPlot

#### catplot 은 <class 'seaborn.axisgrid.FacetGrid'> 를 사용한다
    주로 "g = " 으로 객체를 따로 받아서 사용한다
    category 변수와, 숫자형변수의 관계를 보여주기 위해서 사용한다.
    col 속성을 이용하면, 변수dim 을 더 확장하여 보여주게 된다 (여러개 그리고 싶을때 핵심)
    내가 선택한 변수관계에 따라서, 자동으로 gird 구조의 subplot을 그려주는 것이 특징
    kind 로 표현하는 그래프 형식을 여러개 바꿀수 있다


```python
# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")
```


```python
titanic.head(3)
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
  </tbody>
</table>
</div>




```python
# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="class", y="survived", hue="sex",col="who", data=titanic,height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
```




    <seaborn.axisgrid.FacetGrid at 0x1bf45e652b0>




![png](/assets/images/output_10_1.png)



```python
print(type(g))
```

    <class 'seaborn.axisgrid.FacetGrid'>
    


```python
# Load the example exercise dataset
df = sns.load_dataset("exercise")
```


```python
# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(x="time", y="pulse", hue="kind", col="diet",capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=df)
g.despine(left=True)
```




    <seaborn.axisgrid.FacetGrid at 0x1bf45248518>




![png](/assets/images/output_13_1.png)


## PairGrid

#### PairGrid 는 PairPlot의 불편함을 없애려고 나온것이며, 대각선을 기준으로 위, 아래에 대해 각각의 시각화도구를 적용할 수 있는 점이 가장 큰 차이점이다
    숫자형변수 - 숫자형변수 의 관계를 기본으로 그린다 (default 로 x_vars=None, y_vars = None 일 경우)
    숫자형변수 - 카테고리컬변수 관계도 그릴 수 있으며, (x_vars , y_vars 를 설정하면)
    내가 선택한 변수관계에 따라서, 자동으로 gird 구조의 subplot을 그려주는 것이 특징


```python
gpg = sns.PairGrid(iris, diag_sharey=False, hue='species')
gpg.map_upper(sns.scatterplot)
gpg.map_lower(sns.kdeplot)
gpg.map_diag(sns.kdeplot, lw=2)
```




    <seaborn.axisgrid.PairGrid at 0x1bf491a0ac8>




![png](/assets/images/output_16_1.png)



```python
print(type(gpg))
```

    <class 'seaborn.axisgrid.PairGrid'>
    


```python
# Set up a grid to plot survival probability against several variables
gtitan = sns.PairGrid(titanic, y_vars="survived",x_vars=["class", "sex", "who", "alone"],height=5, aspect=.5)

# Draw a seaborn pointplot onto each Axes
gtitan.map(sns.pointplot, scale=1.3, errwidth=4, color="xkcd:plum")
gtitan.set(ylim=(0, 1))
sns.despine(fig=gtitan.fig, left=True)
```


![png](/assets/images/output_18_0.png)



```python
# Load the dataset
crashes = sns.load_dataset("car_crashes")
```


```python
# Make the PairGrid
gcrash = sns.PairGrid(crashes.sort_values("total", ascending=False),x_vars=crashes.columns[:-3], y_vars=["abbrev"],
                      height=10, aspect=.25)

# Draw a dot plot using the stripplot function
gcrash.map(sns.stripplot, size=10, orient="h",palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")
#############################################################################
# Use semantically meaningful titles for the columns
titles = ["Total crashes", "Speeding crashes", "Alcohol crashes","Not distracted crashes", "No previous crashes"]
## gcrash.axes 각각의 subplot 변 title 를 붙여주는 과정임
for ax, title in zip(gcrash.axes.flat, titles):
    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
```


![png](/assets/images/output_20_0.png)



```python
gcrash.axes
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001BF47261F28>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001BF47281710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001BF472A6EF0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001BF472D4710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001BF472FBF28>]],
          dtype=object)


