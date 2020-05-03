---
title:  "Association Rule more using python"
excerpt: "python 으로 하는 연관성분석"

categories:
  - Machine-Learning
tags:
  - mlxtend
  - Recomandation
  - association rule
  - 장바구니 분석
  - 연관성 분석
last_modified_at: 2020-05-03T21:15:00-05:00
---

기존에 했던, mlstend 에서 추가기능 활용을 하기 위한 포스팅이다.


```python
import pandas as pd
import numpy as np
import itertools ## 조합만들때, 필요하다.
```


```python
mdf = pd.read_csv('D:/★2020_ML_DL_Project/Alchemy/dataset/marketbasket.csv',encoding='UTF8',header='infer')
```


```python
print(mdf.shape) ## 트랜잭션 수는 1361 건이다.
print(mdf.columns) ## 품목이 255건이나 된다.
mdf.head()
```

    (315, 7)
    Index(['0', '1', '2', '3', '4', '5', '6'], dtype='object')
    




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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bread</td>
      <td>Wine</td>
      <td>Eggs</td>
      <td>Meat</td>
      <td>Cheese</td>
      <td>Pencil</td>
      <td>Diaper</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bread</td>
      <td>Cheese</td>
      <td>Meat</td>
      <td>Diaper</td>
      <td>Wine</td>
      <td>Milk</td>
      <td>Pencil</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cheese</td>
      <td>Meat</td>
      <td>Eggs</td>
      <td>Milk</td>
      <td>Wine</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cheese</td>
      <td>Meat</td>
      <td>Eggs</td>
      <td>Milk</td>
      <td>Wine</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meat</td>
      <td>Pencil</td>
      <td>Wine</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_columns = mdf.columns.str.strip().to_list()
mdf.columns = new_columns
mdf.head(2)
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
      <th>Hair Conditioner</th>
      <th>Lemons</th>
      <th>Standard coffee</th>
      <th>Frozen Chicken Wings</th>
      <th>98pct. Fat Free Hamburger</th>
      <th>Sugar Cookies</th>
      <th>Onions</th>
      <th>Deli Ham</th>
      <th>Dishwasher Detergent</th>
      <th>Beets</th>
      <th>...</th>
      <th>Lollipops</th>
      <th>Plain White Bread</th>
      <th>Blueberry Yogurt</th>
      <th>Frozen Chicken Thighs</th>
      <th>Mixed Vegetables</th>
      <th>Souring Pads</th>
      <th>Tuna Spread</th>
      <th>Toilet Paper</th>
      <th>White Wine</th>
      <th>Columbian Coffee</th>
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
  </tbody>
</table>
<p>2 rows × 255 columns</p>
</div>



### mlxtend 활용


```python
from tqdm import tqdm
tqdm.pandas()
```


```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules,fpgrowth
```

#### TransactionEncoder  
- 연관성분석 하기 좋은 데이터셋으로, 변환시켜주는 라이브러리
 > cust_id,[item01,item02,item03 ~ item100] 처럼 만들어준다.

본 데이터셋에서는 이미, 변환이 되어 있기 때문에 필요없다. [2차원 list or array 를 형태를 받아서, 변환시킨다.]  
> te = TransactionEncoder() 예시  
> te_rslt = te.fit(mdf_lst).transform(mdf_lst) 예시



```python
mdf.shape
```




    (1361, 255)




```python
item_set = fpgrowth(mdf,min_support=0.05,use_colnames=True)
```


```python
item_set
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
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.055841</td>
      <td>( Plums)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.057311</td>
      <td>( Pancake Mix)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.109478</td>
      <td>( 2pct. Milk)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.119030</td>
      <td>( White Bread)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.097722</td>
      <td>( Potato Chips)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.093314</td>
      <td>( 98pct. Fat Free Hamburger)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.074210</td>
      <td>( Toilet Paper)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.080088</td>
      <td>( Onions)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.071271</td>
      <td>( Hamburger Buns)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.058780</td>
      <td>( French Fries)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.055107</td>
      <td>( Sugar Cookies)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.092579</td>
      <td>( Hot Dogs)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.067597</td>
      <td>( Domestic Beer)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.063924</td>
      <td>( Popcorn Salt)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.058780</td>
      <td>( Hair Conditioner)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.051433</td>
      <td>( Waffles)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.122704</td>
      <td>( Eggs)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.085231</td>
      <td>( Sweet Relish)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.079353</td>
      <td>( Toothpaste)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.066128</td>
      <td>( Tomatoes)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.054372</td>
      <td>( Canned Tuna)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.053637</td>
      <td>( Apples)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.050698</td>
      <td>( Sour Cream)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.077884</td>
      <td>( Cola)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.069067</td>
      <td>( Pepperoni Pizza - Frozen)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.062454</td>
      <td>( Ramen Noodles)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.058780</td>
      <td>( Hot Dog Buns)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.057311</td>
      <td>( Garlic)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.077149</td>
      <td>( Wheat Bread)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.064658</td>
      <td>( Bologna)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.059515</td>
      <td>( Bananas)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.055841</td>
      <td>( Frozen Shrimp)</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.054372</td>
      <td>( Sandwich Bags)</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.060985</td>
      <td>( Raisins)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.056576</td>
      <td>( Orange Juice)</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.052902</td>
      <td>( C Cell Batteries)</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.055841</td>
      <td>( Oranges)</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.054372</td>
      <td>( Mushrooms)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.052168</td>
      <td>( Eggs,  2pct. Milk)</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.051433</td>
      <td>( 2pct. Milk,  White Bread)</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.055107</td>
      <td>( Eggs,  White Bread)</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.051433</td>
      <td>( White Bread,  Potato Chips)</td>
    </tr>
  </tbody>
</table>
</div>




```python
def calculate_length(df,idx):
    for i in df.index:
        df.at[i,'length_consequent'] = int(len(list(df.iloc[i,idx])))
    return df
```


```python
rule_rslt_lift = association_rules(item_set, metric="lift", min_threshold=3.0) ## 0.53 / 5.6
```


```python
rule_rslt_lift
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>( Eggs)</td>
      <td>( 2pct. Milk)</td>
      <td>0.122704</td>
      <td>0.109478</td>
      <td>0.052168</td>
      <td>0.425150</td>
      <td>3.883414</td>
      <td>0.038734</td>
      <td>1.549137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>( 2pct. Milk)</td>
      <td>( Eggs)</td>
      <td>0.109478</td>
      <td>0.122704</td>
      <td>0.052168</td>
      <td>0.476510</td>
      <td>3.883414</td>
      <td>0.038734</td>
      <td>1.675861</td>
    </tr>
    <tr>
      <th>2</th>
      <td>( 2pct. Milk)</td>
      <td>( White Bread)</td>
      <td>0.109478</td>
      <td>0.119030</td>
      <td>0.051433</td>
      <td>0.469799</td>
      <td>3.946889</td>
      <td>0.038402</td>
      <td>1.661576</td>
    </tr>
    <tr>
      <th>3</th>
      <td>( White Bread)</td>
      <td>( 2pct. Milk)</td>
      <td>0.119030</td>
      <td>0.109478</td>
      <td>0.051433</td>
      <td>0.432099</td>
      <td>3.946889</td>
      <td>0.038402</td>
      <td>1.568093</td>
    </tr>
    <tr>
      <th>4</th>
      <td>( Eggs)</td>
      <td>( White Bread)</td>
      <td>0.122704</td>
      <td>0.119030</td>
      <td>0.055107</td>
      <td>0.449102</td>
      <td>3.773010</td>
      <td>0.040501</td>
      <td>1.599152</td>
    </tr>
    <tr>
      <th>5</th>
      <td>( White Bread)</td>
      <td>( Eggs)</td>
      <td>0.119030</td>
      <td>0.122704</td>
      <td>0.055107</td>
      <td>0.462963</td>
      <td>3.773010</td>
      <td>0.040501</td>
      <td>1.633586</td>
    </tr>
    <tr>
      <th>6</th>
      <td>( White Bread)</td>
      <td>( Potato Chips)</td>
      <td>0.119030</td>
      <td>0.097722</td>
      <td>0.051433</td>
      <td>0.432099</td>
      <td>4.421702</td>
      <td>0.039801</td>
      <td>1.588793</td>
    </tr>
    <tr>
      <th>7</th>
      <td>( Potato Chips)</td>
      <td>( White Bread)</td>
      <td>0.097722</td>
      <td>0.119030</td>
      <td>0.051433</td>
      <td>0.526316</td>
      <td>4.421702</td>
      <td>0.039801</td>
      <td>1.859825</td>
    </tr>
  </tbody>
</table>
</div>




```python
rule_rslt_lift.index
```




    RangeIndex(start=0, stop=8, step=1)




```python
rule_rslt_lift01 = calculate_length(rule_rslt_lift,1)
```


```python
rule_rslt_lift01
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>length_consequent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>( Eggs)</td>
      <td>( 2pct. Milk)</td>
      <td>0.122704</td>
      <td>0.109478</td>
      <td>0.052168</td>
      <td>0.425150</td>
      <td>3.883414</td>
      <td>0.038734</td>
      <td>1.549137</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>( 2pct. Milk)</td>
      <td>( Eggs)</td>
      <td>0.109478</td>
      <td>0.122704</td>
      <td>0.052168</td>
      <td>0.476510</td>
      <td>3.883414</td>
      <td>0.038734</td>
      <td>1.675861</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>( 2pct. Milk)</td>
      <td>( White Bread)</td>
      <td>0.109478</td>
      <td>0.119030</td>
      <td>0.051433</td>
      <td>0.469799</td>
      <td>3.946889</td>
      <td>0.038402</td>
      <td>1.661576</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>( White Bread)</td>
      <td>( 2pct. Milk)</td>
      <td>0.119030</td>
      <td>0.109478</td>
      <td>0.051433</td>
      <td>0.432099</td>
      <td>3.946889</td>
      <td>0.038402</td>
      <td>1.568093</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>( Eggs)</td>
      <td>( White Bread)</td>
      <td>0.122704</td>
      <td>0.119030</td>
      <td>0.055107</td>
      <td>0.449102</td>
      <td>3.773010</td>
      <td>0.040501</td>
      <td>1.599152</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>( White Bread)</td>
      <td>( Eggs)</td>
      <td>0.119030</td>
      <td>0.122704</td>
      <td>0.055107</td>
      <td>0.462963</td>
      <td>3.773010</td>
      <td>0.040501</td>
      <td>1.633586</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>( White Bread)</td>
      <td>( Potato Chips)</td>
      <td>0.119030</td>
      <td>0.097722</td>
      <td>0.051433</td>
      <td>0.432099</td>
      <td>4.421702</td>
      <td>0.039801</td>
      <td>1.588793</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>( Potato Chips)</td>
      <td>( White Bread)</td>
      <td>0.097722</td>
      <td>0.119030</td>
      <td>0.051433</td>
      <td>0.526316</td>
      <td>4.421702</td>
      <td>0.039801</td>
      <td>1.859825</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
