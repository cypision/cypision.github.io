---
title:  "Useful_Function 01"
excerpt: "numpy concatenate"

categories:
  - Function
tags:
  - Numpy
  - concatenate
  - hstack,vstack
  - column_stack
last_modified_at: 2020-03-04T23:06:00-05:00
---

Numpy libraray 에서, 배열을 붙이는 거에 대한 설명을 다루어 본다.  
1. 두 배열을 왼쪽에서 오른쪽으로 붙이기 
 >  numpy.r_[a, b]  
 >  numpy.hstack([a, b])  
 >  numpy.concatenate((a, b), axis = 0)
2. 두 배열을 위에서 아래로 붙이기

3. 두 개의 1차원 배열을 칼럼으로 세로로 붙여서 2차원 배열 만들기

출처: https://rfriend.tistory.com/352 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]

## numpy.r_

[공식 scipy 설명page](https://docs.scipy.org/doc/numpy/reference/generated/numpy.r_.html)


```python
import numpy as np
```


```python
ar = np.array([1,2,3])
br = np.array([4,5,6])
cr = np.zeros((2,3))

np.r_[ar, 0, 0, br]
```




    array([1, 2, 3, 0, 0, 4, 5, 6])



np.r_ : 배열을 붙이는 역할을 하는데, 같은 ndim 을 가진 배열을 붙인다.  
np.r_[ar, 0, 0, br,cr] 하면 cr 의 rank (ndim=2) 이기 때문에 바로 에러난다.


```python
print(cr.shape,np.ones((2,3)).shape)
rslt = np.r_[cr,np.ones((2,3))]
print(rslt.shape,'\n',rslt)
```

    (2, 3) (2, 3)
    (4, 3) 
     [[0. 0. 0.]
     [0. 0. 0.]
     [1. 1. 1.]
     [1. 1. 1.]]
    


```python
print(ar.ndim,br.ndim,cr.ndim)
```

    1 1 2
    


```python
np.r_[[ar], [br]] # array 에 [] 를 씌우면서 [ar]은 list type 이 되었고, 이를 2줄로 axis=0 으로 결합하고, array 로 return 한것
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
np.r_[-1:10:1, [77]*3, 5, 6]
```




    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 77, 77, 77,  5,  6])




```python
a = np.array([[0, 1, 2], [3, 4, 5]])
print(np.r_['0',a, a]) ## 0일때는 ndim=0 기준으로 default. 결과값의 ndim 은 변화없음  
print(np.r_['1',a, a]) ## 1일때는 ndim=1 기준으로 default. 결과값의 ndim 은 변화없음 ndim=1 인 배열이라면, 축이 0번째 밖에 없기 때문에 에러남
```

    [[0 1 2]
     [3 4 5]
     [0 1 2]
     [3 4 5]]
    [[0 1 2 0 1 2]
     [3 4 5 3 4 5]]
    

print(np.r_['2',a, a]) --> error


```python
a3 = np.array([[[0, 1, 2], [3, 4, 5]],[[10, 11, 12], [13, 14, 15]]])
# print(np.r_['0',a3, a3]) ## 0일때는 ndim=0 기준으로 default. 결과값의 ndim 은 변화없음  
print(np.r_['2',a3, a3])
```

    [[[ 0  1  2  0  1  2]
      [ 3  4  5  3  4  5]]
    
     [[10 11 12 10 11 12]
      [13 14 15 13 14 15]]]
    


```python
np.r_['1',ar, br]
```


    ---------------------------------------------------------------------------

    AxisError                                 Traceback (most recent call last)

    <ipython-input-9-55dc571915a4> in <module>
    ----> 1 np.r_['1',ar, br]
    

    C:\ProgramData\Anaconda3\envs\test\lib\site-packages\numpy\lib\index_tricks.py in __getitem__(self, key)
        402                 objs[k] = objs[k].astype(final_dtype)
        403 
    --> 404         res = self.concatenate(tuple(objs), axis=axis)
        405 
        406         if matrix:
    

    AxisError: axis 1 is out of bounds for array of dimension 1



```python
np.r_['0,2', ar, br] ## 콤마 뒷자리로, 배열끼리 합쳤을때, 원하는 차원을 구현할 수 있다.
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
np.r_['0,2', ar, br]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
np.r_['0,2,0', ar, br]
```




    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])




```python
np.r_['1,2,0', ar, br]
```




    array([[1, 4],
           [2, 5],
           [3, 6]])



__생각보다 활용이 다양한다.__

## numpy.c_

[공식 scipy 페이지 설명](https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html)

np.r_['1,2,0', ar, br] 으로 사용하는것과 동일하다. np.r_ 이 사용하기 복잡해서 좀더 유용하게 등장한것


```python
np.c_[ar,br]
```




    array([[1, 4],
           [2, 5],
           [3, 6]])




```python
print([ar],type([ar]))
np.c_[[ar],[br]]
```

    [array([1, 2, 3])] <class 'list'>
    




    array([[1, 2, 3, 4, 5, 6]])




```python
np.array([[1,2,3]]).ndim
```




    2




```python
np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
```




    array([[1, 2, 3, 0, 0, 4, 5, 6]])




```python

```
