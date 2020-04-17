---
title:  "Useful_Function 03"
excerpt: "numpy stack"

categories:
  - Function
tags:
  - Numpy
  - concatenate
  - hstack,vstack    
last_modified_at: 2020-04-17T23:06:00-05:00
---


```python
import numpy as np
```


```python
a = np.array((1,2,3))
b = np.array((2,3,4))
print("a.ndim:{}".format(a.ndim),"b.ndim:{}".format(b.ndim))
print("a.shape:{}".format(a.shape),"b.shape:{}".format(b.shape))
```

    a.ndim:1 b.ndim:1
    a.shape:(3,) b.shape:(3,)
    


```python
print(np.hstack((a,b)).shape)
np.hstack((a,b))
```

    (6,)
    




    array([1, 2, 3, 2, 3, 4])




```python
print(np.vstack((a,b)).shape)
np.vstack((a,b))
```

    (2, 3)
    




    array([[1, 2, 3],
           [2, 3, 4]])




```python
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
print("a.ndim:{}".format(a.ndim),"b.ndim:{}".format(b.ndim))
print("a.shape:{}".format(a.shape),"b.shape:{}".format(b.shape))
```

    a.ndim:2 b.ndim:2
    a.shape:(3, 1) b.shape:(3, 1)
    


```python
print(np.hstack((a,b)).shape)
np.hstack((a,b))
```

    (3, 2)
    




    array([[1, 2],
           [2, 3],
           [3, 4]])




```python
print(np.vstack((a,b)).shape)
np.vstack((a,b))
```

    (6, 1)
    




    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])



`hstack` : 경우는 horizon 으로 옆으로 붙이는 느낌. return 값은 제약이 없다. axis=1 자리의 val 가 늘어나도록 붙인다.  
> (3,) 2개를 붙이면, (6,) 이 된다.
> (3,1) 2개를 hstack 하면, (3,2) 가 된다.  

`vstack` : return 에서, 최소 2-D 이상의 array 가 된다.  
> (3,) 2개를 붙이면, (2,3) 이 된다.
> (3,1) 2개를 vstack 하면, (6,1) 가 된다.  


```python

```
