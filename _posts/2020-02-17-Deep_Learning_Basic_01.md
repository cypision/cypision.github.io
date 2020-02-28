---
title:  'Neuralnet Basic 01'
excerpt: 'Basic Neural Net using numpy,tensor-flow,keras'

categories:
  - Deep-Learning
tags:
  - DL
  - Neuralnet Using Numpy
  - 밑바닥부터 시작하는 딥러닝
  - 딥러닝
last_modified_at: 2020-02-16T08:06:00-05:00
---

##### MNIST 데이터를 활용하여, 기본 뉴럴넷을 구성해보자
> minst 데이터는 기본 keras에 있는 걸 활용한다.


```python
from collections import OrderedDict
import numpy as np
import keras
keras.__version__
```

    Using TensorFlow backend.





    '2.3.1'




```python
from keras.datasets import mnist
(train_images,train_labels),(test_images, test_labels) = mnist.load_data()
```


```python
print("train_images.shape",train_images.shape)
print("train_labels.shape",train_labels.shape)
print("test_images.shape",test_images.shape)
print("test_labels",test_labels.shape)
```

    train_images.shape (60000, 28, 28)
    train_labels.shape (60000,)
    test_images.shape (10000, 28, 28)
    test_labels (10000,)



```python
train_images = train_images.reshape(60000,28*28)
test_images = test_images.reshape(10000,28*28)
```

#### 본 실습은 "밑바닥부터 시작하는 Deep Learing" = "Deep Learning From Scratch"를 활용하여, DL 기초부분을 설명함  
+ 여기서는 역전파에 대해서, 수치미분 과 계산그래프의 개념을 통해서, 설명하는데, 주로 "계산그래프" 방식을 통해서, 딥러닝의 역전파 핵심을 설명한다. 이 부분은 별도로 공부하거나, 문의주시길.
> 수치미분 : Lim f(x+h)/f(x) 극한의 개념을 가져갈때, 애기한 미분값. 가장 일반적인 설명에 사용함  
> 계산그래프 : Andrej Karpathy 블로그,  Fei-Fei Le (stanford 교수) 의 아이디어임

##### TwoLayerNet class 를 만들어서, 진행함
+ 내부에서 활용하기 위해 필요한 class 와 def 정의함

**Relu Class**
> Activation 함수로 Relu를 사용


```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```

**Affine Class**
> WX+B 선현결합과 activation 함수의 결합한 1개의 Node를 말한다.
> 여기서 사용된 코드들은 계산그래프 내용과 닿아있으며, 여기서는 설명을 생략한다. 


```python
class Affine:
    def __init__(self, W, b):
        self.W = W ## 해당 Node의 가중치 값을 자체 클래스 내부에 저장하기 위한것 - forward
        self.b = b ## 해당 노드의 bias 값을 자체 클래스 내부에 저장하기 위한것  - forward
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None ## 역전파시 사용되는 미분값을 저장하기 위함. Node별 가중치 값이 1번 계산되고 저장되기에, 이 값만 알면, 100,1000,10만 번이든 무관하게 간단한 연산만으로 속도를 향상시킨다
        self.db = None ## 이 점이 바로 "오차역전파법" 의 내용과 연관되어있다.
        
    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x ## 나중에 역전파일때, 사용하기 위해 노드의 변수로 저장해둔다. - backward
        out = np.dot(self.x, self.W) + self.b ## 선형결합 결과값을 넘김 이 다음에 activation 함수가 이 결과값을 인자로 받는다.
                                              ## 다른 책에서는 선형결합+activation 을 한개로 묶어서 설명하기도 한다.
        return out
    
    ## network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    ## affine layer는 2개임. 1layer:50,2layer:10
    def backward(self, dout):
        dx = np.dot(dout, self.W.T) ## dout는 최종노드에서부터 오는 값. 
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
```

##### **SoftmaxWithLoss Class**
> 마지막 layer 로서, 소프트맥스 함수와 Loss 함수인 Cross Entropy 노드를 담고 있다.
> 통상 Train 단계가 아닌 추론 단계에서는 Sofrmax With Loss 노드를 생략하기도 한다.
>> 왜냐하면, ohe-hot 인코딩인드 아니든 y_pred 값이 (10 class 분류문제일 경우) 큰 값만을 가져서와 계산을 해도 문제가 없기 때문


```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size ## 해석적 미분을 하면, 소프트맥스함수와 크로스엔트로피 결합은 (self.y - self.t) 와 같다.
        else:
            dx = self.y.copy()             ##일단 softmax 값으로 나온 예측값 y를 copy하고, y의 shape는 (batch_size,10) <minist class가 10개고, X 데이터 shape[1] 이 10 one-hot이기때문
            dx[np.arange(batch_size), self.t] -= 1 ## 해당 예측값을 찾아서 1을 빼줌으로서  (self.y - self.t) 와 같게만듬. -1 에서 1은 정답레이블을 그냥 원-핫 취급하기 위해 빼준는것
            dx = dx / batch_size            ## 예측값이 원-핫 형태이니, y-t 에서 t 도 원-핫 처럼 다뤄주기 위해서임
        return dx
            
    def cross_entropy_error(self,y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    
    def softmax(self,x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x) # 오버플로 대책
        return np.exp(x) / np.sum(np.exp(x))
```


```python
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        np.random.seed(42) ## 확인용
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size,)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        
        ##갱신여부 알고리즘 start check##
        aa = self.params['W1'].copy()
        bb = self.params['b1'].copy()
        cc = self.params['W2'].copy()
        dd = self.params['b2'].copy()
        
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
#         self.layers['Affine1'] = Affine(aa, bb)
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
#         self.layers['Affine2'] = Affine(cc, dd)
        
        self.lastLayer = SoftmaxWithLoss()
        print("create TwoLayerNet")
        
        ##갱신여부 알고리즘 End check##
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
#         print("loss function go")
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
#     def numerical_gradient(self, x, t):
#         loss_W = lambda W: self.loss(x, t)
        
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
#         return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
#         print("Affine1_dW", np.sum(np.sum(self.layers["Affine1"].dW)))
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))    

    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        
        x = x - np.max(x) # 오버플로 대책
        return np.exp(x) / np.sum(np.exp(x))
```

##### 실제 TwoLayerNet 를 활용한 Mnist data 학습


```python
# train_images , test_images , train_labels , test_labels
```


```python
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
```

    create TwoLayerNet



```python
iters_num = 6000
train_size = train_images.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
print("iter_per_epoch",iter_per_epoch)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
#     batch_mask = range(batch_size) 고정
    x_batch = train_images[batch_mask]
    t_batch = train_labels[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    ##갱신여부 start check##
#     p1=network.params['W1'].sum()
#     p2=network.params['b1'].sum()
#     print("network_param",p1,p2)
#     tw1 = network.layers['Affine1'].W.sum()
#     tb1 = network.layers['Affine1'].b.sum()
#     print("before modify",tw1,tb1)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
#     tw2 = network.layers['Affine1'].W.sum()
#     tb2 = network.layers['Affine1'].b.sum()
#     print("after modify",tw2,tb2,"\n")
    ##갱신여부 end check##
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(train_images, train_labels)
        test_acc = network.accuracy(test_images, test_labels)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```

    iter_per_epoch 600.0
    0.10441666666666667 0.1028
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136
    0.11238333333333334 0.1136



```python
print(len(train_loss_list),len(train_acc_list),len(test_acc_list))
```

    6000 10 10


상기 정확도가 높지않은 이유는 몇가지 보정처리를 추가적으로 해줘야 하기 때문
상세내용은 다음 기회에 다룬다.
혹 W,b 의 갱신이 이루어지지 않는 것이 아닐까 추정했지만, 결론적으론 그건 아닌걸로.

#### 가중치, bias 갱신 부분 코드 체크
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]  
        
상기 부분은 TwoLayerNet 클래스의 params dict 객체안의 값을 갱신시킨다. 그러나, 실제로 갱신되어야 하는 값은, TwoLayerNet - layers(OrderDict 객체) - "Affine1","Relu","Affine2" - W,b 값들이다.  
헌데, 확인해보니 가중치,bias갱신은 일어난다. 어찌된일일까?  

비밀은 __call_by_reference__ 에 있다 (필자추정)  
TwoLayerNet 에서, layers(OrderDict 객체) 의 value로 "Affine calss" 를 최초에 생성하는데, 이때의 구문은  
##### 
    self.layers = OrderedDict()  
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])  
    self.layers['Relu1'] = Relu()  
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])  
    self.lastLayer = SoftmaxWithLoss()  
여기서, **Affine(self.params['W1'], self.params['b1'])** 을 보면, 내부 TwoLayerNet 클래스의 params 변수로 생성했다. 이때, self.params['b1'] 와 같은 값들은, 메모리에 값(1,2같은)이 들어가는게 아니라  
"주소" 값이 들어간다. 그렇게 때문에, self.params 의 값이 갱신되면, 그걸 바라보고 있는 Affine 클래스의 W,b (="Affine1","Relu","Affine2" - W,b)들도 같이 변하게 되는 것이다.



```python

```
