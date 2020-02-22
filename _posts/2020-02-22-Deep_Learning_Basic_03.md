---
title:  "Neuralnet Basic 03"
excerpt: "Basic Neural Net using numpy,tensor-flow,keras"

categories:
  - Deep-Learning
tags:
  - DL
  - Neuralnet Using keras
  - KEARS 창시자에게 배우는 딥러닝
  - 딥러닝
last_modified_at: 2020-02-22T13:06:00-05:00
---

KEARS 창시자에게 배우는 딥러닝 - 3장 -01 이진분류문제
> Basic 2 에서, Mnist를 했다면, 여기서는 간단한 영화리뷰에 대해서, binaray classfication을 수행한다.
> [책 관련 Blog 로 이동](https://tensorflow.blog/%EC%BC%80%EB%9D%BC%EC%8A%A4-%EC%B0%BD%EC%8B%9C%EC%9E%90%EC%97%90%EA%B2%8C-%EB%B0%B0%EC%9A%B0%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D/)


```python
import tensorflow as tf
from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
```


```python
import keras
keras.__version__
```




    '2.2.4'




```python
from keras.datasets import imdb
## num_words=10000 는 훈련데이터에서, 자주 사용하는 단어 1만개만 사용하겠다는 의미임
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

### --------------------------------**데이터 내용 파악 start**--------------------------------------------------


```python
print("train_data.shape",train_data.shape)
print("train_labels.shape",train_labels.shape)
print("test_data.shape",test_data.shape)
print("test_labels",test_labels.shape)
```

    train_data.shape (25000,)
    train_labels.shape (25000,)
    test_data.shape (25000,)
    test_labels (25000,)
    


```python
print(len(train_data[25]))
train_data[25][0:10] ## 25번째 데이터의 구성을 보면, 총 142개의 단어로 되어 있고, 0~9번째까지의 단어는 하기와 같다.
```

    142
    




    [1, 14, 9, 6, 55, 641, 2854, 212, 44, 6]




```python
max([max(sequence) for sequence in train_data]) ## num_words=10000 제한이 없었으면, 88586 단어의 데이터가 존재한다
```




    9999




```python
for line_idx in range(0,len(train_data)):
    if (line_idx%5000)==0:
        print(len(train_data[line_idx]))
## 보시다시피, 각 train_data 라인당 모두 길이가 다르다
```

    218
    124
    118
    281
    252
    

원래의 영어단어로 바꾸기


```python
# word_index는 단어와 정수 인덱스를 매핑한 딕셔너리입니다
word_index = imdb.get_word_index()
```


```python
print(type(word_index),len(word_index))
word_index['faw'] ## faw 란 단어는 이 train 셋의 단어사전에서, index 번호가 88584 이다.
```

    <class 'dict'> 88584
    




    84994




```python
# 정수 인덱스와 단어를 매핑하도록 뒤집습니다
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
reverse_word_index[1]
```




    'the'



dictinary 깨알상식
dict.get(key, default = None)  
>Parameters  
>key − This is the Key to be searched in the dictionary.  
>default − This is the Value to be returned in case key does not exist.


```python
# 리뷰를 디코딩합니다. 
# 0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺍니다 (그렇게 구성된듯...내가 한게 아니니 원..어떤의미인줄은 알겠다.)
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[25]])
```


```python
decoded_review
```




    '? this is a very light headed comedy about a wonderful family that has a son called pecker because he use to peck at his food pecker loves to take all kinds of pictures of the people in a small ? of ? ? and manages to get the attention of a group of photo art lovers from new york city pecker has a cute sister who goes simply nuts over sugar and is actually an addict taking ? of sugar from a bag there are scenes of men showing off the ? in their ? with ? movements and ? doing pretty much the same it is rather hard to keep your mind out of the ? with this film but who cares it is only a film to give you a few laughs at a simple picture made in 1998'



### --------------------------------**데이터 내용 파악 end**--------------------------------------------------

신경망에는 list를 input data로 활용할수없기때문에, vector 로 바꾼다. 
이때, Embedding 이나, one-hotencoding을 사용하는데, 여기서는 one-hotencoding을 활용한다.


```python
print("train_data.shape",train_data.shape)
print("test_data.shape",test_data.shape)
```

    train_data.shape (25000,)
    test_data.shape (25000,)
    


```python
import numpy as np

def vectorize_sequences(sequences, dimension=10000): ## 처음 데이터를 불러올때부터, 높은 빈도수 10000 개로 제한했기때문
    # 크기가 (len(sequences), dimension))이고 모든 원소가 0인 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
#         if i==100:
#             print(i,type(sequence))
        results[i, sequence] = 1.  # results[i]에서 특정 인덱스의 위치를 1로 만듭니다. 여기서 sequence 는 list type 이다. 
        ## 즉 1개 행에서, 존재하는 모든 단어들을 그 위치에 1로서 원핫인코딩 때린다.
    return results

# 훈련 데이터를 벡터로 변환합니다
x_train = vectorize_sequences(train_data)
# 테스트 데이터를 벡터로 변환합니다
x_test = vectorize_sequences(test_data)
```


```python
print("after one-hot x_train.shape",x_train.shape)
print("after one-hot x_test.shape",x_test.shape)
```

    after one-hot x_train.shape (25000, 10000)
    after one-hot x_test.shape (25000, 10000)
    


```python
print(type(train_labels))
```

    <class 'numpy.ndarray'>
    


```python
# 레이블을 벡터로 바꿉니다
## np.asarray 는 numpy 객체를 copy 생성하면서,  numpy 형태로 변형한다.
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
```


```python
print(train_labels[0])
print(y_train[0])
```

    1
    1.0
    

신경망 모형

![3-layer network](https://s3.amazonaws.com/book.keras.io/img/ch3/3_layer_network.png)


```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='sigmoid', input_shape=(10000,))) ## sigmoid, tahn
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
```


```python
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
## 상기처럼 해도 되지만, 객체를 넣어서 불러와도 된다.
# from keras import losses
# from keras import metrics

# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])
```


```python
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```


```python
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 2s 120us/step - loss: 0.6833 - acc: 0.5481 - val_loss: 0.6590 - val_acc: 0.5768
    Epoch 2/20
    15000/15000 [==============================] - 2s 101us/step - loss: 0.6276 - acc: 0.7648 - val_loss: 0.6023 - val_acc: 0.8385
    Epoch 3/20
    15000/15000 [==============================] - 1s 98us/step - loss: 0.5638 - acc: 0.8631 - val_loss: 0.5422 - val_acc: 0.8443
    Epoch 4/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.4962 - acc: 0.8825 - val_loss: 0.4812 - val_acc: 0.8655
    Epoch 5/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.4315 - acc: 0.8946 - val_loss: 0.4274 - val_acc: 0.8722
    Epoch 6/20
    15000/15000 [==============================] - 2s 101us/step - loss: 0.3727 - acc: 0.9060 - val_loss: 0.3811 - val_acc: 0.8793
    Epoch 7/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.3216 - acc: 0.9170 - val_loss: 0.3453 - val_acc: 0.8831
    Epoch 8/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.2793 - acc: 0.9252 - val_loss: 0.3168 - val_acc: 0.8870
    Epoch 9/20
    15000/15000 [==============================] - 2s 101us/step - loss: 0.2443 - acc: 0.9332 - val_loss: 0.2974 - val_acc: 0.8900
    Epoch 10/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.2152 - acc: 0.9393 - val_loss: 0.2841 - val_acc: 0.8906
    Epoch 11/20
    15000/15000 [==============================] - 1s 100us/step - loss: 0.1910 - acc: 0.9457 - val_loss: 0.2746 - val_acc: 0.8918
    Epoch 12/20
    15000/15000 [==============================] - 2s 101us/step - loss: 0.1702 - acc: 0.9515 - val_loss: 0.2729 - val_acc: 0.8912
    Epoch 13/20
    15000/15000 [==============================] - 1s 100us/step - loss: 0.1527 - acc: 0.9555 - val_loss: 0.2694 - val_acc: 0.8906
    Epoch 14/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.1372 - acc: 0.9611 - val_loss: 0.2716 - val_acc: 0.8900
    Epoch 15/20
    15000/15000 [==============================] - 1s 100us/step - loss: 0.1243 - acc: 0.9651 - val_loss: 0.2756 - val_acc: 0.8892
    Epoch 16/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.1124 - acc: 0.9701 - val_loss: 0.2841 - val_acc: 0.8886
    Epoch 17/20
    15000/15000 [==============================] - 2s 101us/step - loss: 0.1020 - acc: 0.9740 - val_loss: 0.2877 - val_acc: 0.8881
    Epoch 18/20
    15000/15000 [==============================] - 1s 100us/step - loss: 0.0923 - acc: 0.9768 - val_loss: 0.2963 - val_acc: 0.8870
    Epoch 19/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.0841 - acc: 0.9798 - val_loss: 0.3054 - val_acc: 0.8866
    Epoch 20/20
    15000/15000 [==============================] - 1s 99us/step - loss: 0.0764 - acc: 0.9822 - val_loss: 0.3148 - val_acc: 0.8853
    


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])




```python
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# ‘bo’는 파란색 점을 의미합니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# ‘b’는 파란색 실선을 의미합니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```


![png](/assets/images/output_32_0.png))



```python
plt.clf()   # 그래프를 초기화합니다
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```


![png](/assets/images/output_33_0.png)


위 표 결과를 볼대, 과적합을 피하기 위해, epoch=4 에서부터, 적당히 잘라주면 된다. 


```python
model.fit(x_train, y_train, epochs=5, batch_size=512)
results = model.evaluate(x_test, y_test)
```

    Epoch 1/5
    25000/25000 [==============================] - 2s 62us/step - loss: 0.0906 - acc: 0.9757
    Epoch 2/5
    25000/25000 [==============================] - 2s 61us/step - loss: 0.0850 - acc: 0.9784
    Epoch 3/5
    25000/25000 [==============================] - 2s 63us/step - loss: 0.0802 - acc: 0.9797
    Epoch 4/5
    25000/25000 [==============================] - 2s 61us/step - loss: 0.0751 - acc: 0.9818
    Epoch 5/5
    25000/25000 [==============================] - 2s 61us/step - loss: 0.0706 - acc: 0.9836
    25000/25000 [==============================] - 2s 82us/step
    


```python
results ## relu -> sigmoid 가 약간 더 acc 가 더 높게 나옴
```




    [0.4458531785559654, 0.8628]




```python
y_proba = model.predict(x_test)
```


```python
len(y_proba) ## test set의 결과표
```




    25000




```python

```
