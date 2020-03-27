---
title:  "혼자하는 Text분석_05"
excerpt: "Keras에서 Embedding 이후, RNN적용한 Text 분석 비교"

categories:
  - Machine-Learning
tags:
  - Keras
  - text anlysis
  - 머신러닝
  - keras tokenizer
  - minsuk-heo youtube  
last_modified_at: 2020-03-27T20:13:00-05:00
---

본 posting 내용은 필자가 존경하는 minsuk-heo 님의 youtube 강의노트를 사용해서, Embedding 에 대한 나름의 정리를 하고자 한다.  
(https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb)

# Word2Vec
here I implement word2vec with very simple example using tensorflow  
word2vec is vector representation for words with similarity

# Collect Data
we will use only 10 sentences to create word vectors


```python
corpus = ['king is a strong man', 
          'queen is a wise woman', 
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong', 
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']
```

쓸데없는 단어 제거...관용사 a 같은거..


```python
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    
    return results
```


```python
corpus = remove_stop_words(corpus)
```


```python
## corpus 를 이후 unique 하게 정리해서, corpus 의 어휘사전을 만들면,
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)
```


```python
words
```




    {'boy',
     'girl',
     'king',
     'man',
     'pretty',
     'prince',
     'princess',
     'queen',
     'strong',
     'wise',
     'woman',
     'young'}



# data generation
we will generate label for each word using skip gram. - wor2Vec 의 알고리즘 중 하나  
※ wor2vec : Embedding 기법의 한 종류이고, target 을 인접한 neighbor 단어들로 잡는다.


```python
word2int = {}
## 단어 사전 만들기, 단어에 정수 sequence 를 부여한다.
for i,word in enumerate(words):
    word2int[word] = i
```


```python
sentences = []
for sentence in corpus: ## data의 1개의 smaple 씩, 그니깐, 1개의 문장씩 sentences 란 새로운 list 에 단어별로 list를 만든다.
    sentences.append(sentence.split())
```


```python
cnt = 0 
for sentence in sentences:
    cnt +=1
    print("{}번째 문장 value:{} 그리고 총 length:{}".format(cnt,sentence,len(sentence)))
```

    1번째 문장 value:['king', 'strong', 'man'] 그리고 총 length:3
    2번째 문장 value:['queen', 'wise', 'woman'] 그리고 총 length:3
    3번째 문장 value:['boy', 'young', 'man'] 그리고 총 length:3
    4번째 문장 value:['girl', 'young', 'woman'] 그리고 총 length:3
    5번째 문장 value:['prince', 'young', 'king'] 그리고 총 length:3
    6번째 문장 value:['princess', 'young', 'queen'] 그리고 총 length:3
    7번째 문장 value:['man', 'strong'] 그리고 총 length:2
    8번째 문장 value:['woman', 'pretty'] 그리고 총 length:2
    9번째 문장 value:['prince', 'boy', 'king'] 그리고 총 length:3
    10번째 문장 value:['princess', 'girl', 'queen'] 그리고 총 length:3
    

위 결과처럼, 각 문장별로 길이가 다르다...keras 로 치면, max_len 에 해당하는 값이다.


```python
WINDOW_SIZE = 2
data = []

for sentence in sentences:
    for idx, word in enumerate(sentence): # 개별 sample(문장별) 
         #각 해당단어에서, 2개 씩 좌우로 후보 단어더미를 list 로 만들고 차례대로 neighbor 로 추출
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] :
            if neighbor != word:
                data.append([word, neighbor]) # 해당단어만 빼고, 이웃되는 (좌 0~2개,우 0~2개 총 4개이하) 로 꾸러미를 만든다
```


```python
import pandas as pd
for text in corpus:
    print(text)

df = pd.DataFrame(data, columns = ['input', 'label'])
```

    king strong man
    queen wise woman
    boy young man
    girl young woman
    prince young king
    princess young queen
    man strong
    woman pretty
    prince boy king
    princess girl queen
    


```python
df.head(13) ## input 과 타겟 label 등장
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
      <th>input</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>king</td>
      <td>strong</td>
    </tr>
    <tr>
      <th>1</th>
      <td>king</td>
      <td>man</td>
    </tr>
    <tr>
      <th>2</th>
      <td>strong</td>
      <td>king</td>
    </tr>
    <tr>
      <th>3</th>
      <td>strong</td>
      <td>man</td>
    </tr>
    <tr>
      <th>4</th>
      <td>man</td>
      <td>king</td>
    </tr>
    <tr>
      <th>5</th>
      <td>man</td>
      <td>strong</td>
    </tr>
    <tr>
      <th>6</th>
      <td>queen</td>
      <td>wise</td>
    </tr>
    <tr>
      <th>7</th>
      <td>queen</td>
      <td>woman</td>
    </tr>
    <tr>
      <th>8</th>
      <td>wise</td>
      <td>queen</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wise</td>
      <td>woman</td>
    </tr>
    <tr>
      <th>10</th>
      <td>woman</td>
      <td>queen</td>
    </tr>
    <tr>
      <th>11</th>
      <td>woman</td>
      <td>wise</td>
    </tr>
    <tr>
      <th>12</th>
      <td>boy</td>
      <td>young</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.shape)
word2int # 어휘사전
```

    (52, 2)
    




    {'wise': 0,
     'strong': 1,
     'young': 2,
     'queen': 3,
     'king': 4,
     'boy': 5,
     'girl': 6,
     'woman': 7,
     'prince': 8,
     'man': 9,
     'princess': 10,
     'pretty': 11}



# Define Tensorflow Graph


```python
import tensorflow as tf
import numpy as np

ONE_HOT_DIM = len(words)

# function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM) ## 여기서는 max_len 이 결국 len(words) = 12 인 것을 알 수 있다. 
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = [] # input word
Y = [] # target word
```


```python
print(words, "\t", len(words))
```

    {'wise', 'strong', 'young', 'queen', 'king', 'boy', 'girl', 'woman', 'prince', 'man', 'princess', 'pretty'} 	 12
    


```python
for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))
## X:input 데이터는 sequence_length (통상 keras 모델 내부에서는 max_len 으로 configure 하는) 값이 12로 설정된것을 알 수 있다.

# convert them to numpy arrays
X_train = np.asarray(X)
Y_train = np.asarray(Y)
```


```python
X_train.shape
```




    (52, 12)




```python
# making placeholders for X_train and Y_train
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

# word embedding will be 2 dimension for 2d visualization
EMBEDDING_DIM = 2 
```

**tensorflow 로 표현되었지만 여기서부터가 Embedding layer 구성 부분이다. Start**


```python
print("ONE_HOT_DIM:",ONE_HOT_DIM,"/ EMBEDDING_DIM:",EMBEDDING_DIM)
```

    ONE_HOT_DIM: 12 / EMBEDDING_DIM: 2
    


```python
# hidden layer: which represents word vector eventually
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias
hidden_layer = tf.add(tf.matmul(x,W1), b1)
## keras 와 헷갈렸던 것은 keras 의 max_len 개념이다. keras 에서는 Embedding 층을 만들때, max_len 보다 훨씬 큰 값으로 공간을 구성하고, 실제로 작업도 한다.
## 여기서는 ONE_HOT_DIM 이 그런 역할을 하는데, 실제로, one-hot-encoding 으로 사전에 없는 단어는 모두 0으로 해서 들어오면서 shape을 맞추고 들어온다. 

# output layer ## target 을 값을 설정하는 것
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
```

**tensorflow 로 표현되었지만 여기서부터가 Embedding layer 구성 부분이다. End**


```python
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 

iteration = 20000
for i in range(iteration):
    # input is X_train which is one hot encoded word
    # label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))
```

    iteration 0 loss is :  5.0788465
    iteration 3000 loss is :  1.8832028
    iteration 6000 loss is :  1.7283807
    iteration 9000 loss is :  1.6977545
    iteration 12000 loss is :  1.6829585
    iteration 15000 loss is :  1.6734779
    iteration 18000 loss is :  1.6665666
    


```python
# Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
print(vectors)
```

    [[-2.4922278  -3.5493858 ]
     [ 2.8152335   1.5877781 ]
     [ 0.36534062 -0.11101   ]
     [-0.5324085  -0.5098076 ]
     [-0.1849049   1.0182045 ]
     [-0.04679373  1.0136434 ]
     [-0.75224614 -0.77613664]
     [-1.8010311  -1.0071619 ]
     [-1.0527248   5.2118983 ]
     [-1.2981308   5.186151  ]
     [-4.9132915  -3.2144003 ]
     [ 2.0580463  -2.0683904 ]]
    


```python
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]
w2v_df
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
      <th>word</th>
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wise</td>
      <td>-2.492228</td>
      <td>-3.549386</td>
    </tr>
    <tr>
      <th>1</th>
      <td>strong</td>
      <td>2.815233</td>
      <td>1.587778</td>
    </tr>
    <tr>
      <th>2</th>
      <td>young</td>
      <td>0.365341</td>
      <td>-0.111010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>queen</td>
      <td>-0.532408</td>
      <td>-0.509808</td>
    </tr>
    <tr>
      <th>4</th>
      <td>king</td>
      <td>-0.184905</td>
      <td>1.018204</td>
    </tr>
    <tr>
      <th>5</th>
      <td>boy</td>
      <td>-0.046794</td>
      <td>1.013643</td>
    </tr>
    <tr>
      <th>6</th>
      <td>girl</td>
      <td>-0.752246</td>
      <td>-0.776137</td>
    </tr>
    <tr>
      <th>7</th>
      <td>woman</td>
      <td>-1.801031</td>
      <td>-1.007162</td>
    </tr>
    <tr>
      <th>8</th>
      <td>prince</td>
      <td>-1.052725</td>
      <td>5.211898</td>
    </tr>
    <tr>
      <th>9</th>
      <td>man</td>
      <td>-1.298131</td>
      <td>5.186151</td>
    </tr>
    <tr>
      <th>10</th>
      <td>princess</td>
      <td>-4.913291</td>
      <td>-3.214400</td>
    </tr>
    <tr>
      <th>11</th>
      <td>pretty</td>
      <td>2.058046</td>
      <td>-2.068390</td>
    </tr>
  </tbody>
</table>
</div>



# word vector in 2d chart


```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1,x2 ))
    
PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
 
plt.xlim(x_axis_min,x_axis_max)
plt.ylim(y_axis_min,y_axis_max)
plt.rcParams["figure.figsize"] = (10,10)

plt.show()
```


![png](/assets/images/text_keras_minsuk_word2Vec/output_33_0.png)