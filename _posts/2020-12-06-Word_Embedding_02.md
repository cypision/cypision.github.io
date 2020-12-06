---
title:  "Word Embedding using konlpy"  
excerpt: "Text Analysis"  

categories:  
  - Deep-Learning  
tags:  
  - Text Analysis
  - embedding
  - Colab
last_modified_at: 2020-12-06T14:13:00-05:00
---

## Reference  
* 사설 강의 참조

Word_Embedding_01 에서, 이어지며 형태소분석은 colab에서 수행한 자료를 활용한다.

#### google 내 my drive 연동하기


```python
# from google.colab import drive
# drive.mount('/content/gdrive')
```


```python
## 구글 드라이브에 있는 단어 리스트 로딩
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/meta.tsv") as f:
    vocabulary_list = [v.strip() for v in f.readlines()]
print("토큰 개수: {}개".format(len(vocabulary_list)))
```

    토큰 개수: 70002개
    


```python
print(len(vocabulary_list)) ## 한국어 형태소들을 list 형태로 만든 단어사전
print(type(vocabulary_list))
vocabulary_list[:10]
```

    70002
    <class 'list'>
    




    ['[PAD]', '[UNK]', '하', '이', '.', '의', '는', '을', '다', 'ㄴ']




```python
class TextEncoder(object):
    def __init__(self, vocab_list): # vocab_list를 받아서 기능을 수행함
        self.pad_token = "[PAD]" # 딥러닝 패딩 처리를 위한 토큰 명시
        self.oov_token = "[UNK]" # Unknown 토큰 처리하는 토큰 명시

        ## 토큰을 인덱스로 바꾸는 dictionary 만들기 
        token_to_id = {}
        ## 인덱스를 토큰으로 바꾸는 dictionary 만들기 
        ids_to_tokens = {}
        """ 
      token_to_id : vocab_list에 있는 토큰을 순서대로 토큰 - 정수로 매핑하는 dictionary
          (예) "[PAD]"  : 0
               "[UNK]"  : 1
               "하"  : 2
               "이" : 3
               "." : 4
        """
      ## Python에서 enumerate 함수는 리스트의 인덱스와 아이템을 튜플로 반환합니다
        for i, token in enumerate(vocab_list):
            token_to_id[token] = i
            ids_to_tokens[i] = token
#         ids_to_tokens = {v:k for k,v in token_to_id.items()}  
        self.token_to_id = token_to_id
        self.ids_to_tokens = ids_to_tokens
        self.vocab_size = len(token_to_id)   
        self.vocab_list = vocab_list
    
    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                """토큰을 token_to_id에서 찾을 수 있으면 해당 인덱스를 ids에 append"""
                ids.append(self.token_to_id[token])
            else:
                """토큰을 token_to_id에서 찾을 수 없으면 [UNK]에 해당하는 인덱스를 append"""
                ids.append(self.token_to_id["[UNK]"])
        return ids

    def convert_ids_to_tokens(self, ids): ## ids = list type
        return [self.ids_to_tokens[i] for i in ids]
```


```python
text_encoder = TextEncoder(vocabulary_list)
```


```python
print("PAD 토큰 확인:", text_encoder.pad_token)
print("OOV 토큰 확인:", text_encoder.oov_token)
print("단어사전 크기:", text_encoder.vocab_size)
```

    PAD 토큰 확인: [PAD]
    OOV 토큰 확인: [UNK]
    단어사전 크기: 70002
    

#### wikipidia paragraph 를 불러와서, 딥러닝학습을 시킬수 있는 정수형벡터로 변환합니다.


```python
## 구글 드라이브에 있는 문단들(형태소 분석기를 통해서, 한국어에 맞추어 토크나이즈된 문단파일을 로컬 다운로드하여 실행
import json
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/TOKEN_PARAS_tr.json" , 'r') as f:
    token_PARAS_tr = json.loads(f.read())
    
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/TOKEN_PARAS_dev.json" , 'r') as f:
    token_PARAS_dev = json.loads(f.read())
```


```python
print("Train: {} | Val: {}".format(len(token_PARAS_tr), len(token_PARAS_dev)))
```

    Train: 8000 | Val: 1681
    


```python
print(len(token_PARAS_tr[0]))
token_PARAS_tr[0][0:10] ## 한국어 특성에 맞추어 토크나이즈 된것을 알 수 있다.
```

    293
    




    ['차량', '정비', '소가', '있', '는', '패턴', '에', '만', '있', '다']




```python
sample = text_encoder.convert_tokens_to_ids(token_PARAS_tr[0])
print(sample)
```

    [1042, 2501, 9759, 22, 6, 3518, 10, 50, 22, 8, 4, 3770, 24, 1157, 17, 6, 2454, 2024, 7904, 3577, 20, 51, 71, 22, 16, 2103, 7, 19526, 2, 56, 11, 1268, 10, 4432, 449, 2103, 7, 62, 74, 1746, 10, 2749, 26, 34, 9, 3029, 12867, 5, 243, 14, 209, 26, 62, 16, 2064, 20, 1468, 37, 4, 7904, 20, 209, 26, 62, 9, 3029, 12867, 19, 7564, 29, 11940, 7, 513, 51, 97, 917, 233, 7, 41, 22, 8, 4, 1061, 34, 47, 569, 511, 724, 554, 146, 257, 7904, 6, 11941, 26, 330, 26, 39, 3632, 14, 1825, 32, 41, 22, 8, 4, 7904, 20, 338, 9, 146, 1670, 3519, 14, 515, 18, 15410, 6, 111, 3, 101, 6, 2981, 20, 2982, 30, 21, 268, 2, 37, 4, 7904, 6, 743, 15, 7, 69, 233, 7, 41, 22, 6, 5478, 3, 1208, 20, 213, 34, 47, 38, 86, 10, 17, 212, 1061, 6, 35, 3, 5479, 8, 4, 11942, 7904, 49, 2103, 7, 62, 15, 7, 69, 11, 7904, 20, 1881, 669, 36, 122, 10394, 57, 2, 56, 11, 3, 69, 4330, 74, 1918, 10, 2699, 16, 3194, 709, 6, 141, 20, 22, 56, 11, 1377, 39, 24, 1157, 17, 37, 4, 3708, 29, 11943, 13, 58, 59, 1158, 3268, 36, 3072, 3, 324, 342, 1882, 17, 26, 22, 8, 4, 22756, 11, 11944, 36, 309, 2399, 10, 11081, 9, 3633, 14, 1256, 2, 18, 511, 81, 917, 554, 2, 64, 4060, 102, 13, 580, 7, 1169, 2, 6, 35, 3, 308, 8, 4, 926, 3708, 11, 11943, 43, 7, 1228, 38, 60, 18, 1883, 10, 2455, 2, 171, 2, 6, 2663, 1047, 100, 3, 653, 6, 2345, 5, 141, 89, 9, 4761, 7, 1825, 34, 6, 124, 37, 4]
    


```python
## index 형태로 변형하기
## Train Data
TOKEN_IDS_tr = []
for sent in token_PARAS_tr:
    TOKEN_IDS_tr.append(text_encoder.convert_tokens_to_ids(sent))

## Dev Data
TOKEN_IDS_test = []
for sent in token_PARAS_dev:
    TOKEN_IDS_test.append(text_encoder.convert_tokens_to_ids(sent))
```


```python
print(len(TOKEN_IDS_tr),len(TOKEN_IDS_test))
```

    8000 1681
    


```python
token_PARAS_tr[0][0:11]
```




    ['차량', '정비', '소가', '있', '는', '패턴', '에', '만', '있', '다', '.']




```python
print("Index Example:", TOKEN_IDS_tr[0])
print(text_encoder.convert_ids_to_tokens(TOKEN_IDS_tr[0]))
```

    Index Example: [1042, 2501, 9759, 22, 6, 3518, 10, 50, 22, 8, 4, 3770, 24, 1157, 17, 6, 2454, 2024, 7904, 3577, 20, 51, 71, 22, 16, 2103, 7, 19526, 2, 56, 11, 1268, 10, 4432, 449, 2103, 7, 62, 74, 1746, 10, 2749, 26, 34, 9, 3029, 12867, 5, 243, 14, 209, 26, 62, 16, 2064, 20, 1468, 37, 4, 7904, 20, 209, 26, 62, 9, 3029, 12867, 19, 7564, 29, 11940, 7, 513, 51, 97, 917, 233, 7, 41, 22, 8, 4, 1061, 34, 47, 569, 511, 724, 554, 146, 257, 7904, 6, 11941, 26, 330, 26, 39, 3632, 14, 1825, 32, 41, 22, 8, 4, 7904, 20, 338, 9, 146, 1670, 3519, 14, 515, 18, 15410, 6, 111, 3, 101, 6, 2981, 20, 2982, 30, 21, 268, 2, 37, 4, 7904, 6, 743, 15, 7, 69, 233, 7, 41, 22, 6, 5478, 3, 1208, 20, 213, 34, 47, 38, 86, 10, 17, 212, 1061, 6, 35, 3, 5479, 8, 4, 11942, 7904, 49, 2103, 7, 62, 15, 7, 69, 11, 7904, 20, 1881, 669, 36, 122, 10394, 57, 2, 56, 11, 3, 69, 4330, 74, 1918, 10, 2699, 16, 3194, 709, 6, 141, 20, 22, 56, 11, 1377, 39, 24, 1157, 17, 37, 4, 3708, 29, 11943, 13, 58, 59, 1158, 3268, 36, 3072, 3, 324, 342, 1882, 17, 26, 22, 8, 4, 22756, 11, 11944, 36, 309, 2399, 10, 11081, 9, 3633, 14, 1256, 2, 18, 511, 81, 917, 554, 2, 64, 4060, 102, 13, 580, 7, 1169, 2, 6, 35, 3, 308, 8, 4, 926, 3708, 11, 11943, 43, 7, 1228, 38, 60, 18, 1883, 10, 2455, 2, 171, 2, 6, 2663, 1047, 100, 3, 653, 6, 2345, 5, 141, 89, 9, 4761, 7, 1825, 34, 6, 124, 37, 4]
    ['차량', '정비', '소가', '있', '는', '패턴', '에', '만', '있', '다', '.', '화가', '로', '추정', '되', '는', '중립', '민간인', '노숙자', 'NPC', '가', '1', '명', '있', '고', '음식', '을', '구걸', '하', '는데', ',', '종류', '에', '상관', '없이', '음식', '을', '주', '면', '지하', '에', '숨기', '어', '지', 'ㄴ', '물품', '더미', '의', '존재', '를', '알리', '어', '주', '고', '사기', '가', '올라가', 'ㄴ다', '.', '노숙자', '가', '알리', '어', '주', 'ㄴ', '물품', '더미', '에서', '보석', '과', '알콜', '을', '각각', '1', '개', '씩', '얻', '을', '수', '있', '다', '.', '돕', '지', '않', '으면', '몇', '차례', '방문', '뒤', '결국', '노숙자', '는', '굶', '어', '죽', '어', '그', '시체', '를', '가져오', 'ㄹ', '수', '있', '다', '.', '노숙자', '가', '나타나', 'ㄴ', '뒤', '누', '군가', '를', '찾', '아', '헤매', '는', '사람', '이', '오', '는', '이벤트', '가', '확률', '적', '으로', '발생', '하', 'ㄴ다', '.', '노숙자', '는', '죽이', '었', '을', '때', '얻', '을', '수', '있', '는', '아이템', '이', '가치', '가', '높', '지', '않', '기', '때문', '에', '되', '도록', '돕', '는', '것', '이', '낫', '다', '.', '간혹', '노숙자', '에게', '음식', '을', '주', '었', '을', '때', ',', '노숙자', '가', '감사', '인사', '와', '함께', '따라오', '라고', '하', '는데', ',', '이', '때', '따라가', '면', '중간', '에', '멈추', '고', '혼자', '돌아가', '는', '경우', '가', '있', '는데', ',', '버', '그', '로', '추정', '되', 'ㄴ다', '.', '식량', '과', '의약품', '은', '없', '지만', '각종', '재료', '와', '부품', '이', '매우', '많이', '소장', '되', '어', '있', '다', '.', '보리스', ',', '마르코', '와', '같이', '수집', '에', '능하', 'ㄴ', '생존자', '를', '활용', '하', '아', '몇', '번', '씩', '방문', '하', '면서', '최대한', '많', '은', '양', '을', '확보', '하', '는', '것', '이', '좋', '다', '.', '반면', '식량', ',', '의약품', '등', '을', '구하', '기', '위하', '아', '약탈', '에', '의존', '하', '아야', '하', '는', '다소', '안정', '성', '이', '떨어지', '는', '조합', '의', '경우', '크', 'ㄴ', '이득', '을', '가져오', '지', '는', '못하', 'ㄴ다', '.']
    

## CBOW 학습하기  
**CBOW 학습 가능한 형태로 변환하기**  
> 우리가 가진 데이터는 위키피디아 문단을 토크나이징한 데이터입니다.  
  하지만 CBOW 학습을 위해서는 예측하고자 하는 토큰 앞뒤로 w개의 토큰을 가지고 와야 하지요.  
  예를 들어 w = 3으로 설정했다면, 각각의 train example은  

X : [(t-3째 토큰), (t-2째 토큰), (t-1째 토큰), (t+1째 토큰), (t+2째 토큰), (t+3째 토큰)]  
Y : [(t번째 토큰)]의 형태여야 합니다.  
아래의 generate_context_word_pairs 함수는 토크나이징된 코퍼스와 윈도우 크기를 인풋으로 받아 이러한 형태로 데이터를 만드는 역할을 합니다.


```python
import tensorflow as tf
import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences ## tf.keras 에서 pad_sequences가 입력되는 문장의 padding 하는 역할을 한다.
```


```python
tf.keras.__version__
```




    '2.2.4-tf'



윈도우 크기를 3으로 설정하고, 데이터를 만들겠습니다.

  inputs_tr의 각 example은 길이가 6 (앞뒤로 3개, 총 6개 토큰),labels_tr의 각 example은 길이가 1 (맞춰야 하는 가운데 토큰)의 형태가 됩니다.
  
validation 데이터도 마찬가지로 만들어줍니다.


```python
def generate_context_word_pairs(corpus, window_size = 3):
    """
    index로 바뀐 코퍼스 리스트를 인풋으로 받아
    CBOW 학습용 input과 label을 리턴하는 함수
    """
    inputs = []
    labels = []

    context_length = window_size*2 # CBOW 방법이기 때문에 Target을 중심으로 좌우 window(3)*2 가 된다.
    
    for words in corpus: ## 여기선 8000개 문단 = corpus
        sentence_length = len(words) ## 1개 문단씩에 들어있는 tokenize된 단어들의 길이를 파악한다.
        
        for index, word in enumerate(words):
            if index < window_size or index >= sentence_length - window_size:
                # window size 안에 만들 수 없는 것들은 만들지 않음.
                continue ## for index, word in enumerate(words) 으로 다시 간다.
            
            context_words = []            
            start = index - window_size
            end = index + window_size + 1
      
            context_words= [words[i] for i in range(start, end) if 0 <= i < sentence_length and i != index]
        
            assert(len(context_words) == context_length)
            inputs.append(context_words)
            labels.append(word)
    
    return inputs, labels
```


```python
print(len(TOKEN_IDS_tr[0]),len(TOKEN_IDS_tr[11])) ## 간 문단별로 length가 다름을 알 수 있다.
```

    293 260
    


```python
WINDOW_SIZE = 3
inputs_tr, labels_tr = generate_context_word_pairs(TOKEN_IDS_tr, window_size=WINDOW_SIZE)
inputs_dev, labels_dev = generate_context_word_pairs(TOKEN_IDS_test, window_size=WINDOW_SIZE)
```


```python
print(np.array(inputs_tr).shape,np.array(labels_tr).shape,np.array(inputs_dev).shape,np.array(labels_dev).shape)
```

    (2013892, 6) (2013892,) (420084, 6) (420084,)
    

여기까지 CBOW 학습을 위한 준비입니다.  
이제 진짜 학습을 해봅시다.  
텐서플로우에서는 numpy array 형태의 인풋을 받기 때문에 np.array 형태로 데이터를 바꾸겠습니다.


```python
inputs_tr = np.array(inputs_tr)
labels_tr = np.array(labels_tr)
inputs_dev = np.array(inputs_dev)
labels_dev = np.array(labels_dev)
```


```python
print("TRAIN:", inputs_tr.shape, labels_tr.shape)
print("VAL  :", inputs_dev.shape, labels_dev.shape)
```

    TRAIN: (2013892, 6) (2013892,)
    VAL  : (420084, 6) (420084,)
    


```python
### CBOW 예시
print("** Example**")
for i in range(10):
    input_tokens = text_encoder.convert_ids_to_tokens(inputs_tr[i])
    gt_token = text_encoder.convert_ids_to_tokens([labels_tr[i]])[0]
    print("{} [ ] {} -> {}".format(" ".join(input_tokens[:WINDOW_SIZE])," ".join(input_tokens[WINDOW_SIZE:]), gt_token))
```

    ** Example**
    차량 정비 소가 [ ] 는 패턴 에 -> 있
    정비 소가 있 [ ] 패턴 에 만 -> 는
    소가 있 는 [ ] 에 만 있 -> 패턴
    있 는 패턴 [ ] 만 있 다 -> 에
    는 패턴 에 [ ] 있 다 . -> 만
    패턴 에 만 [ ] 다 . 화가 -> 있
    에 만 있 [ ] . 화가 로 -> 다
    만 있 다 [ ] 화가 로 추정 -> .
    있 다 . [ ] 로 추정 되 -> 화가
    다 . 화가 [ ] 추정 되 는 -> 로
    

 -> 오른쪽의 값들을 Target 으로 추정하면 됩니다.


```python
print(len(vocabulary_list))
len(vocabulary_list) == text_encoder.vocab_size
```

    70002
    




    True




```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
 
VOCAB_SIZE = text_encoder.vocab_size ## 단어사전 개수. 토큰화된 모든 단어에 대한 사전 여기선 70002
EMBED_SIZE = 128 ## 임베딩 차원 개수
INPUT_LENGTH = WINDOW_SIZE * 2 ## 인풋 길이 ## 6 dim ; WINDOW_SIZE = 3
 
cbow_model = Sequential()
# 1. 임베딩 레이어 추가
cbow_model.add( Embedding(VOCAB_SIZE, EMBED_SIZE, input_length = INPUT_LENGTH) ) ## output:(batch,sequence_length,embedding_dim)
 
# 2. 임베딩된 벡터들의 평균 구하기 (Lambda 레이어 함수 사용)
cbow_model.add(Lambda(lambda x: tf.keras.backend.mean(x, axis=1), output_shape=(EMBED_SIZE,))) ##내부적으로 OHE로 변형후,+되고 * 1/n 되니, 평균을 하는 것이 맞다.
 
# 3. 가운데 들어갈 단어를 예측하는 Fully Connected Layer 연결
cbow_model.add(Dense(VOCAB_SIZE, activation = "softmax")) ##

print(cbow_model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 6, 128)            8960256   
    _________________________________________________________________
    lambda (Lambda)              (None, 128)               0         
    _________________________________________________________________
    dense (Dense)                (None, 70002)             9030258   
    =================================================================
    Total params: 17,990,514
    Trainable params: 17,990,514
    Non-trainable params: 0
    _________________________________________________________________
    None
    

#### <span style='color:red'>mini topic</span>  
[sparse_categorical_crossentropy vs categorical_crossentropy](https://kminito.github.io/machine_learning/2018/10/22/studypie_deeplearning_1/)  

* sparse_categorical_crossentropy : 범주형 교차 엔트로피와 동일하지만 이 경우 __원-핫 인코딩이 된 상태일 필요없이 정수 인코딩 된 상태__에서 수행 가능.  
  > 한개의 있는 정답레이블을 가지고 있음  
  > 여기선 labels_tr 안에, one-hot-encoding 이 아닌 정수 레이블 값만 가지고 있으므로 sparse_categorical_crossentropy를 사용하는 것이 맞다.  
  > 마찬가지로 inputs_tr 에 해당하는 값들도 OHE 가 아니다. "sparse_categorical_crossentropy"으로 선택하면, 내부적으로 이를 OHE 형태로 바꾸어서 tensorflow 에서 수행한다.
* categorical_crossentropy : 원-핫-인코딩된 레이블로 cross entropy를 계산 (target과 output의 shape가 같아야 함)


```python
cbow_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
```

#### <span style='color:red'>mini topic</span>  
[Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)

__Callback 은 학습을 도와주는 도구이며, 기존것을 그대로 사용하기도 하고, 상속받아서 나만의 함수로 사용하기도 한다.__  

* 자주사용하는 사례  
1) tf.keras.callbacks.ModelCheckpoint : 모델 체크포인트 저장. 훈련하는 동안 어떤 지점에서 모델의 현재 가중치를 저장합니다.  
2) tf.keras.callbacks.EarlyStopping : 조기종료. 검증 손실이 더 이상 향상되지 않을 때 훈련을 중지합니다.  
3) tf.keras.callbacks.LearningRateScheduler : 정해진 epoch시에 learning rate 조절(미리 정해준다.)  
4) tf.keras.callbacks.ReduceLROnPlateau : 모델은 학습이 정체되면 학습률을 2-10배 감소시킴으로써 이익을 얻는 경우가 많다. 이 콜백은 양을 모니터링하며, '인내' 에폭 수에 대한 개선이 보이지 않으면 학습률이 감소한다. epoch 타임을 지정하는 3번과는 다르다.    
5) tf.keras.callbacks.CSVLogger : CSV 파일형태로 string 타입으로 log를 남긴다.

* 상속해서 사용시, 참고할 사항 
1) on_epoch_begin : 각 에폭이 시작할때 호출  
2) on_epoch_end : 각 에폭이 끝날때 호출  
3) on_batch_begin : 각 배치처리가 시작되기 전에 호출  
4) on_batch_end : 각 배치처리가 끝난 후에 호출  
5) on_train_begin : 훈련이 시작될 때 호출  
6) on_train_end : 훈련이 끝날 때 호출  


```python
print(len(cbow_model.get_weights()))
for i in range(len(cbow_model.get_weights())):
    print(cbow_model.get_weights()[i].shape)
```

    3
    (70002, 128)
    (128, 70002)
    (70002,)
    

70002*128 : 입력층과 Embedding 사이의 가중치. Embedding 아웃풋을 만들어내는 가중치(아마 6차원의 input이나, 내부적으로 ohe로 변환해서 70002 값이 나왔을 것으로 예상)  


```python
## 하기 사례는 상속받아서, callback 함수를 만드는 예제이다.

class MyCustomCallback(tf.keras.callbacks.Callback):
    """샘플 단어와 가장 유사한 N개 단어를 보여줌"""
    reverse_dictionary = text_encoder.ids_to_tokens
    
    ## 학습이 진행되는동안, valid_dataset에 있는 각각의 명사 토큰에 대해 가장 가깝게 임베딩된 토큰이 어떤 것들이 있는지 확인할 것입니다.
    ## 이를 통해 정말 관련 있는 토큰들이 비슷한 공간에 매핑되는지 확인하겠습니다.
    def on_epoch_end(self, batch, logs=None):
        valid_dataset = ["취업", "사랑", "주식"]
        valid_dataset = text_encoder.convert_tokens_to_ids(valid_dataset)
        embedding = cbow_model.get_weights()[0] ## 임베딩 계층의 가중치를 불러온다.

        reverse_dictionary = text_encoder.ids_to_tokens
        ## square:제곱 -> reduce_sum -> sqrt : 제곱의 합 루트 인데. 이는 square 인자인 embedding이 편차의 개념으로 해석가능하는 얘기인데....
        norm = tf.keras.backend.sqrt(tf.reduce_sum(tf.keras.backend.square(embedding), 1, keepdims=True))
        
        normalized_embeddings = embedding / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        
        print("valid_embeddings.shape {}".format(valid_embeddings.shape))
        print("normalized_embeddings.shape {}".format(normalized_embeddings.shape))
        
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        print("similarity.shape {}".format(similarity.shape))
        print("")
        for val_i in range(len(valid_dataset)):
            valid_word = reverse_dictionary[valid_dataset[val_i]]
            top_k = 8 # number of nearest neighbors
            nearest = np.array(-similarity[val_i, :]).argsort()[1:top_k+1] 
            print("{} -> {}".format(valid_word, " , ".join(text_encoder.convert_ids_to_tokens(nearest))))
```

#### <span style='color:red'>mini topic</span>  
__tensorflow 함수__  

* tf.reduce_sum : np.array 의 sum 과 같음. 특별한거 없음.
* tf.nn.embedding_lookup : [한글설명](https://m.blog.naver.com/wideeyed/221343328832), [공식Tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)  
 > array 계열에서, index 로 값을 찾는 것임. 파라미터로, array 와 idx 를 주고, idx에 해당하는 값을 array에서 불러온다.  
* tf.matmul : [3차원 이상을 곱할 수 있는 tf.matmul 설명](https://bowbowbow.tistory.com/27)


```python
valid_dataset = ["취업", "사랑", "주식"]
valid_dataset = text_encoder.convert_tokens_to_ids(valid_dataset)
```


```python
valid_dataset
```




    [6406, 741, 4564]




```python
embedding = cbow_model.get_weights()[0]
```


```python
embedding.shape
```




    (70002, 128)




```python
bb = tf.nn.embedding_lookup(embedding, valid_dataset)
```


```python
bb.shape
```




    TensorShape([3, 128])




```python
history = cbow_model.fit(inputs_tr, labels_tr, epochs = 3, batch_size=256,validation_data = (inputs_dev, labels_dev),
                         callbacks = [MyCustomCallback()])
```

    Train on 2013892 samples, validate on 420084 samples
    Epoch 1/3
    2013696/2013892 [============================>.] - ETA: 0s - loss: 6.3109 - accuracy: 0.1725valid_embeddings.shape (3, 128)
    normalized_embeddings.shape (70002, 128)
    similarity.shape (3, 70002)
    
    취업 -> 영속 , 반항 , 정력 , 기하급수 , 암시장 , 보통선거 , 낙태 , 스폰서
    사랑 -> 존중 , 충성 , 봉헌 , 추종 , 대변 , 관장 , 침략 , 부탁
    주식 -> 장단 , 생산수단 , 조영구 , 부사장 , 습성 , 대륙군 , 전기공학 , 곤충
    2013892/2013892 [==============================] - 515s 256us/sample - loss: 6.3108 - accuracy: 0.1725 - val_loss: 5.5655 - val_accuracy: 0.2305
    Epoch 2/3
    2013696/2013892 [============================>.] - ETA: 0s - loss: 5.4177 - accuracy: 0.2526valid_embeddings.shape (3, 128)
    normalized_embeddings.shape (70002, 128)
    similarity.shape (3, 70002)
    
    취업 -> 반항 , 외지 , 막 전위 , 정력 , 다방면 , 기하급수 , 낙태 , 기름기
    사랑 -> 존경 , 충성 , 존중 , 형벌 , 소견 , 살육 , 격려 , 자랑
    주식 -> 장단 , 골격 , 양주 , 월식 , 윤리관 , 963 , 가방 , 생산수단
    2013892/2013892 [==============================] - 534s 265us/sample - loss: 5.4177 - accuracy: 0.2526 - val_loss: 5.2952 - val_accuracy: 0.2549
    Epoch 3/3
    2013696/2013892 [============================>.] - ETA: 0s - loss: 5.0977 - accuracy: 0.2784valid_embeddings.shape (3, 128)
    normalized_embeddings.shape (70002, 128)
    similarity.shape (3, 70002)
    
    취업 -> 외지 , 기름기 , 반항 , 후삼국 , 다방면 , 조선 태조 , 카파도키아 , 텐션
    사랑 -> 존경 , 충성 , 형벌 , 격려 , 괴롭힘 , 우롱 , 존중 , 인정사정
    주식 -> 골격 , 평방미터 , 인천 상륙 작전 , 스티브 오스틴 , 수훈 , 도끼눈 , 상공업 , 장단
    2013892/2013892 [==============================] - 523s 260us/sample - loss: 5.0976 - accuracy: 0.2784 - val_loss: 5.1715 - val_accuracy: 0.2637
    

엄청난 효과가 있지는 않아 보인다. epoch 3 회가 짧아서 일 수 있다고 본다.  
상기 실습은 실습만으로 하고, 기본적으로 colab에서 해봤던 10회 학습산 embdding vecs를 불러와서, 활용한다.


```python
# 학습 완료된 임베딩 저장하기 -> colab 불러오기
# final_embeddings = cbow_model.get_weights()[0]
# final_embeddings = np.array(final_embeddings)
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/vecs.tsv") as f:
    vecs = [v.strip() for v in f.readlines()]
```


```python
## 해당 vecs 에 해당하는 원래 단어사전 (형태소 형태로 분해된) 불러오기.
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/meta.tsv") as m:
    meta = [v.strip() for v in m.readlines()]
```


```python
print(len(vecs))
vecs[0]
```

    70002
    




    '0.0365821\t0.020926926\t0.04379524\t0.025817778\t0.047427405\t-0.03714005\t0.038845602\t-0.022639036\t0.04566213\t-0.017310537\t-0.005673837\t0.03389481\t0.031991158\t-0.02141558\t-0.040702354\t0.016473625\t0.03739396\t0.02362654\t-0.0015142187\t-0.022148991\t-0.049571205\t-0.009789277\t0.0285036\t-0.049127974\t0.03504095\t0.03913707\t-0.029798245\t-0.0020481236\t-0.017411698\t-0.02782743\t-0.010669373\t0.006498255\t-0.021289801\t-0.043580055\t0.041048195\t-0.044386815\t-0.04638506\t-0.03911189\t0.0139717795\t-0.04726678\t0.041280355\t-0.04033055\t0.009563409\t-0.004145123\t-4.416704e-05\t0.014288101\t-0.02947756\t0.024156142\t0.03523482\t0.013411369\t-0.029149259\t0.023977172\t0.024519656\t-0.033418883\t-0.022569133\t-0.047230016\t-0.03417834\t0.0027229078\t0.022813905\t-0.010491859\t-0.0108927\t0.030535702\t-0.038820256\t0.04104365\t-0.04767982\t0.04081842\t0.0076200254\t-0.02153082\t-0.04976231\t0.01464298\t-0.026091874\t-0.02954942\t-0.010259282\t-0.02749442\t-0.021459758\t0.04664023\t0.03798366\t-0.026094437\t-0.028986681\t-0.03955709\t0.007559158\t-0.026334787\t-0.01666814\t-0.008553255\t-0.040870905\t0.025430743\t-0.049150765\t0.04360843\t-0.010821722\t0.029513586\t-0.044712353\t0.02648506\t5.115196e-05\t0.015059102\t0.018474046\t-0.026803136\t-0.036164522\t0.0059796795\t0.0049309954\t0.00821067\t-0.044413235\t-0.04525757\t0.046478603\t0.049130883\t-0.042209413\t-0.030328715\t-0.025949169\t-0.011594426\t-0.047002662\t-0.032153085\t-0.046049178\t-0.037304807\t0.03655119\t0.0051257983\t-0.027953863\t-0.047394015\t-0.022994472\t0.024174843\t0.020729396\t0.027144697\t-0.04141145\t-0.0032583103\t0.012337614\t0.013152506\t-0.0022288188\t-0.046839785\t-0.039002337\t0.007838272'




```python
for i in range(10):
    a = vecs[i].split("\t")
    print("{}번째 tsv파일의 dim 길이: {}".format(i,len(a)))
```

    0번째 tsv파일의 dim 길이: 128
    1번째 tsv파일의 dim 길이: 128
    2번째 tsv파일의 dim 길이: 128
    3번째 tsv파일의 dim 길이: 128
    4번째 tsv파일의 dim 길이: 128
    5번째 tsv파일의 dim 길이: 128
    6번째 tsv파일의 dim 길이: 128
    7번째 tsv파일의 dim 길이: 128
    8번째 tsv파일의 dim 길이: 128
    9번째 tsv파일의 dim 길이: 128
    


```python
final_embeddings = [np.float32(v.split("\t")) for v in vecs]
```


```python
print(len(final_embeddings[0])) ## 2중 list 형태로 불러왔다.
```

    128
    


```python
print(meta[0],meta[99])
```

    [PAD] 따르
    

#### 원하는 token과 가장 비슷한 단어를 찾아오기


```python
reverse_dictionary = {}
token_to_id_dictionary = {}
for i, token in enumerate(meta):
    reverse_dictionary[i] = token
    token_to_id_dictionary[token] = i
```


```python
def search_nearest(search_token, top_k = 5):
    if search_token not in token_to_id_dictionary:
        print("해당 토큰은 단어사전에서 찾을 수 없음")
        
    search_id = token_to_id_dictionary[search_token]
    print("{} -> {}".format(search_token, search_id))
    norm = tf.keras.backend.sqrt(tf.reduce_sum(tf.keras.backend.square(final_embeddings), 1, keepdims=True))
    normalized_embeddings = final_embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, [search_id])
    
    print("valid_embeddings.shape {}".format(valid_embeddings.shape))
    print("normalized_embeddings.shape {}".format(normalized_embeddings.shape))
    
    ## 유사도 계산 부분
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    print("similarity.shape {}".format(similarity.shape))
    print("")

    nearest = np.array(similarity[0, :]).argsort()[-(top_k+1):]  ## argsort : 값이 작은 순서데로 앞에. 즉 오름차순, sorting하여 index를 반환한다.
    print("Nearest Tokens: {}".format(" , ".join([reverse_dictionary[s] for s in nearest])))
    
    return similarity ## 확인용으로 추가함
```


```python
# embedding = cbow_model.get_weights()[0] ## 임베딩 계층의 가중치를 불러온다.
tf.keras.backend.sqrt(tf.reduce_sum(tf.keras.backend.square(embedding), 1, keepdims=True))
```




    <tf.Tensor: shape=(70002, 1), dtype=float32, numpy=
    array([[0.33434188],
           [0.314463  ],
           [0.31867397],
           ...,
           [0.32795262],
           [0.33271974],
           [0.33176792]], dtype=float32)>




```python
final_embeddings = np.array(final_embeddings)
final_embeddings.shape
```




    (70002, 128)




```python
embedding.shape
```




    (70002, 128)




```python
similarity = search_nearest("학교")
```

    학교 -> 368
    valid_embeddings.shape (1, 128)
    normalized_embeddings.shape (70002, 128)
    similarity.shape (1, 70002)
    
    Nearest Tokens: 중등 , 부산대학교 , 약대 , 대학교 , 캠퍼스 , 학교
    

#### <span style='color:red'>mini topic</span>  
__==유사도 코딩 부분 상세 분석 Start==__  


```python
similarity[0, :]
```




    <tf.Tensor: shape=(70002,), dtype=float32, numpy=
    array([-0.02431851,  0.22033407,  0.03669621, ...,  0.27966908,
            0.36452034,  0.19863272], dtype=float32)>




```python
## similarity 에 -1 를 곱하고, array로 변환 이를 argsort 함.
np.array(-similarity[0, :]).argsort()[1:top_k+1] 
```




    <tf.Tensor: shape=(70002,), dtype=float32, numpy=
    array([ 0.02431851, -0.22033407, -0.03669621, ..., -0.27966908,
           -0.36452034, -0.19863272], dtype=float32)>




```python
import random
st = np.random.randn(10)
st = st.reshape(2,5)
print(st,"\n")
print(st[0,:])
print(st[0,:].argsort())
```

    [[-0.36165447 -0.74859038 -0.128252   -0.91038556 -0.74254644]
     [ 0.99937184 -0.48977739  0.42717513  2.33262247  0.90280153]] 
    
    [-0.36165447 -0.74859038 -0.128252   -0.91038556 -0.74254644]
    




    array([3, 1, 4, 0, 2], dtype=int64)



(1,128)* transposed(70002*128) ==> (1,128)*(128*70002) = (1*70002)  

valid_embeddings.shape (1, 128): 입력단어의 임베딩값  
normalized_embeddings.shape (70002, 128) : 전체차원의 임베딩 값  
similarity.shape (1, 70002) : valid_embeddings,normalized_embeddings 의 곱은 Cosine유사도를 위한 벡터의 내적이다. 즉, 기존 단어중 가장 유사값이 높은것을 찾기 위함  
허나, norm 과정에서 square를 하는 부분에서, 평균값을 제거하는 부분이 없는것은 의문이긴 하다.
__==유사도 코딩 부분 상세 분석 End==__


```python
search_nearest("어머니")
```

    어머니 -> 557
    valid_embeddings.shape (1, 128)
    normalized_embeddings.shape (70002, 128)
    similarity.shape (1, 70002)
    
    Nearest Tokens: 아들 , 딸 , 할아버지 , 아내 , 아버지 , 어머니
    




    <tf.Tensor: shape=(1, 70002), dtype=float32, numpy=
    array([[-0.13504691,  0.32572174, -0.18390521, ...,  0.12888134,
             0.170396  ,  0.37434417]], dtype=float32)>


