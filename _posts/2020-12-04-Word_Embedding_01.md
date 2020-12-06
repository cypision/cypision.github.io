---
title:  "Word Embedding using konlpy 01"  
excerpt: "Text Analysis"  

categories:  
  - Deep-Learning  
tags:  
  - Text Analysis
  - embedding
  - Colab
last_modified_at: 2020-12-04T14:13:00-05:00
---

## Reference  
* 사설 강의 참조

하기 내용은 본인 컴퓨터가 window os 다 보니, 전적으로 Collab 환경에서만 실행가능합니다.  
(google colab 은 ubuntu라서 좀더 konlp가 쉽게 설치되는 것 같으니, 형태속 분석은 확실히 Colab에서~)

##### 연결link  
[colab](https://colab.research.google.com/github/cypision/Alchemy-in-MLDL/blob/master/DL_Area/Word_Embedding_01.ipynb)

#### google 내 my drive 연동하기  
하기 부분은 본인의 google drive를 연동하는 부분입니다.

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

#### konlpy라는 한국어 자연어처리 라이브러리 설치


```python
!pip install konlpy
!pip install jpype1==0.7.0
```


```python
import json

with open("/content/KorQuAD_v1.0_train.json") as f:
  data = json.loads(f.read())

PARAS = []
for dat in data["data"]:
  for para in dat["paragraphs"]:
    PARAS.append(para["context"])
```

Source Data :

- 한국어 위키백과 단락 일부
- KorQuAD 1.0 데이터 본문 활용


```python
!wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json
```

colab 에서, 기본적인 dataset은 content 으로 저장됩니다. 따라서 embedding 벡터로 만들기 위한 wiki 데이터를 content에 저장합니다.


```python
import json

with open("/content/KorQuAD_v1.0_train.json") as f:
  data = json.loads(f.read())

PARAS = []
for dat in data["data"]:
  for para in dat["paragraphs"]:
    PARAS.append(para["context"])
```


```python
print("** Number of Paragraphs:", len(PARAS))
PARAS[0]
```


```python
from konlpy.tag import Komoran, Hannanum, Kkma, Okt
komoran = Komoran()
hannanum = Hannanum()
kkma = Kkma()
okt = Okt()

# 문장, 형태소분석기를 인풋으로 받아 쪼개진 문장을 리턴하는 함수 정의
def tokenize(sentence, tokenizer):
  return tokenizer.morphs(sentence)
```


```python
sentence = "형태소 분석기마다 쪼개진 문장 특성이 조금씩 달라요"
print("한나눔:", tokenize(sentence, hannanum))
print("꼬꼬마:", tokenize(sentence, kkma))
print("코모란:", tokenize(sentence, komoran))
print("트위터:", tokenize(sentence, okt))
```

실제로 활용한 komoran 분석기를 활용해서 만듭니다.


```python
def tokenize(sentence):
  """ Your Code Here """
  return komoran.morphs(sentence)
```

## 단어사전을 만들기 전, 학습용과 검증요 데이터로 분할하기


```python
# 전체 문단을 Train과 Validation으로 나누기
import random
random.seed(1)
random.shuffle(PARAS)

PARAS_tr = PARAS[:8000]
PARAS_dev = PARAS[8000:]

print("Train: {} | Val: {}".format(len(PARAS_tr), len(PARAS_dev)))
```

#### step01. 토큰단위로 단어사전 만들기  
> Counter 객체를 활용 빈도수가 높은 단어부터 추출될 수 있도록 한다.


```python
# 훈련 셋에 있는 Paragraph를 형태소로 쪼개고,토큰의 빈도를 체크하기 
from tqdm import tqdm
from collections import Counter

vocab_freq = Counter()
TOKEN_PARAS_tr = []
for i,para in tqdm(enumerate(PARAS_tr)):
    tokenized_para = tokenize(para) ## 함수 tokenize 에 의해 list 형태로 return 받는다.
    for word in tokenized_para: ## return 받은 형태소마다, dict처럼 Counter 객체에 삽입
        vocab_freq[word] += 1
    
    TOKEN_PARAS_tr.append(tokenized_para) ## 2차원. list 의 list 형식
```

#### step02.  단어의 Frequency에 따라 단어사전에 포함할 토큰 골라내기


```python
N = 70000
most_common = vocab_freq.most_common(len(vocab_freq))

print("가장 많이 나온 10개 토큰:", most_common[:10])
print("가장 적게 나온 10개 토큰:",most_common[-10:])
```

최종 단어사전은 위에서 골라낸 토큰과 더불어 [PAD]와 [UNK] 토큰을 포함해야 합니다.

딥러닝에서 [PAD]토큰은 0번 인덱스를 사용하는 것이 일반적입니다.

따라서 0번째 자리에 [PAD], 1번째에 [UNK],
2번째부터는 위에서 골라낸 vocabulary_set에 있는 단어들이 차례로 오는 vocabulary_list를 만들겠습니다.

```python
vocabulary_set = [s[0] for s in most_common[:N]]
print(vocabulary_set[:10])
```

```python
## 단어 사전 생성을 위해 [PAD] , [UNK] 토큰을 포함하는 vocabulary_list 생성
vocabulary_list = ["[PAD]", "[UNK]"]
vocabulary_list.extend(vocabulary_set)

print("최종 단어 개수: {}개".format(len(vocabulary_list)) )
```


```python
vocabulary_list[:10]
```

여기서 나온 형태소를 잘 활용하여, 향후에 이용할 예정입니다.  
이후 Posting에서는 이를 이용하여, Embedding 계층 등을 공부하려 합니다.
