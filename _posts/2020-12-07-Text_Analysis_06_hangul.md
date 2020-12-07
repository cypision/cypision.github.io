---
title:  "Sentiment Analysis using Korean"  
excerpt: "Text Analysis"  

categories:  
  - Deep-Learning  
tags:  
  - Text Analysis
  - 한글
  - 웰니스_대화_스크립트_데이터셋.xlsx
last_modified_at: 2020-12-06 T16:13:00-05:00
---

## Reference  
* [AI Hub 에서 download](https://www.aihub.or.kr/)  
* 웰니스 대화 스크립트를 기본적으로 활용한다.

Word_Embedding_01,02 에서 사용했던, Embedding vector를 활용한다.  

#### google 내 my drive 연동하기


```python
# from google.colab import drive
# drive.mount('/content/gdrive')
```


```python
import numpy as np
import json
import random
import pandas as pd
```


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
final_embeddings = [np.float32(v.split("\t")) for v in vecs]
```


```python
print(len(final_embeddings[0])) ## 2중 list 형태로 불러왔다.
```

    128
    


```python
EXCEL_FILE_NALE = "D:/20_CNS_Text_Analysis/data_set/웰니스_대화_스크립트_데이터셋.xlsx"
data = pd.read_excel(EXCEL_FILE_NALE)
```


```python
data.head()
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
      <th>구분</th>
      <th>유저</th>
      <th>챗봇</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>감정/감정조절이상</td>
      <td>제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.</td>
      <td>감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>감정/감정조절이상</td>
      <td>더 이상 내 감정을 내가 컨트롤 못 하겠어.</td>
      <td>저도 그 기분 이해해요. 많이 힘드시죠?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>감정/감정조절이상</td>
      <td>하루종일 오르락내리락 롤러코스터 타는 기분이에요.</td>
      <td>그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>감정/감정조절이상</td>
      <td>꼭 롤러코스터 타는 것 같아요.</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>감정/감정조절이상</td>
      <td>롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(data.shape)
print(len(data['구분']))
```

    (5231, 3)
    5231
    


```python
data['구분'].value_counts()
```




    증상/불면           209
    모호함             180
    감정/힘듦           106
    감정/자살충동          94
    감정/부정적사고         93
                   ... 
    증상/저림현상/발/손       5
    배경/남편/관계소원        5
    배경/부모/가출/아버지      5
    증상/체력저하           5
    배경/부모/어머니/죽음      5
    Name: 구분, Length: 359, dtype: int64



#### **필요한 부분만 발췌해서 데이터화 한다.**


```python
data['챗봇'][3]
```




    nan




```python
## data['챗봇'][3] nan 이기 때문에 nan != nan 이 성립한다.
data['챗봇'][3] != data['챗봇'][3]
```




    True




```python
data['챗봇'][0] != data['챗봇'][0]
```




    False




```python
pd.isnull(data['챗봇'][3])
```




    True




```python
## 챗봇컬럼이 빈칸인지를 확인하기 위한 값. pd.isnull() 해도 될듯
def isNaN(num):
    return num != num
```


```python
DATA = []
RESPONSE = {}
 
for i in range(len(data["구분"])):
    label = data["구분"][i]
    label_split = label.split("/")
    label_1 = "/".join(label_split[:2])
    sent = data["유저"][i]
    if label_1 != "모호함": 
        DATA.append(["Sent_{}".format(i), sent, label_1, label])
    if label_1 in RESPONSE:  ## 이미 RESPONSE dict에 key로 있을 경우
        if not pd.isnull(data["챗봇"][i]): ## 챗봇 컬럼 값이 NAN이 아닌 경우 
            RESPONSE[label_1].append(data["챗봇"][i]) ## key가 존재하는 상태에서 기존 "챗봇"값에 새로운 "챗봇"값을 append한다.
    else: ## label_1 이 "모호함"인 경우 or label_1 이 이미 RESPONSE dict에 key로 있을 경우
        if not pd.isnull(data["챗봇"][i]):  ## 챗봇 컬럼 값이 NAN이 아닌 경우 
            RESPONSE[label_1] = [data["챗봇"][i]] ## value 를 추가하는데, 최초 list 형대로 값을 집어넣는다.
 
"""random shuffle & make them into train/test set"""
labels = [dat[2] for dat in DATA]
```


```python
print(len(DATA))
print(DATA[0])
print(labels[0],type(labels[0]))
```

    5051
    ['Sent_0', '제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.', '감정/감정조절이상', '감정/감정조절이상']
    감정/감정조절이상 <class 'str'>
    


```python
type(RESPONSE)
```




    dict




```python
RESPONSE['증상/피해망상']
```




    ['그런 기분이 들 때 정말 힘들죠. 생각을 좀 달리 가져보면 편하지 않을까요?',
     '그 기분 뭔지 알 것 같아요. 조금은 다르게 생각을 해보는 것도 좋을 것 같아요.',
     '정말 힘드시겠어요. 제가 옆에서 힘이 되어 드릴게요.',
     '정말 스트레스 받으시겠어요. 다른 분에게 도움을 요청해보는 건 어떨까요?',
     '누군가 나를 지켜보고 있다 생각하면 너무 힘들겠어요. 제가 도움이 되고 싶네요.',
     '감시 당하는 것만큼 신경쓰이는 게 없죠. 아직도 그 상황에 처해 계신가요?',
     '감시를 당하시는 건가요? 정말 힘드시겠어요.',
     '정말 곤란하시겠군요. 한 번 대화를 나눠 보는 건 어떨까요?',
     '정도가 심하면 경찰이나 병원의 도움을 받아보는 건 어떠세요?',
     '정말 힘든 상황이시군요. 제가 작게나마 위로가 되고 싶어요.',
     '많이 힘드시겠어요. 고민을 털어 놓을 데가 필요하시면 제가 도와드릴게요.',
     '고민이 많으셨겠어요. 확신을 위해 좀 더 생각해보는 건 어떨까요?',
     '정말 힘드시겠어요. 제가 항상 옆에 있어 드릴게요. 힘내세요.',
     '감시 당하는 것만큼 괴로운 게 없죠. 제가 도움이 되어 드리고 싶네요.',
     '생활이 불편하시겠어요. 다른 사람의 도움을 받아 보는 건 어떨까요?']



================================================================================================================================
#### 웰니스 데이터 감성분석 train, test 셋으로 나누기  
이후, 감정분석 label 갯수확인


```python
print(len(DATA),DATA[0])
print(len(labels),labels[0])
```

    5051 ['Sent_0', '제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.', '감정/감정조절이상', '감정/감정조절이상']
    5051 감정/감정조절이상
    


```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(DATA, random_state = 2020, stratify = labels, test_size = 400)
```


```python
print("Data Example")
for i in range(5):
    print(train[i])
```

    Data Example
    ['Sent_4393', '뭔가 하루종일 이렇게 들뜬 기분이다 보니까 잠도 잘 안 와.', '증상/불면', '증상/불면']
    ['Sent_603', '아무한테나 화내고 그러지는 않아.', '감정/분노', '감정/분노']
    ['Sent_4224', '잠자리에 누워도 맨날 뒤척이고... 잠을 제대로 잘 수 있을 리가 없지.', '증상/불면', '증상/불면']
    ['Sent_3849', '5일 전에는 새벽에 일어나서 화장실을 가다가 순간적으로 정신을 잃었어.', '증상/기절', '증상/기절']
    ['Sent_666', '그냥 감정이입이 심하게 되고 불안감도 잘 느끼는 것 같아요.', '감정/불안감', '감정/불안감']
    


```python
import collections

train_counter = collections.Counter()
```


```python
for dat in train:
    train_counter[dat[2]] += 1 ## dat[2] DATA 내에서, 2 index 의 증상부분을 key로 삽입
print("라벨 개수:", len(train_counter), "\n") ## 176 개
print("*** LABEL 분포 ***")

for cnt in train_counter.most_common():
    print("{} : {} ({:.2f}%)".format(cnt[0], cnt[1], 100*cnt[1]/len(train))) ## cnt[0]:key , cnt[1]:counting 갯수
```

    라벨 개수: 176 
    
    *** LABEL 분포 ***
    증상/불면 : 236 (5.07%)
    배경/직장 : 152 (3.27%)
    배경/남편 : 142 (3.05%)
    감정/걱정 : 134 (2.88%)
    배경/부모 : 125 (2.69%)
    감정/힘듦 : 111 (2.39%)
    배경/생활 : 107 (2.30%)
    배경/성격 : 94 (2.02%)
    감정/불안감 : 91 (1.96%)
    감정/우울감 : 87 (1.87%)
    감정/자살충동 : 87 (1.87%)
    증상/무기력 : 87 (1.87%)
    감정/부정적사고 : 86 (1.85%)
    증상/피해망상 : 82 (1.76%)
    증상/식욕저하 : 67 (1.44%)
    배경/건강문제 : 65 (1.40%)
    배경/남자친구 : 60 (1.29%)
    증상/반복행동 : 59 (1.27%)
    배경/학교 : 56 (1.20%)
    배경/문제 : 55 (1.18%)
    배경/음주 : 53 (1.14%)
    감정/답답 : 51 (1.10%)
    배경/대학 : 48 (1.03%)
    배경/연애 : 47 (1.01%)
    감정/짜증 : 46 (0.99%)
    배경/경제적문제 : 46 (0.99%)
    배경/사업 : 45 (0.97%)
    증상/기억력저하 : 45 (0.97%)
    증상/호흡곤란 : 44 (0.95%)
    배경/여자친구 : 41 (0.88%)
    치료이력/병원내원 : 40 (0.86%)
    증상/두통 : 39 (0.84%)
    증상/두근거림 : 37 (0.80%)
    배경/친구 : 37 (0.80%)
    배경/어린시절 : 35 (0.75%)
    감정/화 : 35 (0.75%)
    증상/환청 : 34 (0.73%)
    배경/대인관계 : 33 (0.71%)
    부가설명 : 33 (0.71%)
    증상/은둔 : 32 (0.69%)
    감정/심란 : 31 (0.67%)
    증상/통증 : 31 (0.67%)
    배경/취업 : 30 (0.65%)
    배경/결혼 : 30 (0.65%)
    배경/가족 : 30 (0.65%)
    감정/후회 : 30 (0.65%)
    감정/눈물 : 29 (0.62%)
    배경/시댁 : 29 (0.62%)
    배경/자녀 : 29 (0.62%)
    자가치료/심리조절 : 29 (0.62%)
    감정/괴로움 : 28 (0.60%)
    증상/폭식 : 28 (0.60%)
    감정/생각 : 28 (0.60%)
    감정/분노 : 27 (0.58%)
    증상/죽음공포 : 27 (0.58%)
    배경/학업 : 26 (0.56%)
    감정/자괴감 : 25 (0.54%)
    증상/체중감소 : 25 (0.54%)
    배경/사고 : 24 (0.52%)
    증상/어지러움 : 22 (0.47%)
    감정/무서움 : 22 (0.47%)
    증상/피로 : 22 (0.47%)
    증상/대인기피 : 22 (0.47%)
    감정/외로움 : 21 (0.45%)
    감정/자존감저하 : 21 (0.45%)
    치료이력/검사 : 21 (0.45%)
    증상/집중력저하 : 21 (0.45%)
    감정/의욕상실 : 21 (0.45%)
    감정/불만 : 21 (0.45%)
    일반대화 : 19 (0.41%)
    감정/감정조절이상 : 19 (0.41%)
    증상/반복사고 : 19 (0.41%)
    감정/억울함 : 19 (0.41%)
    증상/공황발작 : 18 (0.39%)
    감정/서운함 : 18 (0.39%)
    감정/충격 : 18 (0.39%)
    증상/자해 : 18 (0.39%)
    감정/모호함 : 18 (0.39%)
    감정/두려움 : 17 (0.37%)
    감정/불쾌감 : 17 (0.37%)
    감정/절망감 : 17 (0.37%)
    감정/슬픔 : 17 (0.37%)
    증상/가슴답답 : 17 (0.37%)
    상태/증상지속 : 16 (0.34%)
    감정/신경쓰임 : 16 (0.34%)
    배경/애완동물 : 16 (0.34%)
    배경/임신 : 16 (0.34%)
    증상/이명 : 15 (0.32%)
    감정/자신감저하 : 15 (0.32%)
    감정/기분저하 : 15 (0.32%)
    감정/공포 : 15 (0.32%)
    증상/악몽 : 15 (0.32%)
    증상/자살시도 : 14 (0.30%)
    증상/기억상실 : 14 (0.30%)
    감정/속상함 : 14 (0.30%)
    상태/양호 : 14 (0.30%)
    감정/긴장 : 14 (0.30%)
    내원이유/상담 : 14 (0.30%)
    감정/비관적 : 14 (0.30%)
    배경/자각 : 13 (0.28%)
    증상/기절예기 : 13 (0.28%)
    증상/체중증가 : 13 (0.28%)
    배경/진로 : 13 (0.28%)
    증상/기절 : 12 (0.26%)
    감정/살인욕구 : 12 (0.26%)
    배경/공부 : 12 (0.26%)
    증상/가슴떨림 : 12 (0.26%)
    감정/허무함 : 12 (0.26%)
    감정/멍함 : 12 (0.26%)
    감정/즐거움 : 11 (0.24%)
    치료이력/응급실 : 11 (0.24%)
    증상/힘빠짐 : 11 (0.24%)
    감정/의기소침 : 11 (0.24%)
    감정/고독감 : 11 (0.24%)
    증상/과수면 : 10 (0.22%)
    배경/이혼 : 10 (0.22%)
    현재상태/증상악화 : 10 (0.22%)
    감정/무미건조 : 10 (0.22%)
    증상/알코올의존 : 10 (0.22%)
    감정/통제력상실 : 9 (0.19%)
    증상/환각 : 9 (0.19%)
    배경/이사 : 9 (0.19%)
    배경/아르바이트 : 9 (0.19%)
    증상/건강염려 : 9 (0.19%)
    증상/소화불량 : 9 (0.19%)
    감정/불편감 : 8 (0.17%)
    감정/좌절 : 8 (0.17%)
    감정/공허감 : 8 (0.17%)
    감정/당황 : 8 (0.17%)
    증상/이인감 : 8 (0.17%)
    감정/불신 : 8 (0.17%)
    증상/컨디션저조 : 8 (0.17%)
    감정/미움 : 8 (0.17%)
    증상/만성피로 : 8 (0.17%)
    감정/미안함 : 8 (0.17%)
    배경/유학 : 8 (0.17%)
    감정/무력감 : 7 (0.15%)
    증상/생리불순 : 7 (0.15%)
    배경/타인 : 7 (0.15%)
    내원이유/치료 : 7 (0.15%)
    증상/과대망상 : 7 (0.15%)
    감정/예민함 : 7 (0.15%)
    배경/육아 : 7 (0.15%)
    증상/메스꺼움 : 7 (0.15%)
    내원이유/의사소견 : 7 (0.15%)
    증상/편두통 : 7 (0.15%)
    배경/전연인 : 6 (0.13%)
    배경/종교 : 6 (0.13%)
    감정/배신감 : 6 (0.13%)
    배경/귀국 : 6 (0.13%)
    증상/대화기피 : 6 (0.13%)
    현재상태/증상지속 : 6 (0.13%)
    증상/성욕상승 : 6 (0.13%)
    증상/가슴통증 : 6 (0.13%)
    증상/신체이상 : 6 (0.13%)
    현재상태/증상감소 : 6 (0.13%)
    감정/과민반응 : 6 (0.13%)
    감정/죄책감 : 6 (0.13%)
    증상/발작 : 6 (0.13%)
    증상/인지기능저하 : 6 (0.13%)
    상태/증상감소 : 6 (0.13%)
    감정/창피함 : 6 (0.13%)
    증상/속쓰림 : 6 (0.13%)
    감정/초조함 : 6 (0.13%)
    배경/군대 : 5 (0.11%)
    증상/체력저하 : 5 (0.11%)
    증상/공격적성향 : 5 (0.11%)
    증상/시력저하 : 5 (0.11%)
    감정/기시감 : 5 (0.11%)
    자가치료/운동 : 5 (0.11%)
    자가치료/충분한휴식 : 5 (0.11%)
    증상/저림현상 : 5 (0.11%)
    증상/성격변화 : 5 (0.11%)
    증상/떨림 : 5 (0.11%)
    감정/곤혹감 : 5 (0.11%)
    원인/없음 : 5 (0.11%)
    

최종 176개에 해당하는 label값을 가지고 있음을 보였다.
또한 하위 1% 미만의 하위갯수분포를 보이는 label 갯수도 상당하다는 것을 알 수 있다.

## 감성분석 모델링 하기  

### step01. 학습대상이 되는 콘텐츠에 대하여 정수값으로 벡터화 시킨다.  
__이때, 기존 meta 단어(토큰)에서 없는 부분을 확장하여 new_meta를 만들고__  
__이때, 새로 ebedding 계층부터 학습하지 않고 기존에 활용했던, ebedding 객체를 소환하여 활용한다.__


```python
print(len(train),len(test))
train[11]
```

    4651 400
    




    ['Sent_1160', '죽는 게 나을 것 같다는 생각이 들어.', '감정/자살충동', '감정/자살충동']




```python
# 학습 완료된 임베딩 저장하기 -> colab 불러오기
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/vecs.tsv") as f:
    vecs = [v.strip() for v in f.readlines()]
```


```python
## 해당 vecs 에 해당하는 원래 단어사전 (형태소 형태로 분해된) 불러오기.
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/meta.tsv") as m:
    meta = [v.strip() for v in m.readlines()]
```


```python
final_embeddings = [np.float32(v.split("\t")) for v in vecs]
final_embeddings = np.array(final_embeddings)
final_embeddings.shape
```




    (70002, 128)



기존 embedding 단어사전에서 포함하지 못하는 단어들을 파악하여 이들을 처리해야 합니다.  
기존단어사전 : meta  
신규단어사전 : new_meta (이 부분은 colab에서 수행됨)


```python
print(len(meta))
```

    70002
    

단, 여기서 local로 진행하기에는 문제가 생깁니다.(필자의 local에는 konlp가 설치되어 있지 않고, 그동안 Colab에서 수행해 왔음)  

기존 meta 에서 커버하지 못하는 단어(tokenizer된)들을 파악하기 위해서는,  
1) konlpy - Komoran() 으로 기존 문장을 한국어 토큰화 시키고  
2) 기존 meta 파일과 비교하여 신규 단어를 파악  
3) 신규단어들만큼 추가학습을 하여 embedding를 새로 만들거나 or 신규단어 부분만 0인 값으로 embedding에 추가 배분  

상기 과정을 거쳐야 합니다. 하지만, 저는 상기 과정을 colab에서 수행했습니다.  
[colab 수행과정](https://github.com/cypision/Alchemy-in-MLDL/blob/master/word_embedding_add_oob_Word.ipynb)  


```python
oov_counter = collections.Counter()
```

train,test 데이터를 meta 정보에 맞추어 토큰화 -> 정수인덱싱 한 데이터를 불러온다.


```python
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/new_meta.tsv") as m:
    new_meta = [v.strip() for v in m.readlines()]
```


```python
train_ids = np.load('D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/Wellness_data_train_tokenized.npy')
train_labels = np.load('D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/Wellness_data_train_tokenized_label.npy')
test_ids = np.load('D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/Wellness_data_test_tokenized.npy')
test_labels = np.load('D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/Wellness_data_test_tokenized_label.npy')
```


```python
# Wellness_data_label_map.json
with open("D:/★2020_ML_DL_Project/Alchemy/dataset/text_output/Wellness_data_label_map.json" , 'r') as f:
    label_map = json.loads(f.read())
```


```python
print(len(train),len(test))
train[11]
```

    4651 400
    




    ['Sent_1160', '죽는 게 나을 것 같다는 생각이 들어.', '감정/자살충동', '감정/자살충동']




```python
print(train_ids.shape,"\t test_ids.shape:",test_ids.shape)
print(train_ids[11])
```

    (4651, 50) 	 test_ids.shape: (400, 50)
    [330   6  40  75   7  35  82  77 180   3  23  26   4   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0]
    

train 4651 -> (4651,50)  
test   400 -> (400,50)  
으로 변경되었다. 이 과정은 colab 링크주소를 통해서 보면 좀더 확인가능합니다.[colab 수행과정](https://github.com/cypision/Alchemy-in-MLDL/blob/master/word_embedding_add_oob_Word.ipynb)  
간단히 요약하면,  
1) train,test 내의 setence 데이터를 tokenize(한글)  
2) 기존 meta(단어사전)에 대입하여 new_meta로 확장.(기존 meta 단어장에 없는 token이 있기 때문)  
3) new_meta에 따른 정수 인덱스 sentence로 변경하고 padding 을 줘서 각 문장별 setence 길이를 mat_len = 50 기준으로 맞춤  

### step02. 모델을 설계하고 Embedding layer를 수정한다.
__이때, 새로 ebedding 계층부터 학습하지 않고 기존에 활용했던, ebedding 객체를 소환하여 활용한다.__


```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
```


```python
print(final_embeddings.shape)
print(type(label_map),len(label_map))

label_map_reverse = {}
for key,val in label_map.items():
    label_map_reverse[val] = key
```

    (70002, 128)
    <class 'dict'> 176
    


```python
vocab_size = len(new_meta) # 단어사전 개수
embedding_dim = final_embeddings.shape[1] # 임베딩 차원. 여기선 128차원
rnn_hidden_dim = 300 # GRU hidden_size
final_dim = len(label_map) ## 176

""" MAKE MODEL """
model = Sequential(
    [Embedding(vocab_size, embedding_dim, mask_zero=True), ## mask_zero = tf.keras.preprocessing.sequence.pad_sequences 를 통해 input 값들의 길이가 이미 같음을 알림
     GRU(rnn_hidden_dim), ## 
     Dense(rnn_hidden_dim, activation= "relu"),
     Dropout(0.3),
     Dense(final_dim, activation="softmax")] ## final_dim=176 개의 감정분석 label에 대한 softmax 를 적용하여 다중 클래스피케이션으로 모델을 설계한다.
)
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, None, 128)         9020544   
    _________________________________________________________________
    gru (GRU)                    (None, 300)               387000    
    _________________________________________________________________
    dense (Dense)                (None, 300)               90300     
    _________________________________________________________________
    dropout (Dropout)            (None, 300)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 176)               52976     
    =================================================================
    Total params: 9,550,820
    Trainable params: 9,550,820
    Non-trainable params: 0
    _________________________________________________________________
    


```python
for i in range(len(model.get_weights())):
    print("{}번째 가중치 행렬의 shape:{}".format(i, model.get_weights()[i].shape))
```

    0번째 가중치 행렬의 shape:(70473, 128)
    1번째 가중치 행렬의 shape:(128, 900)
    2번째 가중치 행렬의 shape:(300, 900)
    3번째 가중치 행렬의 shape:(2, 900)
    4번째 가중치 행렬의 shape:(300, 300)
    5번째 가중치 행렬의 shape:(300,)
    6번째 가중치 행렬의 shape:(300, 176)
    7번째 가중치 행렬의 shape:(176,)
    


```python
print(len(model.get_weights()))
print(model.get_weights()[0].shape)
```

    8
    (70473, 128)
    

앞서 얘기했다시피, 기존에 wiki 사전으로 학습시킨, embedding(final_embeddings) 을 재활용할 수 있습니다.  
<span style='color:red'>__단어를 CBOW나 skip-gram으로 embedding 가중치행렬을 다시 구할수도 있지만, 단지 학습단계에서 초기값으로만 활용해도 효과가 있다.__</span>  
단 사용시에 기존 배열과의 차원수를 잘 맞춰줘야 합니다.  

본 예에서는  
Embedding(vocab_size, embedding_dim, mask_zero=True) 으로 보여지다시피, vocab_size(70473)이 들어갔습니다.  
이는 input으로 값들의 토큰단어수준 크가가 70743개란 뜻입니다. 하지만 우리가 과거에 학습한 final_embeddings 때의 토큰단어사전 갯수는 70002 였습니다.  

위에서 0번째 가중치 행렬의 shape:(70473, 128) 에 초기값으로 final_embeddings(70002, 128)를 넣어주려고 하는데 행이 맞지 않으니, 이를 0으로 채워서 새롭게 embedding 행렬초기값을 만들어 덮어써줍니다. 


```python
print(vocab_size,final_embeddings.shape)
```

    70473 (70002, 128)
    


```python
## 단어사전 개수 체크
org_vocab_size = final_embeddings.shape[0] ## 70002
new_vocab_size = len(new_meta)       ## 70473
 
print("CBOW initialize될 토큰 개수:", org_vocab_size)
print("새로운 임베딩의 one-hot-vector:", new_vocab_size, "\n")
print("-> 랜덤 초기화해야 할 벡터 차원: {} x {}".format(new_vocab_size-org_vocab_size, embedding_dim))
```

    CBOW initialize될 토큰 개수: 70002
    새로운 임베딩의 one-hot-vector: 70473 
    
    -> 랜덤 초기화해야 할 벡터 차원: 471 x 128
    


```python
rand_initial = np.random.uniform(-1,1,size=[vocab_size-org_vocab_size,embedding_dim])
rand_initial.shape
```




    (471, 128)




```python
initial_weight = np.append(final_embeddings, rand_initial, axis = 0)
initial_weight.shape
```




    (70473, 128)



model.weights[0] 바꿔끼우기


```python
model.weights[0].assign(initial_weight) # model.weights[0] -> 임베딩 레이어에 해당
model.get_weights()[0]
```




    array([[ 3.65821011e-02,  2.09269263e-02,  4.37952392e-02, ...,
            -4.68397848e-02, -3.90023366e-02,  7.83827156e-03],
           [-1.08658604e-01,  1.20607175e-01,  2.56893903e-01, ...,
             2.73515940e-01, -1.67477235e-01,  1.85633793e-01],
           [-1.63065505e+00,  4.06452082e-02,  5.75998187e-01, ...,
            -7.51210332e-01, -2.29754075e-01,  3.93116146e-01],
           ...,
           [-4.35010314e-01, -3.36897731e-01,  2.59104937e-01, ...,
            -2.06732866e-03,  2.52970278e-01,  1.95310205e-01],
           [-9.23107386e-01,  8.29408407e-01,  9.49173152e-01, ...,
            -4.46789980e-01,  1.64622396e-01, -7.64461696e-01],
           [-7.53726959e-01, -7.91572854e-02, -1.17686565e-03, ...,
            -4.95805442e-02,  8.89336020e-02,  3.09113473e-01]], dtype=float32)



### step03. 모델을 compile 이후 학습


```python
## 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

train set 에서, validation set를 분리해낸다.


```python
from sklearn.model_selection import train_test_split
train_ids, val_ids, train_labels, val_labels = train_test_split(train_ids, train_labels , test_size=0.10, random_state=42, stratify=train_labels)
```


```python
## 모델 학습
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1)

num_epochs = 100
history = model.fit(train_ids, train_labels, epochs=num_epochs, batch_size=200,validation_data=(val_ids, val_labels), callbacks=[callback])
```

    Train on 4185 samples, validate on 466 samples
    Epoch 1/100
    4185/4185 [==============================] - 2s 423us/sample - loss: 3.4095 - accuracy: 0.2461 - val_loss: 3.3772 - val_accuracy: 0.2489
    Epoch 2/100
    4185/4185 [==============================] - 1s 215us/sample - loss: 3.2400 - accuracy: 0.2631 - val_loss: 3.2990 - val_accuracy: 0.2575
    Epoch 3/100
    4185/4185 [==============================] - 1s 215us/sample - loss: 3.0403 - accuracy: 0.2984 - val_loss: 3.2013 - val_accuracy: 0.2811
    Epoch 4/100
    4185/4185 [==============================] - 1s 212us/sample - loss: 2.8790 - accuracy: 0.3245 - val_loss: 3.1715 - val_accuracy: 0.2768
    Epoch 5/100
    4185/4185 [==============================] - 1s 212us/sample - loss: 2.6902 - accuracy: 0.3639 - val_loss: 3.1705 - val_accuracy: 0.2983
    Epoch 6/100
    4185/4185 [==============================] - 1s 215us/sample - loss: 2.5241 - accuracy: 0.3845 - val_loss: 3.0713 - val_accuracy: 0.3004
    Epoch 7/100
    4185/4185 [==============================] - 1s 213us/sample - loss: 2.3434 - accuracy: 0.4146 - val_loss: 3.1096 - val_accuracy: 0.3069
    

### step04. test 데이터로 성능평가하기


```python
## sample
scores = model.predict(test_ids)
```


```python
scores.shape
```




    (400, 176)




```python
## 176 차원으로 값을 받았으니, 이중 가장 확률적으로 높은 값을 return한 index를 찾는다.
print(scores[0].shape)
print(np.argmax(scores[0]))
```

    (176,)
    22
    

22 index 가 가장 높은 확률을 보였다.


```python
print(scores[0][22])
print(label_map_reverse[test_labels[22]])
```

    0.08855945
    증상/불면
    

이를 함수로 나타내면 하기와 같다.


```python
def make_prediction(test_ids):
    # model.predict 함수를 통해 확률값 받아오기
    scores = model.predict(test_ids)
    # 확률값이 가장 높은 카테고리로 분류하기
    predictions = np.argmax(scores, axis=1) ## index 값을 return 한다.
    return scores , predictions 
```


```python
scores, predictions = make_prediction(test_ids)
```

이를 확인하는 함수로 구현하면 하기와 같다.


```python
def SCORE(predictions, ground_truth):
    print("TEST SET ACCURACY: {:.2f}".format(sum(predictions == ground_truth) / len(predictions)))
    print("-"*80)
    label_reverse = {v:k for k, v in label_map.items()}
    for i in range(10):
        if predictions[i] != ground_truth[i]:
            print("🥺: {}".format(test[i][1]))
            print("-> 👩‍⚕️: {} 🤖: {}".format( label_reverse[ground_truth[i]], label_reverse[predictions[i]]), "\n")   
```


```python
SCORE(predictions, test_labels)
```

    TEST SET ACCURACY: 0.24
    --------------------------------------------------------------------------------
    🥺: 저는 이제 망했어요…
    -> 👩‍⚕️: 감정/좌절 🤖: 배경/부모 
    
    🥺: 스테로이드를 먹으니까 불면이 더 심해진 것 같아.
    -> 👩‍⚕️: 증상/불면 🤖: 현재상태/증상악화 
    
    🥺: 맛있는 거 먹으면 괜찮아졌는데, 요즘은 아니에요.
    -> 👩‍⚕️: 증상/식욕저하 🤖: 배경/건강문제 
    
    🥺: 이상하게 사고도 자꾸 생기는 것 같고…
    -> 👩‍⚕️: 배경/사고 🤖: 증상/무기력 
    
    🥺: 수술 끝나고 항암 치료 진행 중이에요.
    -> 👩‍⚕️: 배경/건강문제 🤖: 배경/직장 
    
    🥺: 나도 근속을 좀 해보고 싶다.
    -> 👩‍⚕️: 배경/직장 🤖: 증상/피해망상 
    
    🥺: 2년정도 지나니까 빈털터리가 되있었어요.
    -> 👩‍⚕️: 배경/경제적문제 🤖: 배경/어린시절 
    
    🥺: 근데 의사가 그 외에 또 뭐 없는지 자세하게 말해달라고 하는 거야.
    -> 👩‍⚕️: 치료이력/병원내원 🤖: 치료이력/검사 
    
    🥺: 안심이 안 된다…
    -> 👩‍⚕️: 감정/불안감 🤖: 감정/긴장 
    
    

이상으로 포스팅을 마칩니다. 감사합니다.


```python

```
