---
title:  'Crawling 연습'
excerpt: 'Inflearn 강좌 따라하기 1'

categories:
  - Machine-Learning
tags:
  - ML
  - Crawling
  - text anlysis
  - 머신러닝
last_modified_at: 2019-04-13T08:06:00-05:00
---

## Crawling 하기 Post

    ## 인프런 2020년 새해 다짐 이벤트 댓글 크롤링  
    * https://www.inflearn.com/pages/newyear-event-20200102  

    ## 필요한 라이브러리 설치
    * 아나콘다 사용시 다음의 프롬프트 창을 열어 conda 명령어로 설치합니다.
    * pip 사용시 아래에 있는 명령어를 터미널로 설치합니다.
    <img src="https://i.imgur.com/Sar4gdw.jpg">
    ### BeautifulSoup
    * `conda install -c anaconda beautifulsoup4`
    * [Beautifulsoup4 :: Anaconda Cloud](https://anaconda.org/anaconda/beautifulsoup4)
    * pip 사용시 : `pip install beautifulsoup4`

    ### tqdm
    * `conda install -c conda-forge tqdm`
    * [tqdm/tqdm: A Fast, Extensible Progress Bar for Python and CLI](https://github.com/tqdm/tqdm)
    * `pip install tqdm`

# 본 연습은 inflearn - 박조은 강사님의 Code를 활용하여, 변경함


```python
# !pip install tqdm
# !pip install beautifulsoup4
# !pip install requests
```


```python
pwd
```




    'D:\\★2020_ML_DL_Project\\Alchemy\\ML_Area'




```python
# 라이브러리 로드
# requests는 작은 웹브라우저로 웹사이트 내용을 가져온다.
import requests
# BeautifulSoup 을 통해 읽어 온 웹페이지를 파싱한다.
from bs4 import BeautifulSoup as bs
# 크롤링 후 결과를 데이터프레임 형태로 보기 위해 불러온다.
import pandas as pd
import numpy as np
from tqdm import trange
```


```python
# 크롤링 할 사이트
base_url = "https://www.inflearn.com/pages/newyear-event-20200102"
# base_url = "http://www.todayhumor.co.kr/board/view.php?table=sisa&no=1148601&s_no=1148601&page=4"
response = requests.get( base_url )
# response.text
```


```python
soup = bs(response.text, 'html.parser')
```


```python
## 사실상, html 의 selector 를 가져오는게 품이 많이 드는 작업임.
content = soup.select("#main > section > div > div > div.chitchats > div.chitchat-list > div")
content[-1]
```




    <div class="chitchat-item">
    <figure class="image is-48x48">
    <img alt="인프런" class="user_thumb is-rounded" src="https://cdn.inflearn.com/wp-content/uploads/avatars/17/b415d9fa24d186c4adf22ca9a49116b5-bpfull.png"/>
    </figure>
    <div class="content">
    <div class="author">
    <span class="author_name">인프런</span>
    <time class="created_at" datetime="Sun Dec 29 2019 17:59:00 GMT+0900 (GMT+09:00)">⋅ 약 1개월 전</time>
    <a class="update-chitchat edit-chitchat is-hidden" data-id="18594" type="button">저장</a>
    <a class="hidden-editor edit-chitchat is-hidden" type="button">취소</a>
    <a class="edit-chitchat no_cmt_reply" type="button">답글달기</a>
    </div>
    <div class="body edit-chitchat">인프런 0호 팀원이에요!
    그동안 서비스 개발 때문에 js 를 많이 했었는데 앞으론 통계나 분석을 많이 하고 싶어서 파이썬을 공부하고 싶어요! 올해 파이썬 마스터가 되는걸로..
    #관심강의: 남박사의 파이썬 활용</div>
    <textarea class="textarea edit-chitchat is-hidden"></textarea>
    <div class="summary_comments">
    </div>
    <div class="chitchats_and_editor is-hidden">
    <div class="chitchat-comment-list">
    </div>
    <div class="cmt-editor">
    <div class="field">
    <textarea class="textarea" placeholder="내용을 입력해 주세요."></textarea>
    </div>
    <button class="insert-chitchat-cmt button" data-post_id="18594" type="button">등록</button>
    </div>
    </div>
    </div>
    </div>




```python
content[-1].select("div.body.edit-chitchat")[0].get_text(strip=True)
```




    '인프런 0호 팀원이에요!\n그동안 서비스 개발 때문에 js 를 많이 했었는데 앞으론 통계나 분석을 많이 하고 싶어서 파이썬을 공부하고 싶어요! 올해 파이썬 마스터가 되는걸로..\n#관심강의: 남박사의 파이썬 활용'




```python
chitchat = content[-1].select("div.body.edit-chitchat")[0].get_text(strip=True)
chitchat
```




    '인프런 0호 팀원이에요!\n그동안 서비스 개발 때문에 js 를 많이 했었는데 앞으론 통계나 분석을 많이 하고 싶어서 파이썬을 공부하고 싶어요! 올해 파이썬 마스터가 되는걸로..\n#관심강의: 남박사의 파이썬 활용'




```python
## 대략 잘 나오는지 5개만 for문 돌려서 확인하고
events = []
for i in range(5):
    print("-"*20)
    chitchat = content[i].select("div.body.edit-chitchat")[0].get_text(strip=True)
    print(chitchat)
    events.append(chitchat)
```

    --------------------
    2020년에는 데이터를 좀 더 열심히 공부하려고 합니다.
    --------------------
    2020년 목표 - 안주하지 않기
    --------------------
    자바 공부 마스터 하고 싶습니다.   더 자바, 코드를 조작하는 다양한 방법
    --------------------
    파이썬 데이터시각화 분석 실전 프로젝트 수강하고 싶어요
    --------------------
    2020년도 화이팅!, 스프링 프레임워크 개발자를 위한 실습을 통한 입문 과정



```python
content_count = len(content)
content_count
```




    2435




```python
## 실제로 모든 이벤트를 긁어 모은다
## from tqdm import trange 의 trange 는 time 을 for 문 안에서 돌때, 시간을 표시해주는 역할을 한다.
events = []
for i in trange(content_count):
    chitchat = content[i].select("div.body.edit-chitchat")[0].get_text(strip=True)
    events.append(chitchat)
```

    100%|████████████████████████████████████████████████████████████████████████████| 2435/2435 [00:00<00:00, 4306.02it/s]



```python
df = pd.DataFrame({"text": events})
print(df.shape)
df.head()
```

    (2435, 1)





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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020년에는 데이터를 좀 더 열심히 공부하려고 합니다.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020년 목표 - 안주하지 않기</td>
    </tr>
    <tr>
      <th>2</th>
      <td>자바 공부 마스터 하고 싶습니다.   더 자바, 코드를 조작하는 다양한 방법</td>
    </tr>
    <tr>
      <th>3</th>
      <td>파이썬 데이터시각화 분석 실전 프로젝트 수강하고 싶어요</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020년도 화이팅!, 스프링 프레임워크 개발자를 위한 실습을 통한 입문 과정</td>
    </tr>
  </tbody>
</table>
</div>




```python
pwd
```




    'D:\\★2020_ML_DL_Project\\Alchemy\\ML_Area'




```python
df.to_csv("data_source/inflearn-event.csv", index=False)
```


```python
pd.read_csv("D:\\★2020_ML_DL_Project\\Alchemy\\ML_Area\\data_source\\inflearn-event.csv").head()
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
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020년에는 데이터를 좀 더 열심히 공부하려고 합니다.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020년 목표 - 안주하지 않기</td>
    </tr>
    <tr>
      <th>2</th>
      <td>자바 공부 마스터 하고 싶습니다.   더 자바, 코드를 조작하는 다양한 방법</td>
    </tr>
    <tr>
      <th>3</th>
      <td>파이썬 데이터시각화 분석 실전 프로젝트 수강하고 싶어요</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020년도 화이팅!, 스프링 프레임워크 개발자를 위한 실습을 통한 입문 과정</td>
    </tr>
  </tbody>
</table>
</div>


