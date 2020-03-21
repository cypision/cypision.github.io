---
title:  "KoNLP 설치하기 ubuntu 18.04 에서"
excerpt: "Window 10 OS 다 보니...Docker로 해보기"

categories:
  - Machine-Learning
tags:
  - Docker
  - text anlysis
  - KoNLPy
last_modified_at: 2020-03-21T16:13:00-05:00
---

## 텍스트 분석하다가. KoNLPy 설치하려고 하는 Docker 활용 정리

### 일단, Docker 로 Ununtu 18.04 를 만든다.

1. apt-get update 실행 : 그래야 기본적인것을 설치한다. ㅜ.ㅜ  
2. apt-get install sudo : 계정만들고, sudo 권한 부여  
 - sudo useradd -m user01
 - sudo passwd user01 
 
3. 계정의 경로 표시 (root) 유저기준
 - vi /root/.bashrc # .bashrc 파일의 #force_color_prompt=yes 를 풀었더니, 가능해졌다.  
 
4. apt-get install vim : vi 에디터 설치
 - vi /etc/sudoers  :  sudo 권한부여  
 * 헌데,  신규유저는 영 추가설정이 많아서...그냥 root 권한으로 한다. *
 
5. python 설치 (https://phoenixnap.com/kb/how-to-install-python-3-ubuntu)
 - sudo apt install software-properties-common
 - sudo add-apt-repository ppa:deadsnakes/ppa
 - sudo apt update
 - sudo apt install python3.7
 - python3 ––version : 설치확인 과정
6. python을 위한 PIP 설치 (https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
 - sudo apt update
 - sudo apt install python3-pip : python 기반으로 pip 설치
 - pip3 --version : 제대로 설치되었는지 확인하기  
   (필요에 따라서, python3 나 pip3 로 구현이 된다면,  심볼릭 링크로 변경해도 좋다)  
 - pip install --upgrade pip : pip 같은 것은 왠만하면 update 하는게 좋다.

7. tmux 설치 : 이는 jupyterlab을 tmux 백그라운드 세션으로 띄워두는 것이 편해서이다.
 - sudo apt-get install tmux
 - tmux 실행
   > tmux new -s [name]
   > ctrl + b + d : 백그라운드로 두고 빠져나오기


8. python3로 가상환경 꾸미기 (https://dgkim5360.tistory.com/entry/python-virtualenv-on-linux-ubuntu-and-windows)
 - pip install virtualenv  
 - virtualenv jupyter_lab : "jupyter_lab" 으로 가상환경만들기  
 - python -m virtualenv jupyter_lab : user로 할때는 이걸로 설정했다 
 > 가상환경 실행 : source jupyter_lab/bin/activate ( 단 필자는 tmux 세션을 따서, 사용했다)

여기서부터 본인은 tmux 세션을 별도로 따서, 실행했다.  
  
=======================__tmxu session 내부__========================  
이후 필요한 것들은 가상환경 내에서, 설치하면 된다. 만약 jupyterlab 을 설치한다면, 

1) source jupyter_lab/bin/activate = source .bashrc : .bashrc 명령어 끝에 내가 넣었다....편하게 쓰려고  
2) pip install jupyterlab  
3) jupyter notebook --generate-config

출처: https://goodtogreate.tistory.com/entry/IPython-Notebook-설치방법 [GOOD to GREAT]
4) 원격 실행을 위해서는 Ipython 도 있어야 한다. : Ipython 으로 원격접속시, passwd 설정함
 - pip install ipython[all]  
 4-1) ipython  
 4-2) from notebook.auth import passwd  
 4-3) passwd() : 비밀번호 입력...난 cypision  -> quit()  
    --> 여기서 주는 sha key를 반드시 저장해야 한다.  
    
 5) vi .jupyter/jupyter_notebook_config.py  : 이후 ipython 에서 빠져나와서, config 파일을 만든다
    vi .jupyter_lab/jupyter_notebook_config.py  : 이후 ipython 에서 빠져나와서, config 파일을 만든다
 (https://goodtogreate.tistory.com/entry/IPython-Notebook-%EC%84%A4%EC%B9%98%EB%B0%A9%EB%B2%95)  
   'sha1:82b442a7823b:4910f92e9aca49069dd1456eae23e67458a97134'  
막상실행하면, 무수한 에러를 만난다
- OSError: [Errno 99] Cannot assign requested address
> (https://austcoconut.tistory.com/entry/Bug-Report-Linux%ED%99%98%EA%B2%BD%EC%97%90%EC%84%9C-Jupyter-Notebook-%EC%8B%A4%ED%96%89-%EC%8B%9C-OSError-Errno-99-Cannot-assign-requested-address)  

> jupyter lab --ip=0.0.0.0 --port=8888 --allow-root  : 바로 실행하려면, 이 명령어를 사용한다. 그러나, 이게 외부서버라면, 반드시 config 파일을 생성후 수정해야 한다.
  만약 상기명령이 귀찮다면, jupyter notebook 의 config 파일을 찾아서 변경해주면 된다.  
  
=======================__tmxu session 내부__========================

**참조 : jupyter lab config**  
    c = get_config()  
    c.NotebookApp.password = u'sha1:302e7676d240:64c5ce60262844c55d63ada4d0c4b723bfca20f1'  
    #c.NotebookApp.cerfile = u'/home/root/mycert.pem'  
    c.NotebookApp.open_browser = False  
    c.NotebookApp.ip = '*'  
    #c.NotebookApp.port_retries = 9999  
    c.NotebookApp.port = 9999  
    c.NotebookApp.browser = u'/Applications/Gooogle\ Chrome.app %s'

**참조 : 최종 docker run 실행시 예시**  
$ docker container run -it --name jupyter_lab_jjh -p 9999:9999 -v /Alchemy:/home/cypision/Alchemy --privileged cypision:2.0 /bin/bash  
상기 방법 외에도, docker_file 로 container를 만들 수도 있다.

**참조 : python 기본 liabrary 설치**  
pip install pandas  
pip install numpy  
pip install -U scikit-learn  
pip install regex  
python -m pip install -U matplotlib  
pip install seaborn  

**참조 : KoNLP 설치**  
$ sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl : Install Java 1.8 or up  

$ python3 -m pip install konlpy  

$ sudo apt-get install curl git : Mecab 설치 필요시 시행  

$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)  

__참조 : 간간히 적는 Docker 명령어 메모__

**컨테이너 빠져 나오기**
1) 컨테이너에서 빠져나오는 방법은 두 가지가 있습니다. 첫번째로는 컨테이너를 종료하면서 빠져나오는 방법입니다.  
  -> __exit 또는 Ctrl + D__  
2) 두번째로는 컨테이너는 가동되는 상태에서 접속만 종료하는 방법입니다.  
  -> __Ctrl + P 입력 후 Ctrl + Q 입력__  
3) docker 실행중 상태에서, image 생성 (not using dockerfile)
 -> docker commit -a 'CheongJinHwan' -m "python_3.7_update" a6c7e632076e cypision:1.0  
출처: https://www.44bits.io/ko/post/how-docker-image-work  
출처: https://www.bsidesoft.com/?p=7851  
출처: https://m.blog.naver.com/alice_k106/220340499760  
출처: https://dololak.tistory.com/376 [코끼리를 냉장고에 넣는 방법]  

__참조: vim editor 편집기  __  
(https://wayhome25.github.io/etc/2017/03/27/vi/)

**참조 : Docker에서 mount 하기. 공유폴더**  
필자는 win10 이지만, Home edition 의 한계로, VirtualBox 를 활용 - Docker ToolBox 를 사용한다.  
분명히 로컬에서는 폴더내에 파일이 있으나, docker 를 막상띄우면 폴더가 빈 폴더라서, 하루 종일 헤매었다.  
원인은 경로 오류!!  

__Local directory - Oracle Virtual Machine - Docker container__

상기로 연결이 되기 때문에,  (이는 dick mount 뿐만 아니라, ip port 설정에도 동일하다.)
1) 반드시 Local 과 OVM 사이의 폴더를 먼저 mount 하고,   
2) OVM 상의 경로와 Docker container 를 "-v" 옵션으로 경로를 정해야 한다.







