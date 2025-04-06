# 자연어처리 과제_기계번역 - 결과 보고서

|학번|20200232|이름|김승민|
|:---:|:---:|:---:|:---|
|학년|4학년|제출일|2025.04.05|
|제목|기계번역 모델 간의 성능비교|||

## [실험내용]
## 실험 1
|실험요소|설명|
|:---:|:---|
|참고자료|wikidocs 딥 러닝을 이용한 자연어 처리 입문 / 16. [NLP 고급] 시퀀스투시퀀스(Sequence-to-Sequence, seq2seq) / 16-02 Seq2Seq를 이용한 번역기 구현하기 : https://wikidocs.net/216494|
|데이터|프랑스어-영어 병렬 코퍼스 : http://www.manythings.org/anki|
|모델|Seq2Seq|
### [모델 : Seq2Seq]
|구조|설명|
|:---:|:---|
|프레임워크|PyTorch|
|데이터 수|33,000|
|모델 구조|Encoder-Decoder 구조|
|Encoder 구성|임베딩 레이어, LSTM 레이어|
|Decoder 구성|임베딩 레이어, LSTM 레이어, fc레이어|
### [학습과정]
|학습 정보|설명|
|:---:|:---|
|epoch|30|
|손실 함수|CrossEntrophy|
|옵티마이저|Adam|
---
## 실험 2
|실험요소|설명|
|:---:|:---|
|참고자료|wikidocs 딥 러닝을 이용한 자연어 처리 입문 / 16. [NLP 고급] 어텐션 메커니즘 / 17-0 어텐션 메커니즘을 이용한 번역기 구현하기 : https://wikidocs.net/216495|
|데이터|프랑스어-영어 병렬 코퍼스 : http://www.manythings.org/anki|
|모델|Seq2Seq + Attention 메커니즘|
### [모델 : Seq2Seq]
|구조|설명|
|:---:|:---|
|프레임워크|PyTorch|
|데이터 수|33,000|
|모델 구조|Encoder-Decoder 구조|
|Encoder 구성|임베딩 레이어, LSTM 레이어|
|Decoder 구성|임베딩 레이어, LSTM 레이어, fc레이어, attention 메커니즘|
### [학습과정]
|학습 정보|설명|
|:---:|:---|
|epoch|30|
|손실 함수|CrossEntrophy|
|옵티마이저|Adam|
---
## 성능평가 (BLEU 수치 사용)
### BLEU 점수(Bilingual Evaluation Understudy)는 0에서 1 사이의 값을 가진다.
- 1에 가까울수록 → 예측 번역이 참조 문장(정답)과 아주 유사하다는 의미
- 0에 가까울수록 → 예측 번역이 참조 문장과 거의 유사하지 않다는 의미

## Seq2Seq 모델 성능 평가 결과
#### attention메커니즘이 들어가지 않은 seq2seq모델
    Train BLEU: 0.1762 | Valid BLEU: 0.0779
#### attention메커니즘이 들어간 seq2seq모델
    Train BLEU: 0.1794 | Valid BLEU: 0.0801
---
## 분석
처음 결과를 보고 너무 낮은 것이 아닌가 하는 생각이 들었다. 아무리 0에서 1사이의 값이라도 0.1은 너무 작은 것이 아닌가 했다.\
하지만 Chat-gpt4의 BLEU score는 0.88이라는 것을 생각하면 정상적인 수치인 것으로 생각이 되었다.\
여기서 관심을 가져야 할 점은 0.1이라는 점수가 너무 낮은 점수가 아닌가에 대한 것이 아니라 attention메커니즘이 들어갔을 때, 유의미한 점수 변화가 있다는 것이다.
