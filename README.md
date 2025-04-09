# 자연어처리 과제_기계번역 - 결과 보고서

|학번|20200232|이름|김승민|
|:---:|:---:|:---:|:---|
|학년|4학년|제출일|2025.04.10|
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
|참고자료|wikidocs 딥 러닝을 이용한 자연어 처리 입문 / 17. [NLP 고급] 어텐션 메커니즘 / 17-03 어텐션 메커니즘을 이용한 번역기 구현하기 : https://wikidocs.net/216495|
|데이터|프랑스어-영어 병렬 코퍼스 : http://www.manythings.org/anki|
|모델|Seq2Seq + Attention 메커니즘|
### [모델 : Seq2Seq]
|구조|설명|
|:---:|:---|
|프레임워크|PyTorch|
|모델 구조|Transformer (Encoder-Decoder)|
|Encoder 구성|임베딩 레이어, LSTM 레이어|
|Decoder 구성|임베딩 레이어, LSTM 레이어, fc레이어, attention 메커니즘|
### [학습과정]
|학습 정보|설명|
|:---:|:---|
|epoch|30|
|손실 함수|CrossEntrophy|
|옵티마이저|Adam|
---
## 실험 3
|실험요소|설명|
|:---:|:---|
|참고자료|nn.Transformer 와 torchtext로 언어 번역하기 : https://tutorials.pytorch.kr/beginner/translation_transformer.html|
|데이터|torchtext 라이브러리의 Multi30k 데이터셋|
|데이터 구조|train: 29000, valid: 1014, test: 1000|
|모델|Transformer|
### [모델 : Transformer]
|구조|설명|
|:---:|:---|
|프레임워크|PyTorch|
|모델 구조|Transformer|
|Encoder 구성|입력 토큰 임베딩 : self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)<br>위치 인코딩 : self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)<br>Transformer 인코더 레이어 : self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)|
|Decoder 구성|출력 토큰 임베딩	: self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)<br>위치 인코딩 : self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)<br>Transformer 디코더 레이어 : self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)<br>출력 생성 선형 레이어 : self.generator = nn.Linear(emb_size, tgt_vocab_size)
### [학습과정]
|학습 정보|설명|
|:---:|:---|
|데이터 로더|torch.utils.data.DataLoader를 사용하여 배치 처리 (배치 크기: BATCH_SIZE)|
|epoch|18|
|손실 함수|CrossEntropyLoss<br>(loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX))|
|옵티마이저|Adam|
---
## 실험 4
|실험요소|설명|
|:---:|:---|
|참고자료|nn.Transformer 와 torchtext로 언어 번역하기 : https://tutorials.pytorch.kr/beginner/translation_transformer.html|
|데이터|한국어-영어 번역(병렬) 말뭉치 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=126|
|데이터 구조|train: 100,000 parallel corpus, valid: 10,000 (10% of train)|
|모델|Transformer|
### [모델 : Transformer]
|구조|설명|
|:---:|:---|
|프레임워크|PyTorch|
|모델 구조|Transformer|
|Encoder 구성|입력 토큰 임베딩 : self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)<br>위치 인코딩 : self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)<br>Transformer 인코더 레이어 : self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)|
|Decoder 구성|출력 토큰 임베딩	: self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)<br>위치 인코딩 : self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)<br>Transformer 디코더 레이어 : self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)<br>출력 생성 선형 레이어 : self.generator = nn.Linear(emb_size, tgt_vocab_size)
### [학습과정]
|학습 정보|설명|
|:---:|:---|
|데이터분할|train_test_split을 사용하여 train_data를 학습 데이터와 검증 데이터로 분할 (검증 데이터 비율: 10%, test_size=0.1)|
|데이터 로더|torch.utils.data.DataLoader를 사용하여 배치 처리 (배치 크기: BATCH_SIZE)|
|epoch|18|
|손실 함수|CrossEntropyLoss<br>(loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX))|
|옵티마이저|Adam|
---
## 성능평가 (BLEU 수치 사용)
### BLEU 점수(Bilingual Evaluation Understudy)는 0에서 1 사이의 값을 가진다.
- 1에 가까울수록 → 예측 번역이 참조 문장(정답)과 아주 유사하다는 의미
- 0에 가까울수록 → 예측 번역이 참조 문장과 거의 유사하지 않다는 의미
- 가독성을 위해 100을 곱하여 100점을 만점으로 하여 표기하기도 한다.

## Seq2Seq 모델 성능 평가 결과
#### attention메커니즘이 들어가지 않은 seq2seq모델
    Train BLEU: 0.1762 (17.62) | Valid BLEU: 0.0779
#### attention메커니즘이 들어간 seq2seq모델
    Train BLEU: 0.1794 (17.94) | Valid BLEU: 0.0801
#### transformer 모델 de-en 번역
    BLEU score: 39.28
#### transformer 모델 en-kr 번역
    BLEU score: 34.57
---
## 분석
처음 결과를 보고 너무 낮은 것이 아닌가 하는 생각이 들었다. 아무리 0에서 1사이의 값이라도 0.1은 너무 작은 것이 아닌가 했다.\
하지만 Chat-gpt4의 BLEU score는 0.88이라는 것을 생각하면 정상적인 수치인 것으로 생각이 되었다.\
여기서 관심을 가져야 할 점은 0.1이라는 점수가 너무 낮은 점수가 아닌가에 대한 것이 아니라 attention메커니즘이 들어갔을 때, 유의미한 점수 변화가 있다는 것이다.\
transformer모델의 경우 34점과 39점으로 앞선 실험에 비해 굉장히 높은 결과를 보여줬다.\
이는 장거리 의존성을 잘 학습하는 Transformer 아키텍처의 우수성을 보여준다.\
번역결과를 보면 de-en의 경우 각종 웹사이트에서 지원하는 번역기와 같은 결과를 냈지만, en-kr의 경우\
    <pre>
    <code>
    print(translate(transformer, "I don't think that you are expert.")) -> 나는 당신이 전문가가 될 것 같아요.
    </code>
    </pre>
위와 같은 정확하지 않은 결과를 냈다.
