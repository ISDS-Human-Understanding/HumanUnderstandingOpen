# Human Understanding

[2022 휴먼이해 인공지능 논문경진대회](https://aifactory.space/competition/detail/2006)

## MMM: Multi-modal Emotion Recognition in conversation with MLP Mixer

### Environment

```
python version: python3.8
OS type: Linux
requires packages: {
      'numpy==1.22.3',
      'pandas==1.4.2',
      'torch==1.11.0+cu113',
      'torchaudio==0.11.0+cu113',
      'scikit-learn',
      'transformers==4.18.0',
      'tokenizers==0.12.1',
      'soundfile==0.10.3.post1'
}
```

```
# setup environment
pip install numpy==1.22.3 pandas==1.4.2 scikit-learn transformers==4.18.0 tokenizers==0.12.1 soundfile==0.10.3.post1
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

### Directory
<br/>

**코드 구현을 위해서는 KEMDy20 및 AI Hub 감성대화 말뭉치 음성파일이 알맞은 위치에 있어야합니다.**

<br/>

```
+--KEMDy20
      +--annotation
      +--wav
      # train과 inference 속도를 향상시키기 위해 pretrained Wav2Vec2모델에서 연산한 결과를 미리 저장하여 활용하였음.
            +--audio_embeddings    
                  +--hidden_state.json    
                  +--extract_feature.json
      # train과 inference 속도를 향상시키기 위해 pretrained Wav2Vec2모델에서 연산한 결과를 미리 저장하여 활용하였음.
            +--hidden_states
                +-- {file_name}.pt
      # AI Hub 감성대화 말뭉치 file들이 저장된 폴더
            +--emotion_dialogue
                +--F_000001.wav
                ...
                +--M_005000.wav
            +--Sessoion01
            ...
            +--Session40
      +--TEMP
      +--IBI
      +--EDA
+--data
      +--processed_KEMDy20.json   # KEMDy20데이터와 감성대화 말뭉치를 전처리한 파일
+--models
      +--module_for clossattention
      +--multimodal.py
      +--multimodal_attention
      +--multimodal_cross_attention
      +--multimodal_mixer      
+--merdataset.py
+--preprocessing.py
+--utils.py
+--test.py
+--config.py
+--train.py
+--train_crossattention.py
+--train_mixer.py
```

### Base Model
|Encoder|Architecture|pretrained-weights|
|---|---|---|
|**Audio Encoder**|pretrained Wav2Vec 2.0|kresnik/wav2vec2-large-xlsr-korean|
|**Text Encoder**|pretrained Electra|beomi/KcELECTRA-base|

<br/>

### Arguments
#### train.py
| argument           | description                               |
|--------------------|-------------------------------------------|
| --epochs {int}     | epoch 횟수                                 |
| --batch {int}      | batch 사이즈                                |
| --save             | 체크포인트 저장 여부                          |
| --retrain          | 기존 모델을 불러와 다시 학습                   |
| --model_name {str} | 모델을 불러오거나 저장할 때 사용할 모델명        |
| --ws               | weighted random sampling 수행              |        
| --shuffle {bool}   | data shuffle 수행 여부 --ws와 동시 적용 X     |
| --cuda {cuda:num}  | 사용할 gpu 번호                             |
| --K {int}          | 사용할 utterance context 수                 |
| --hidden           | feature가 아닌 hidden state를 사용합니다   |
| --class_weight {bool}| 각 클래스별 loss에 가중치를 두어 학습합니다.|
|--use_threeway|three way concat 모델 구조로 학습을 진행합니다|

```
# vanilla training
python train.py --model_name vanilla --save --class_weight False

# 3way concat training
python train.py --model_name three_way --save --class_weight False
```

#### train_crossattention.py
| argument           | description                 |
|--------------------|-----------------------------|
| --epochs {int}     | epoch 횟수                   |
| --batch {int}      | batch 사이즈                  |
| --save             | 체크포인트 저장 여부            |
| --retrain          | 기존 모델을 불러와 다시 학습       |
| --model_name {str} | 모델을 불러오거나 저장할 때 사용할 모델명 | 
| --shuffle {bool}   | data shuffle 수행 여부     |
| --cuda {cuda:num}  | 사용할 gpu 번호               |

```
# multimodal cross attention training
python train_crossattention.py --model_name cross_attention --save
```

#### train_mixer.py
| argument           | description                      |
|--------------------|----------------------------------|
| --epochs {int}     | epoch 횟수                         |
| --batch {int}      | batch 사이즈                        |
| --save             | 체크포인트 저장 여부                      |
| --retrain          | 기존 모델을 불러와 다시 학습                 |
| --model_name {str} | 모델을 불러오거나 저장할 때 사용할 모델명          |
| --shuffle {bool}   | data shuffle 수행 여부 --ws와 동시 적용 X |
| --cuda {cuda:num}  | 사용할 gpu 번호                       |
|--num_blocks {int} | MLP layer의 수를 설정합니다.             |

```
# multimodal mixer training
python train_mixer.py --model_name mlp_mixer --save
```

#### test.py
| argument           | description                      |
|--------------------|----------------------------------|
| --batch {int}      | batch 사이즈                        |
|--cuda {cuda:num}| 사용할 gpu 번호|
|--model_name| test할 model file 이름|
|--all| test_all 폴더의 모든 file을 test합니다.|

```
python test.py --model_name {model_name}_epoch29 --batch 64
```

### Model Architecture

## Vanilla Concat &  3-way Concat
<img width="30%" src="./img/vanilla concat.png"> <img width="30%" src="./img/3way concat.png">

## Multimodal Cross Attention & Multimodal MLP-Mixer
<img width="30%" src="img/cross attention.png"> <img width="30%" src="./img/Multimodal Mixer.png">

### Mixer Layer
<img width="100%" src="./img/Mixer Layer.png">


### Experiments

<br/>

- Architecture

|Index|Architecture| Accuracy | W-Precision | W-F1   |
|-----|-----|----------|-------------|--------|
|1|Vanilla concat| 71.753   | 69.587      | 70.250 |
|2|3 way concat| 73.127   | 72.088      | 72.138 |
|3|Cross-Attention| 77.749   | 78.298      | 77.343 |
|4|MLP-Mixer| 78.884   | 78.450      | 78.418 |



### Data Source

[일반인 대상 자유발화](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)
<br/>
[AI Hub 감성대화 말뭉치](https://aihub.or.kr/aidata/7978)

### References

Noh, K.J.; Jeong, C.Y.; Lim, J.; Chung, S.; Kim, G.; Lim, J.M.; Jeong, H. Multi-Path and Group-Loss-Based Network for Speech Emotion Recognition in Multi-Domain Datasets. Sensors 2021, 21, 1579. https://doi.org/10.3390/s21051579

Tsai, Yao-Hung Hubert, et al. Multimodal transformer for unaligned multimodal language sequences. Association for Computational Linguistics (ACL), 2019. https://github.com/yaohungt/Multimodal-Transformer

```
@inproceedings{tsai2019MULT,
  title={Multimodal Transformer for Unaligned Multimodal Language Sequences},
  author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Liang, Paul Pu and Kolter, J. Zico and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month = {7},
  year={2019},
  address = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
}
```

Junbum, Lee. KcELECTRA: Korean comments ELECTRA, 2021.
```
@misc{lee2021kcelectra,
  author = {Junbum Lee},
  title = {KcELECTRA: Korean comments ELECTRA},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Beomi/KcELECTRA}}
}
```
