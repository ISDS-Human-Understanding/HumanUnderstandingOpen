import os
import json
from tqdm import tqdm

import torch
import soundfile as sf
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import ElectraTokenizer, ElectraModel

from multimodal_attention import *


def read_file(file_name):
    wav, _ = sf.read(file_name,samplerate=16000)
    return wav

def encoding(raw_wavs,cuda, processor=None, encoder=None, return_hidden_state=False):
    assert bool(processor) == bool(encoder)

    inputs = processor(raw_wavs,
                       sampling_rate=16000,
                       return_attention_mask=True,
                       return_tensors="pt")
    inputs = inputs.to(cuda)
    encoder = encoder.to(cuda)
    outputs = encoder(output_hidden_states=return_hidden_state, **inputs)

    return outputs

class SpeechEncoder(nn.Module):
    def __init__(self,audio_config):
        super().__init__()
        self.args = audio_config
        self.processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
        self.encoder = Wav2Vec2Model.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
        self.pooled_hidden = {}
        self.pooled_feature = {}

        print("Wav Embedding Save & Load")
        if 'audio_embeddings' not in os.listdir(self.args.path):
            os.mkdir(self.args.path+'audio_embeddings')

        embed_path = self.args.path+'audio_embeddings/'
        embed_files = os.listdir(embed_path)

        # Encoder에서는 projection layer를 제외하고는 학습을 하지 않기 때문에 학습 및 추론 속도 개선을 위해
        # 모델 최초 선언시 파일의 embedding을 먼저 저장합니다.
        if 'hidden_state.json' not in embed_files or 'extract_feature.json' not in embed_files:
            self.encoder.eval()
            with torch.no_grad():
                for idx,i in enumerate(os.listdir(self.args.path)):
                    print('{}/48'.format(idx+1))
                    if '.' not in i:
                        for j in tqdm(os.listdir(self.args.path+i)):
                            if '.wav' in j:
                                wav = self.readfile(j)
                                encoded = self._encoding(wav,output_hidden_state=True)
                                pooled_hidden, pooled_feature = encoded.last_hidden_state.mean(dim=1).tolist(),\
                                                                encoded.extract_features.mean(dim=1).tolist()

                                self.pooled_hidden[j] = pooled_hidden
                                self.pooled_feature[j] = pooled_feature
                                torch.cuda.empty_cache()

            with open(embed_path+'hidden_state.json', 'w') as j:
                json.dump(self.pooled_hidden, j)

            with open(embed_path+'extract_feature.json', 'w') as j:
                json.dump(self.pooled_feature, j)

        # 저장한 Embedding Vector를 load 합니다.
        with open(embed_path+'hidden_state.json', 'r') as j:
            self.pooled_hidden = json.load(j)

        with open(embed_path+'extract_feature.json', 'r') as j:
            self.pooled_feature = json.load(j)

        #Embedding vector를 불러왔으므로 encoder를 삭제해줍니다.
        del self.encoder

        #for params in self.encoder.parameters():
        #    params.requires_grad = False

        if self.args.use == 'hidden_state':
            input_dim = 1024
        elif self.args.use == 'extract_feature':
            input_dim = 512

        self.projection = nn.Linear(input_dim,self.args.output_dim)
        self.clf = nn.Linear(self.args.output_dim*self.args.K,self.args.num_label)
        self.flatten = nn.Flatten()

    def _encoding(self,raw_wav,output_hidden_state=False):
        extract_feature = encoding(raw_wavs=raw_wav,
                                   cuda=self.args.cuda,
                                   encoder=self.encoder,
                                   processor=self.processor,
                                   return_hidden_state=output_hidden_state)

        return extract_feature

    def readfile(self,file_name):
        if file_name[0] in ['M','F']:
            path = self.args.path + 'emotiondialogue/' + file_name
        else:
            session = 'Session'+file_name[4:6]+'/'
            path = self.args.path + session + file_name
        wav, _ = sf.read(path)
        return wav

    def contextualize(self,data):
        if self.args.use == 'hidden_state':
            embedding = self.pooled_hidden
        elif self.args.use == 'extract_feature':
            embedding = self.pooled_feature

        context = data['previous_wavs'][:self.args.K-1]
        utterance = data['wav']

        dialogue_wav = []

        # wav file에 해당하는 embedding vector를 K턴에 맞게 불러옵니다.
        dialogue_wav.append(torch.Tensor(embedding[utterance]).to(self.args.cuda))
        for i in context:
            dialogue_wav.append(torch.Tensor(embedding[i]).to(self.args.cuda))

        if len(dialogue_wav)!=self.args.K:
            padding_dim = self.args.K - len(dialogue_wav)
            for i in range(padding_dim):
                if self.args.use == 'hidden_state':
                    dialogue_wav.append(torch.Tensor([[0]*1024]).to(self.args.cuda))
                elif self.args.use == 'extract_feature':
                    dialogue_wav.append(torch.Tensor([[0]*512]).to(self.args.cuda))

        return dialogue_wav

    def forward(self,batch, do_clf=False):

        #batch의 각 데이터를 contextualize 해줍니다.
        context = list(map(self.contextualize,batch))



        if self.args.use == 'hidden_state':

            hidden_state_batch = torch.stack([torch.cat([hidden[i] for hidden in context],dim=0)\
                                  for i in range(self.args.K)],dim=1)
            output = self.projection(hidden_state_batch)

        elif self.args.use == 'extract_feature':

            extract_feature_batch = torch.stack([torch.cat([feature[i] for feature in context],dim=0)\
                                     for i in range(self.args.K)],dim=1)
            output = self.projection(extract_feature_batch)

        # (B,K,D) > (B, D*K) : B - batch K - # context D - output_dim
        output = self.flatten(output)


        if do_clf:
            output = self.clf(output)

        return output


class TextEncoder(nn.Module):
    def __init__(self,text_config):
        super().__init__()
        self.args = text_config
        self.tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.model = ElectraModel.from_pretrained("beomi/KcELECTRA-base").to(self.args.cuda)

        if self.args.freeze == True:
            for params in self.model.parameters():
                params.requires_grad = False


        input_dim = 768
        self.projection = nn.Linear(input_dim,self.args.output_dim)
        self.clf = nn.Linear(self.args.output_dim, self.args.num_label)

    def forward(self,batch, do_clf=False, return_hidden=False):
        data = [d['dialogue'] for d in batch]
        inputs = self.tokenizer(data,max_length=self.args.max_length,padding='max_length',return_tensors='pt',
                                truncation=True)
        inputs = {k:v.to(self.args.cuda) for k,v in inputs.items()}
        hidden_states = self.model(**inputs)

        if return_hidden:
            return hidden_states

        x = hidden_states[0][:,0,:]
        x = self.projection(x)

        if do_clf:
            x = self.clf(x)

        return x


class MultiModalForClassification(nn.Module):
    def __init__(self,audio_config,text_config,multi_modal_config):
        super().__init__()
        self.audio_args = audio_config
        self.text_args = text_config
        self.args = multi_modal_config

        self.audio_encoder = SpeechEncoder(self.audio_args)
        self.text_encoder = TextEncoder(self.text_args)

        if self.args.use_threeway:
            input_dim = self.audio_args.K * self.audio_args.output_dim * 3 + self.text_args.output_dim
            self.classifier = nn.Sequential(
                nn.Dropout(self.args.dropout),
                nn.Linear(input_dim, int(self.args.output_dim*1.5)),
                nn.GELU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(int(self.args.output_dim*1.5), self.args.num_labels)
            )

        elif self.args.use_attention:
            input_dim = self.text_args.output_dim * 2
            self.classifier = nn.Sequential(
                nn.Dropout(self.args.dropout),
                nn.Linear(input_dim, self.args.output_dim),
                nn.GELU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(self.args.output_dim, self.args.num_labels)
            )
            self.crossattention = MultimodalCrossAttention(self.audio_args, self.text_args)

        else:
            input_dim = self.audio_args.K * self.audio_args.output_dim + self.text_args.output_dim
            self.classifier = nn.Sequential(
                nn.Dropout(self.args.dropout),
                nn.Linear(input_dim, self.args.output_dim),
                nn.GELU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(self.args.output_dim, self.args.num_labels)
            )

        self.flatten = nn.Flatten()


    def freeze(self):
        # self.audio_encoder.encoder.eval() (audio encoder 삭제)
        if self.text_args.freeze == True:
            self.text_encoder.model.eval()

    def forward(self, batch):
        text_out = self.text_encoder(batch)
        if self.args.use_threeway:
            audio_out = self.audio_encoder(batch)
            temp = torch.stack([text_out] * self.audio_args.K, dim=1)
            temp = self.flatten(temp)
            minus_ = self.flatten(torch.abs(audio_out - temp))
            mul_ = self.flatten(audio_out * temp)
            audio_out = self.flatten(audio_out)

            out = torch.cat([text_out, audio_out, minus_, mul_], dim=-1)

        elif self.args.use_attention:
            audio_out = self.audio_encoder(batch)
            tts_attention_out, tts_normalized_weights, stt_attention_out, stt_normalized_weights = self.crossattention(audio_out, text_out)
            out = torch.cat([tts_attention_out, stt_attention_out], dim=1)

        else:
            audio_out = self.audio_encoder(batch)
            out = torch.cat([audio_out, text_out], dim=1)
        out = self.classifier(out)

        return out

