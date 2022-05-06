import os
from tqdm import tqdm

import torch
import soundfile as sf
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, ElectraTokenizer, ElectraModel
from transformers import ElectraTokenizer, ElectraModel

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


class SpeechExtractorForMixer():
    def __init__(self, config):
        self.args = config
        self.file_path = self.args.path
        self.max_len = self.args.max_length


        if 'hidden_states' not in os.listdir(self.args.path):
            #학습속도를 개선하기 위해 raw wav로 부터 추출한 hidden feature를 저장합니다.
            self.processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
            self.encoder = Wav2Vec2Model.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
            self.encoder.to(self.args.cuda)

            print("Wav Embedding Save")
            os.mkdir(self.args.path + 'hidden_states')
            embed_path = self.args.path + 'hidden_states/'
            self.encoder.to(self.args.cuda)
            len_ = len(os.listdir(self.args.path))

            self.encoder.eval()
            with torch.no_grad():
                for idx, i in enumerate(os.listdir(self.args.path)):
                    print('{}/{}'.format(idx + 1, len_))
                    if '.' not in i:
                        for j in tqdm(os.listdir(self.args.path + i)):
                            if '.wav' in j:
                                wav = self.readfile(j)
                                encoded = self._encoding(wav, output_hidden_state=False)
                                pooled_hidden = encoded.last_hidden_state
                                torch.save(pooled_hidden, embed_path + j[:-4] + '.pt')
                                torch.cuda.empty_cache()

            print("Wav Embedding Save finished")

        del self.encoder

    def readfile(self,file_name):
        if file_name[0] in ['M', 'F']:
            path = self.args.path + 'emotiondialogue/' + file_name
        else:
            session = 'Session' + file_name[4:6] + '/'
            path = self.args.path + session + file_name
        wav, _ = sf.read(path)
        return wav

    def _encoding(self,raw_wav,output_hidden_state=False):
        extract_feature = encoding(raw_wavs=raw_wav,
                                   cuda=self.args.cuda,
                                   encoder=self.encoder,
                                   processor=self.processor,
                                   return_hidden_state=output_hidden_state)

        return extract_feature
    def __call__(self,batch):

        hidden_batch = torch.Tensor().to(self.args.cuda)
        file_name = [data['file_name']+'.pt' for data in batch]
        for data in file_name:

            hidden = torch.load(self.file_path+'hidden_states/'+data,map_location=self.args.cuda)
            #print(hidden.size())
            seq = hidden.size()[1]
            if seq > self.max_len:
                # truncation
                hidden = hidden[:,:self.max_len,:].to(self.args.cuda)
            elif seq < self.max_len:
                # padding
                pad = torch.Tensor([[[0]*1024]*(self.max_len-seq)]).to(self.args.cuda)
                hidden = torch.cat([hidden,pad], dim=1)

            hidden_batch = torch.cat([hidden_batch,hidden],dim=0)
        #print(hidden_batch.size())
        return hidden_batch

class TextEncoderForMixer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.args = config
        self.tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.model = ElectraModel.from_pretrained("beomi/KcELECTRA-base")
        self.model.to(self.args.cuda)

        for params in self.model.parameters():
            params.requires_grad = False


    def forward(self,batch):
        data = [d['dialogue'] for d in batch]
        inputs = self.tokenizer(data,max_length=self.args.max_length,padding='max_length',return_tensors='pt',
                                truncation=True)
        inputs = {k:v.to(self.args.cuda) for k,v in inputs.items()}



        with torch.no_grad():
            self.model.eval()
            hidden_states = self.model(**inputs).last_hidden_state

        return hidden_states

class MlpBlock(nn.Module):
    def __init__(self,input_dim,dropout=0.3):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim,input_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(input_dim,input_dim)

    def forward(self, x):
        y = self.fc(x)
        y = self.gelu(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return y


class MixerBlock(nn.Module):
    def __init__(self, input_dim, sequence_length,dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.modal_mixing = MlpBlock(input_dim,dropout)
        self.sequence_mixing = MlpBlock(sequence_length,dropout)

    def transpose(self,x):
        return x.permute(0,2,1)

    def forward(self, x):
        y = self.ln(x)
        y = self.transpose(y)
        y = self.sequence_mixing(y)
        y = self.transpose(y)
        x = x + y
        y = self.ln(y)
        y = self.modal_mixing(y)
        y = y+x
        return y


class MultiModalMixer(nn.Module):
    def __init__(self,multimodal_config,audio_config,text_config,audio_encoder,text_encoder):
        super().__init__()

        self.args = multimodal_config
        self.audio_args = audio_config
        self.text_args = text_config

        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder

        sequence_length = self.text_args.max_length + self.audio_args.max_length

        self.audio_projection = nn.Linear(1024,self.args.projection_dim)
        self.text_projection = nn.Linear(768,self.args.projection_dim)

        self.m_blocks = nn.ModuleList([
            MixerBlock(self.args.projection_dim, sequence_length, self.args.dropout) for i in range(self.args.num_blocks)
        ])

        self.ln = nn.LayerNorm(self.args.projection_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.projection_dim, self.args.output_dim),
            nn.GELU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.output_dim, self.args.num_labels)
        )

    def freeze(self):
        self.text_encoder.model.eval()

    def forward(self,batch):
        self.audio_encoder(batch)
        audio_hidden_states = self.audio_encoder(batch)
        audio_hidden_states = audio_hidden_states.to(self.args.cuda)
        text_hidden_states = self.text_encoder(batch)

        audio_hidden_states = self.audio_projection(audio_hidden_states)
        text_hidden_states = self.text_projection(text_hidden_states)
        x = torch.cat([text_hidden_states,audio_hidden_states],dim=1)

        for block in self.m_blocks:
            x = block(x)

        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
