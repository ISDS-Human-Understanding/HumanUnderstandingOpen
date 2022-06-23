import os
import torch
from torch import nn

import soundfile as sf
from tqdm import tqdm
from models.multimodal import encoding

from transformers import Wav2Vec2Processor, Wav2Vec2Model, ElectraTokenizer, ElectraModel
from models.module_for_crossattention.Transformer import *


class SpeechExtractorForCrossAttention():
    def __init__(self, config):
        self.args = config
        self.file_path = self.args.path
        self.max_len = self.args.max_length
        self.processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
        self.encoder = Wav2Vec2Model.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

        if 'hidden_states' not in os.listdir(self.args.path):
            print("Wav Embedding Save")
            os.mkdir(self.args.path + 'hidden_states')
            embed_path = self.args.path + 'hidden_states/'
            self.encoder.to(self.args.cuda)
            len_ = len(os.listdir(self.args.path))
            # if 'hidden_state.json' not in embed_files or 'extract_feature.json' not in embed_files:
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


class TextEncoderForCrossAttention(nn.Module):
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


class MultiModalForCrossAttention(nn.Module):
    def __init__(self, audio_config, text_config, multi_modal_config):
        super().__init__()
        self.audio_args = audio_config
        self.text_args = text_config
        self.args = multi_modal_config

        self.audio_encoder = SpeechExtractorForCrossAttention(self.audio_args)
        self.text_encoder = TextEncoderForCrossAttention(self.text_args)

        self.num_heads = self.args.num_heads
        self.layers = self.args.layers
        self.attn_dropout = self.args.attn_dropout
        self.relu_dropout = self.args.relu_dropout
        self.res_dropout = self.args.res_dropout
        self.embed_dropout = self.args.embed_dropout

        input_dim = self.args.projection_dim * 2

        self.audio2text_transformer = self.get_network(self_type='audio2text').to(self.args.cuda)
        self.text2audio_transformer = self.get_network(self_type='text2audio').to(self.args.cuda)

        self.classifier = nn.Sequential(
            nn.Dropout(self.args.dropout),
            nn.Linear(input_dim, self.args.output_dim),
            nn.GELU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.output_dim, self.args.num_labels)
        ).to(self.args.cuda)

        self.projection = nn.Conv1d(1024, self.args.projection_dim, kernel_size=1, padding=0, bias=False).to(self.args.cuda)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()


    def _conv1d(self, input_features):
        hidden_states = input_features.transpose(1,2).contiguous()  # -> B x (D x L)
        hidden_states = self.projection(hidden_states)
        out = hidden_states.transpose(1, 2).contiguous()            # -> B x (L x D)
        return out

    def get_network(self, self_type='l'):
        if self_type in ['audio', 'text']:
            embed_dim = 2*self.args.projection_dim
        elif self_type in ['audio2text']:
            embed_dim = self.args.projection_dim
        elif self_type in ['text2audio']:
            embed_dim = self.args.projection_dim
            #embed_dim = 1024
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=None)

    def freeze(self):
        if self.text_args.freeze == True:
            self.text_encoder.model.eval()

    def forward(self, batch):
        """
        text, audio should have dimension [batch_size, seq_len, n_features]
        """
        text_out = self.text_encoder(batch)
        audio_out = self.audio_encoder(batch)
        audio_out = self._conv1d(audio_out)

        x_text = text_out.transpose(1, 2)
        x_audio = audio_out.transpose(1, 2)

        proj_text = x_text.permute(2, 0, 1)
        proj_audio = x_audio.permute(2, 0, 1)

        hidden_audio2text = self.audio2text_transformer(proj_audio, proj_text, proj_text)
        hidden_text2audio = self.text2audio_transformer(proj_text, proj_audio, proj_audio)
        hidden_audio2text = hidden_audio2text.permute(1,2,0)
        hidden_audio2text = self.avgpool(hidden_audio2text)
        hidden_audio2text = hidden_audio2text.reshape(hidden_audio2text.shape[0], -1)
        hidden_text2audio = hidden_text2audio.permute(1,0,2)[:,0,:]   # take <s> token (equiv. to [CLS])   # batch, 768
        out = torch.cat([hidden_audio2text, hidden_text2audio], dim=1)
        out = self.classifier(out)

        return out


if __name__ == '__main__':
    import pandas as pd

    audio_config = {
        'K': 3,
        'output_dim': 256,
        'use': 'hidden_state',
        'num_label': 7,
        'path': '../KEMDy20/wav/',
        'cuda': 'cuda:1',
        'max_length': 512
    }

    text_config = {
        'K': 3,
        'output_dim': 256,
        'num_label': 7,
        'max_length': 128,
        'cuda': 'cuda:1'
    }

    multimodal_config = {
        'projection_dim' : 768,
        'output_dim': 768,
        'num_labels':7,
        'dropout': 0.1,
        'cuda':'cuda:1',
        'num_heads': 8,
        'layers': 6,
        'attn_dropout': 0,
        'relu_dropout': 0,
        'res_dropout': 0,
        'embed_dropout' : 0
    }

    audio_conf = pd.Series(audio_config)
    text_conf = pd.Series(text_config)
    multimodal_conf = pd.Series(multimodal_config)

    from merdataset import MERDataset

    dataset = MERDataset(data_option='train', path='../data/')
    dataset.prepare_text_data(text_conf)

    model = MultiModalForCrossAttention(audio_conf,text_conf,multimodal_conf)
    out = model(dataset[:10])
    print(out.size())
