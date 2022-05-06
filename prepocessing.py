import os
import sys
import json
import re

import numpy as np
import pandas as pd

file_path20 = './KEMDy20/'
file_path19 = './KEMDy19/'

def sess_correct(x,sess):
    x = sess+x[len(sess):]
    return x

def file2utterance(sess,file,file_path=file_path20):
    with open(file_path+'wav/'+sess+'/'+file+'.txt','r', encoding='cp949') as f:
        return ' '.join(re.sub('[^ㄱ-힣\!\?\.\, ]','',f.readline().strip()).split())
    
def lencheck(sess,file,file_path=file_path20):
    with open(file_path+'wav/'+sess+'/'+file+'.txt','r', encoding='cp949') as f:
        return len(f.readlines())


df20 = {}

annotation = sorted(os.listdir(file_path20+'annotation'))
sess1 = pd.read_csv(file_path20+'annotation/'+annotation[0])
keys = [i[:6] for i in annotation]

# load raw data
for i in range(len(annotation)):
    df20[keys[i]] = pd.read_csv(file_path20+'annotation/'+annotation[i],skiprows=1,index_col=0)

cols = list(df20[keys[0]].columns)
cols[2] = 'file_name'

for i in df20.keys():
    df20[i].columns = cols
    
for i in df20.keys():
    df20[i]['file_name'] = df20[i]['file_name'].apply(lambda x: sess_correct(x,i))

sess_script = {}
for i in keys:
    sess_script[i] = df20[i].file_name.apply(lambda x: x[7:15]).unique().tolist()

for i in keys:
    df20[i]['script'] = df20[i].file_name.apply(lambda x: x[7:15])

text_data={sess:{script:df20[sess][df20[sess]['script']==script][['file_name','Emotion','Valence','Arousal']] for script in sess_script[sess] } for sess in keys}

lst = {sess:{} for sess in keys}
for sess in keys:
    session = "Session"+sess[-2:]
    for script in sess_script[sess]:
        lst[sess][script] = text_data[sess][script]['file_name'].apply(lambda x: lencheck(session,x)).tolist()

# check # of utterace in each file
for sess in keys:
    for script in sess_script[sess]:
        if 1!=any(lst[sess][script]):
            print(sess,script)
        
# preprocessing data file        
for sess in keys:
    for script in sess_script[sess]:
        text_data[sess][script]['speaker'] = text_data[sess][script]['file_name'].apply(lambda x: x[16:24])
        text_data[sess][script].reset_index(drop=True,inplace=True)
        
        session = "Session"+sess[-2:]
        
        text_data[sess][script]['utterance'] = text_data[sess][script]['file_name'].apply(lambda x: file2utterance(session,x))
        history = []
        speaker_history = []
        previous_files = []
        
        text_data[sess][script]['wav'] = text_data[sess][script]['file_name'].apply(lambda x: x+'.wav')
        for i in range(len(text_data[sess][script])):
            history.append(text_data[sess][script]['utterance'][:i].tolist()[::-1])
            speaker_history.append(text_data[sess][script]['speaker'][:i].tolist()[::-1])
            previous_files.append(text_data[sess][script]['wav'][:i].tolist()[::-1])
        
        text_data[sess][script]['history'] = history
        text_data[sess][script]['speaker_hist'] = speaker_history
        text_data[sess][script]['previous_wavs'] = previous_files
        text_data[sess][script] = list(text_data[sess][script].T.to_dict().values())

# make data directory
if 'data' not in os.listdir():
    os.mkdir('data')

# save preprocessed data
with open('./data/processed_KEMDy20.json','w') as j:
    json.dump(text_data,j,ensure_ascii=False, indent=4)

# load test
with open('./data/processed_KEMDy20.json','r') as j:
    data = json.load(j)

print(data['Sess01']['script01'][3])
