import json
from collections import Counter

from torch.utils.data import Dataset

class MERDataset(Dataset):
    def __init__(self,data_option='train',path='./'):
        with open(path+'processed2_KEMDy20.json','r') as file:
            data = json.load(file)



        train = list(range(1,33))
        test = list(range(33,41))

        self.emo_map = {'neutral': 0,
         'happy': 1,
         'surprise':2,
         'angry': 3,
         'sad':4,
         'disqust': 5,
         'fear': 6}
        script = data['Sess01'].keys()

        self.data = []
        if data_option == 'train':
            r = train

        elif data_option == 'test':
            r = test

        for sess in r:
            session = 'Sess'+'{0:0>2}'.format(sess)
            for s in script:
                self.data.extend(data[session][s])

        del_idx = []
        for idx,data in enumerate(self.data):
            if ';' in data['Emotion']:
                temp = data
                emotions = data['Emotion'].split(';')
                for emotion in emotions:
                    temp['Emotion'] = emotion
                    self.data.append(temp)
                del_idx.append(idx)

        for idx in del_idx[::-1]:
            del self.data[idx]

        for idx in range(len(self.data)):
            label = self.emo_map[self.data[idx]['Emotion']]
            self.data[idx]['label'] = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    def prepare_text_data(self,text_config):
        K = text_config.K
        for idx,data in enumerate(self.data):
            dialogue = [data['utterance']]+data['history'][:K-1]
            dialogue = '[SEP]'.join(dialogue)
            self.data[idx]['dialogue'] = dialogue

    def get_weight(self):
        weight = Counter([data['label'] for data in self.data])
        weight = [weight[i] for i in range(0,7)]
        sum_ = len(self.data)
        weight = [sum_/i for i in weight]
        return weight




if __name__ == '__main__':
    dataset = MERDataset(data_option='train',path='./data/')
    weight = dataset.get_weight()
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.sampler import WeightedRandomSampler

    labels = [data['label'] for data in dataset]
    counter = Counter(labels)
    counter = {k:len(labels)/v for k,v in counter.items()}
    import torch
    weight = [counter[i] for i in labels]
    sampler = WeightedRandomSampler(weight,len(weight))
    dataloader = DataLoader(dataset,batch_size=16,sampler=sampler,collate_fn= lambda x: (x,torch.Tensor([i['label']for i in x])))
    l = []
    for batch_x, batch_y in dataloader:
        l.extend(batch_y.tolist())
    print(Counter(l))