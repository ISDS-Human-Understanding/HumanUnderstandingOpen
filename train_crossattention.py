import argparse
import random
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader

from models.multimodal_cross_attention import *
from merdataset import *
from config import *

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')
    parser.add_argument(
        '--is_training',
        default=True,
        required=False,
        help='run train'
    )
    parser.add_argument(
        '--epochs',
        default=train_config['epochs'],
        type=int,
        required=False,
        help='epochs'
    )
    parser.add_argument(
        '--batch',
        default=train_config['batch_size'],
        type=int,
        required=False,
        help='batch size'
    )
    parser.add_argument(
        '--shuffle',
        default=False,
        required=False,
        help='shuffle'
    )
    parser.add_argument(
        '--lr',
        default=train_config['lr'],
        type=float,
        required=False,
        help='learning rate'
    )
    parser.add_argument(
        '--acc_step',
        default=train_config['accumulation_steps'],
        type=int,
        required=False,
        help='accumulation steps'
    )

    parser.add_argument(
        '--cuda',
        default='cuda:0',
        help='class weight'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='save checkpoint'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='test',
        help='checkpoint name to load or save'
    )

    args = parser.parse_args()
    return args

args = parse_args()
if args.cuda != 'cuda:0':
    audio_config['cuda'] = args.cuda
    text_config['cuda'] = args.cuda
    cross_attention_config['cuda'] = args.cuda
    train_config['cuda'] = args.cuda


def train(model,optimizer, dataloader):
    print("Train start")
    model.train()
    model.freeze()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1).to(train_config['cuda'])

    tqdm_train = tqdm(total=len(dataloader), position=1)
    accumulation_steps = train_config['accumulation_steps']
    loss_list = []
    for batch_id, batch in enumerate(dataloader):
        batch_x, batch_y = batch[0], batch[1]

        outputs = model(batch_x)
        loss = loss_func(outputs.to(train_config['cuda']), batch_y.to(train_config['cuda']))
        loss_list.append(loss.item())

        tqdm_train.set_description('loss is {:.2f}'.format(loss.item()))
        tqdm_train.update()
        loss = loss / accumulation_steps
        loss.backward()
        if batch_id % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer.zero_grad()
    tqdm_train.close()
    print("Train Loss: {:.5f}".format(sum(loss)/len(loss)))

def main():
    audio_conf = pd.Series(audio_config)
    text_conf = pd.Series(text_config)
    cross_attention_conf = pd.Series(cross_attention_config)

    print(audio_conf)
    print(text_conf)
    print(cross_attention_conf)
    print(train_config)

    audio_conf['path'] = './KEMDy20/wav/'

    if args.is_training == True:
        dataset = MERDataset(data_option='train', path='./data/')
        valid_data = MERDataset(data_option='valid', path='./data/')
        dataset.prepare_text_data(text_conf)
        valid_data.prepare_text_data(text_conf)

        seed = 1024
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = MultiModalForCrossAttention(audio_conf,text_conf,cross_attention_conf)

        device = args.cuda
        print('---------------------',device)
        model = model.to(device)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

        print(model)
        for epoch in range(args.epochs):

            dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=args.shuffle,
                                        collate_fn=lambda x: (x, torch.LongTensor([i['label'] for i in x])))
            train(model, optimizer, dataloader)


        if 'ckpt' not in os.listdir():
            os.mkdir('ckpt')
        if args.save:
            torch.save(model,'./ckpt/{}_epoch{}.pt'.format(args.model_name,epoch))



if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
