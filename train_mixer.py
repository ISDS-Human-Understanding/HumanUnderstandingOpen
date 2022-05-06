import argparse
import random
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader

from models.multimodal_mixer import SpeechExtractorForMixer, TextEncoderForMixer, MultiModalMixer
from merdataset import *
from config import *
from utils import *

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

    parser.add_argument(
        '--num_blocks',
        type=int,
        default=0,
        help='# of mixer block'
    )
    args = parser.parse_args()
    return args

args = parse_args()
if args.cuda != 'cuda:0':
    audio_config['cuda'] = args.cuda
    text_config['cuda'] = args.cuda
    mixer_config['cuda'] = args.cuda
    train_config['cuda'] = args.cuda

if args.num_blocks:
    mixer_config['num_blocks'] = args.num_blocks

def train(model,optimizer, dataloader):
    model.train()
    model.freeze()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1).to(train_config['cuda'])

    tqdm_train = tqdm(total=len(dataloader), position=1)
    accumulation_steps = train_config['accumulation_steps']

    for batch_id, batch in enumerate(dataloader):
        batch_x, batch_y = batch[0], batch[1]

        outputs = model(batch_x)
        loss = loss_func(outputs.to(train_config['cuda']), batch_y.to(train_config['cuda']))

        tqdm_train.set_description('loss is {:.2f}'.format(loss.item()))
        tqdm_train.update()
        loss = loss / accumulation_steps
        loss.backward()
        if batch_id % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer.zero_grad()
    tqdm_train.close()

def main():
    audio_conf = pd.Series(audio_config)
    text_conf = pd.Series(text_config)
    mixer_conf = pd.Series(mixer_config)

    print(audio_conf)
    print(text_conf)
    print(mixer_conf)
    print(train_config)

    audio_conf['path'] = './KEMDy20/wav/'

    if args.is_training == True:
        dataset = MERDataset(data_option='train', path='./data/')
        dataset.prepare_text_data(text_conf)

        seed = 1024
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        audio = SpeechExtractorForMixer(config=audio_conf)
        text = TextEncoderForMixer(config=text_conf)
        model = MultiModalMixer(mixer_conf, audio_conf, text_conf, audio, text)


        device = args.cuda
        print('---------------------',device)
        model = model.to(device)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

        if 'ckpt' not in os.listdir():
            os.mkdir('ckpt')

        print(model)
        get_params(model)

        if args.save:
            print("checkpoint will be saved every 5epochs!")

        for epoch in range(args.epochs):

            dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=args.shuffle,
                                        collate_fn=lambda x: (x, torch.LongTensor([i['label'] for i in x])))
            train(model, optimizer, dataloader)

            # save model every 5epochs
            if (epoch+1) % 5 == 0:
                if args.save:
                    torch.save(model,'./ckpt/{}_epoch{}.pt'.format(args.model_name,epoch))



if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
