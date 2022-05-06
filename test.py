import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from models.multimodal import TextEncoder, SpeechEncoder
from merdataset import *
from config import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')
    parser.add_argument(
        '--batch',
        default=test_config['batch_size'],
        type=int,
        required=False,
        help='batch size'
    )

    parser.add_argument(
        '--cuda',
        default=test_config['cuda'],
        help='cuda'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='checkpoint name to load'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='test all model ckpt in dir'
    )
    parser.add_argument(
        '--do_clf',
        action='store_true',
    )
    args = parser.parse_args()
    return args


args = parse_args()
if args.cuda != 'cuda:0':
    text_config['cuda'] = args.cuda
    test_config['cuda'] = args.cuda


def test(model, test_dataset):
    print("Test start")
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        dataloader = DataLoader(test_dataset, args.batch,
                                collate_fn=lambda x: (x, torch.LongTensor([i['label'] for i in x])))
        losses = 0
        pred = []
        labels = []

        tq_test = tqdm(total=len(dataloader), desc="testing", position=2)

        for batch in dataloader:
            batch_x, batch_y = batch[0], batch[1]
            batch_y = batch_y.to(args.cuda)

            if isinstance(model,SpeechEncoder) or isinstance(model,TextEncoder):
                outputs = model(batch_x,do_clf=args.do_clf)

            else:
                outputs = model(batch_x)

            loss = loss_func(outputs.to(args.cuda), batch_y)
            outputs = outputs.max(dim=1)[1].tolist()

            loss = loss.item()
            losses += loss

            label = batch_y.tolist()
            labels.extend(label)
            pred.extend(outputs)

            tq_test.update()

        losses = losses / len(test_dataset)
        acc = accuracy_score(labels, pred) * 100
        recall = recall_score(labels, pred, average='weighted') * 100
        precision = precision_score(labels, pred, average='weighted') * 100
        f1 = f1_score(labels, pred, average='weighted') * 100
        confusion = confusion_matrix(labels, pred)
        accs = []
        for idx, vec in enumerate(confusion):
            accs.append(vec[idx] / sum(vec))

        std = np.std(accs)

        print('Test Result: Loss - {:.5f} | Acc - {:.3f} |\
    Recall - {:.3f} | Precision = {:.3f} | F1 - {:.3f} | STD = {:.3f}'.format(losses, acc, recall, precision, f1,std))
    return losses, acc, recall, precision, f1, confusion, std


def main():
    text_conf = pd.Series(text_config)

    if args.model_name:
        test_data = MERDataset(data_option='test', path='./data/')
        test_data.prepare_text_data(text_conf)

        model = torch.load('./ckpt/{}.pt'.format(args.model_name))
        loss, acc, recall, precision, f1, confusion, std = test(model, test_data)
        result = {'loss':loss, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1}
        print(len(confusion))
        print(confusion)
        print("Saving your test result")
        test_result_path = os.path.join('result', args.model_name, "test_result.json")

        if os.path.isdir(os.path.join('result', args.model_name)) == False:
            os.system('mkdir -p ' + os.path.join('result', args.model_name))
        save_dict_to_json(result, test_result_path)
        print("Finish testing")

    elif args.all:
        model_names = os.listdir('./ckpt/test_all/')
        print(model_names)
        test_data = MERDataset(data_option='test', path='./data/')
        test_data.prepare_text_data(text_conf)
        df = []
        for name in model_names:

            model = torch.load('./ckpt/{}'.format(name))
            loss, acc, recall, precision, f1, confusion,std = test(model, test_data)

            result = {'model_name':name,
                      'loss': loss,
                      'acc': acc,
                      'recall': recall,
                      'precision': precision,
                      'f1': f1,
                      'std':std}
            df.append(result)

        print("Saving your test result")
        df = pd.DataFrame(df)
        print(df)
        df.to_csv('./result_all_model2.csv')

    else:
        print("You need to define specific model name to test")


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
