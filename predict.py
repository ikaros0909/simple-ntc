import sys
import argparse
import json
import random
from sympy import li

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification

from torch.utils.data import DataLoader

# python .\classify_plm.py --model_fn .\models\review.native.kcbert.pth --test_file .\data\test.tsv --gpu_id 0 --top_n=10
# python .\classify_plm.py --model_fn .\models\y.native.kcbert.pth --test_file .\data\y_test.tsv --gpu_id 0 --top_n=10
def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser() #인자를 받기위한 객체 생성

    p.add_argument('--model_fn', required=True) #저장이 되어있는 모델파일
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256) #추론을 위한 batch size는 좀더 커도됨.
    p.add_argument('--top_k', type=int, default=1) #top 몇개까지 출력할건지
    p.add_argument('--top_n', type=int, default=1) #top 몇개까지 데이터를 입력받을건지

    config = p.parse_args()

    return config


def main(config, p_data):
    saved_data = torch.load( #저장된 모델을 불러옴
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id #gpu_id가 -1이면 cpu로, 아니면 gpu로
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = p_data['sentence']

    return_data = {}
    return_data['prediction'] = []

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name) #tokenizer를 불러옴
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification #albert를 쓸지 bert를 쓸지
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        ) #모델을 불러옴
        model.load_state_dict(bert_best) #모델에 가중치를 불러옴

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id) #gpu_id가 -1이면 cpu로, 아니면 gpu로
        device = next(model.parameters()).device #모델의 device를 가져옴

        # Don't forget turn-on evaluation mode.
        model.eval() #모델을 평가모드로 바꿈
        y_hats = []

        for idx in range(0, len(lines), config.batch_size): #batch_size만큼씩 끊어서 추론
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size], #batch_size만큼의 데이터를 가져옴
                padding=True, #가장 긴것 기준으로 padding
                truncation=True, #maxlength 기준으로 잘라줌
                return_tensors="pt", #pytorch tensor로 변환
            )

            x = mini_batch['input_ids'] #word idx의 tensor가 나옴
            x = x.to(device) #device에 맞게 tensor를 옮김
            mask = mini_batch['attention_mask'] #padding이 된곳에는 attention이 되면 안됨
            mask = mask.to(device) #device에 맞게 tensor를 옮김

            # Take feed-forward
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1) #.logits 는 softmax직전의 hidden state weght
            y_hats += [y_hat]

        y_hats = torch.cat(y_hats, dim=0) #batch_size만큼의 tensor를 합쳐줌
        probs, indice = y_hats.cpu().topk(config.top_k) #top_k개만큼의 확률과 label을 가져옴

        return_data['PredictCnt'] = len(lines) #몇개의 데이터를 입력받았는지
        for i in range(len(lines)):
            prediction_data = {}
            prediction_data['id'] = i,
            prediction_data['prob'] = str(format(probs[i].item(), '2f')) #확률
            prediction_data['label'] = [index_to_label[int(indice[i][j])] for j in range(config.top_k)], #label
            prediction_data['sentence'] = lines[i] #입력받은 문장
            return_data['prediction'].append(prediction_data) #리턴할 데이터에 추가
            # sys.stdout.write('%s\n' % (prediction_data))

    sys.stdout.write(str(return_data)) #리턴할 데이터를 출력
    # return return_data

if __name__ == '__main__':
    config = define_argparser() #config를 가져옴
    main(config, json.loads(input())) #input으로 들어온 데이터를 json으로 변환
