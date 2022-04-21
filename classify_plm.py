import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True) #저장이 되어있는 모델파일
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256) #추론을 위한 batch size는 좀더 커도됨.
    p.add_argument('--top_k', type=int, default=1) #top 몇개까지 출력할건지

    config = p.parse_args()

    return config


def read_text():
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip()]

    return lines


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = read_text()

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],
                padding=True, #가장 긴것 기준으로 padding
                truncation=True, #maxlength 기준으로 잘라줌
                return_tensors="pt",
            )

            x = mini_batch['input_ids'] #word idx의 tensor가 나옴
            x = x.to(device)
            mask = mini_batch['attention_mask'] #padding이 된곳에는 attention이 되면 안됨
            mask = mask.to(device)

            # Take feed-forward
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1) #.logits 는 softmax직전의 hidden state weght

            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)

        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                ' '.join([index_to_label[int(indice[i][j])] for j in range(config.top_k)]), 
                lines[i]
            ))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
