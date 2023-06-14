import argparse
import random

# conda install pytorch torchvision torchaudio cpuonly -c pytorch
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# conda install -c huggingface transformers
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# conda install -c conda-forge torch-optimizer
import torch_optimizer as custom_optim

from simple_ntc.bert_trainer import BertTrainer as Trainer
from simple_ntc.bert_dataset import TextClassificationDataset, TextClassificationCollator
from simple_ntc.utils import read_text

# negative positive
# training
# python .\finetune_plm_native_jinhak.py --model_fn ./models/review.native.kcbert_20230317.pth --train_fn ./data/review.sorted.uniq.refined.shuf.train_edit.tsv --gpu_id 0 --batch_size 80 --n_epochs 5 --pretrained_model_name 'beomi/kcbert-base'
# test
# python .\classify_plm.py --model_fn .\models\review.native.kcbert.pth --test_file ./data/review.sorted.uniq.refined.shuf.test.tsv --gpu_id 0 --top_n=20
# python .\classify_plm.py --model_fn .\models\review.native.kcbert_20230317.pth --test_file .\data\y_test_h.tsv  --save_file .\data\result_20230317h.csv --gpu_id 0 --top_n=17799

# humanism
# training
# python .\finetune_plm_native_jinhak.py --model_fn ./models/y.native.kcbert_20230318.pth --train_fn ./data/y_train_20230317.tsv --gpu_id 0 --batch_size 80 --n_epochs 20 --pretrained_model_name 'beomi/kcbert-base'
# testing
# python .\classify_plm.py --model_fn .\models\y.native.kcbert_20230317.pth --test_file .\data\y_test_20230317.tsv --save_file .\data\result_20230317.csv --gpu_id 0 --top_n=17799

# ai-hub
# training
# python .\finetune_plm_native_jinhak.py --model_fn ./models/ai-hub-c1.kcbert_20230414.pth --train_fn ./data/ai_hub_c1.tsv --gpu_id 0 --batch_size 24 --n_epochs 10 --pretrained_model_name 'beomi/kcbert-base'
# python .\finetune_plm_native_jinhak.py --model_fn ./models/ai-hub-c2.kcbert_20230414.pth --train_fn ./data/ai_hub_c2.tsv --gpu_id 0 --batch_size 24 --n_epochs 10 --pretrained_model_name 'beomi/kcbert-base'
# python .\finetune_plm_native_jinhak.py --model_fn ./models/ai-hub-c3.kcbert_20230414.pth --train_fn ./data/ai_hub_c2.tsv --gpu_id 0 --batch_size 24 --n_epochs 10 --pretrained_model_name 'beomi/kcbert-base'
# testing
# python .\classify_plm.py --model_fn .\models\ai-hub-c1.kcbert_20230414.pth  --test_file .\data\ai-hub-test.tsv --save_file .\data\result_ai-hub-c1_20230414.csv --gpu_id 0 --top_n=355
# python .\classify_plm.py --model_fn .\models\ai-hub-c2.kcbert_20230414.pth  --test_file .\data\ai-hub-test.tsv --save_file .\data\result_ai-hub-c2_20230414.csv --gpu_id 0 --top_n=355
# python .\classify_plm.py --model_fn .\models\ai-hub-c3.kcbert_20230414.pth  --test_file .\data\ai-hub-test.tsv --save_file .\data\result_ai-hub-c3_20230414.csv --gpu_id 0 --top_n=355


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    # Recommended model list:
    # - kykim/bert-kor-base
    # - kykim/albert-kor-base
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--pretrained_model_name', type=str,
                   default='beomi/kcbert-base')
    p.add_argument('--use_albert', action='store_true')

    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)  # 숫자가 높을수록 자세히 보여줌

    # 2080ti기준 : batchsize 11기가 80=>64=>48=>32=>24
    p.add_argument('--batch_size', type=int, default=80)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--lr', type=float, default=5e-5)  # running rate
    # Adam만 쓰면 학습이 잘 안됨.Adam을 쓰면서 warmup하는 방법,
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    # If you want to use RAdam, I recommend to use LR=1e-4.
    # Also, you can set warmup_ratio=0.
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--valid_ratio', type=float, default=.2)

    # max_length 늘리면 batchsize 줄여야함.(기본 100)
    p.add_argument('--max_length', type=int, default=300)

    config = p.parse_args()

    return config


def get_loaders(fn, tokenizer, valid_ratio=.2):
    # Get list of labels and list of texts.
    labels, texts = read_text(fn)

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader, index_to_label


def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer


def main(config):
    # Get pretrained tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders(
        config.train_fn,
        tokenizer,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )
    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


# python finetune_plm_native.py --model_fn ./models/review.native.kcbert.pth --train_fn ./data/review.sorted.uniq.refined.shuf.train.tsv --gpu_id 0 --batch_size 42 --n_epochs 2 --pretrained_model_name 'beomi/kcbert-base'
if __name__ == '__main__':
    config = define_argparser()
    main(config)
