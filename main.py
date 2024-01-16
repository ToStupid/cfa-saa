# --*-- coding: utf-8 --*--
import os

from fastNLP import Vocabulary, BucketSampler, DataSetIter
from modules.get_context import get_neighbor_for_vocab, build_instances
from modules.pipe import WNUT_17NERPipe, Conll2003NERPipe

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import argparse
import random
from functools import partial
from collections import Counter, defaultdict


import pickle
import logging
import torch
import tqdm


import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup

from utils.metrics import NERMetric, POSMetric
from utils.tools import log_wrapper
from utils.file_reader import ner_reader, pos_reader
from PipeLine.glue_utils_transformer import SeqDataset, CollateFnSeq
from model.transformer_base import BertForSeqTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

READER = {
    'absa': ner_reader,
    'NER': ner_reader,
    'pos': pos_reader
}

METRIC = {
    'absa': NERMetric,
    'NER': NERMetric,
    'pos': POSMetric
}

def init_args():
    argument = argparse.ArgumentParser()
    # basic configuration
    argument.add_argument('--log_dir', type=str, default='/home/lzy22/mac/cam_self/resources/log')
    argument.add_argument('--cache_dir', type=str, help='pretrained model cache directory', default='/home/lzy22/resources/embeddings/xlnet-base-cased')
    argument.add_argument('--result_dir', type=str, default='/home/lzy22/mac/cam_self/resources/results/')
    argument.add_argument('--dataset_dir', type=str, default='Dataset')
    argument.add_argument('--model_dir', type=str, default='/home/lzy22/mac/cam_self/resources/saved_model')
    argument.add_argument('--best_model', type=str, default='/home/lzy22/mac/cam_self/resources/results/wnut16/pretrained/xlnet-base-cased_LSTM_global_8_1e-05.pth')
    argument.add_argument('--device', type=str, default='cuda')
    argument.add_argument('--fix_pretrained', type=str, default='False')
    argument.add_argument('--task_type', type=str, default='NER')
    argument.add_argument('--train', type=str, default='True')
    argument.add_argument('--mode', type=str, choices=['light', 'pretrained'], default='pretrained')
    # common configuration for models
    argument.add_argument('--num_epoch', type=int, default=300)
    argument.add_argument('--learning_rate', type=float, default=1e-5)
    argument.add_argument('--learning_rate_tagger', type=float, default=1e-3)
    argument.add_argument('--learning_rate_context', type=float, default=1e-3)
    argument.add_argument('--learning_rate_classifier', type=float, default=1e-4)
    argument.add_argument('--momentum', type=float, default=0.9)
    argument.add_argument('--dropout_rate', type=float, default=0.1)
    argument.add_argument('--grad_norm', type=float, default=5.0, help="argument for cutting gradient")
    argument.add_argument('--weight_decay', type=float, default=0.0)
    argument.add_argument('--adam_eps', type=float, default=1e-08)
    argument.add_argument('--warmup_step', type=int, default=5, help='how many steps to execute warmup strategy')
    argument.add_argument('--warmup_strategy', type=str, choices=['None', 'Linear', 'Cosine', 'Constant'],
                          default='None')
    argument.add_argument('--no_improve', type=int, default=5, help='how many steps no improvement to stop training')
    # configuration for pretrained models
    argument.add_argument('--seed', type=int, default=44)
    argument.add_argument('--model_name', type=str, default='xlnet-base-cased') #xlm-roberta-base xlnet-base-cased
    argument.add_argument('--dataset_name', type=str, default='wnut17')  # wnut17 Conll2003 wnut16 BC2GM BC5CDR-disease wnut17_bertscore_eos_doc_full
    argument.add_argument('--bert_size', type=int, default=768)         #wnut17_ret
    argument.add_argument('--batch_size', type=int, default=8)
    argument.add_argument('--use_tagger', type=str, default='True')
    argument.add_argument('--use_crf', type=str, default='False')
    argument.add_argument('--use_context', type=str, default='True')
    argument.add_argument('--use_aug', type=str, default='True')
    argument.add_argument('--context_mechanism', type=str, choices=['global', 'self-attention'], default='global')
    argument.add_argument('--tagger_name', type=str, choices=['LSTM', 'GRU'], default='LSTM')
    argument.add_argument('--tagger_size', type=int, default=768)
    argument.add_argument('--tagger_bidirectional', type=str, default='True')
    args = argument.parse_args()
    return args

def batch_to_device(batch, device):
    for key, value in batch.items():
        if key != 'label_ids_original':
           batch[key] = batch[key].to(device=device)


def pretrained_mode(args):

    train_source = os.path.join(args.dataset_dir, args.task_type, args.dataset_name, 'train.txt')
    valid_source = os.path.join(args.dataset_dir, args.task_type, args.dataset_name, 'dev.txt')
    test_source = os.path.join(args.dataset_dir, args.task_type, args.dataset_name, 'test.txt')

    reader_ = READER.get(args.task_type, ner_reader)
    train_data = SeqDataset(train_source, read_method=reader_)
    valid_data = SeqDataset(valid_source, read_method=reader_)
    test_data = SeqDataset(test_source, read_method=reader_)

    # for i in range(len(valid_data.examples)):
    #     train_data.examples.append(valid_data.examples[i])

    encoding_type = 'bio'
    dataset = 'W17' #Conll2003 W16 BC5CDR-disease
    context_num = 10
    glove_path = '/home/lzy22/resources/embeddings/glove.6B.100d.txt'
    paths = {
        "train": "Dataset/SANER/{}/train.txt".format(dataset),
        "test": "Dataset/SANER/{}/test.txt".format(dataset),
        "dev": "Dataset/SANER/{}/dev.txt".format(dataset)
    }
    data = WNUT_17NERPipe(encoding_type=encoding_type).process_from_file(paths)
    # data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    dict_save_path = os.path.join("Dataset/SANER/{}/data.pth".format(dataset))
    context_dict, context_word2id, context_id2word = get_neighbor_for_vocab(
        data.get_vocab('words').word2idx, glove_path, dict_save_path
    )
    train_feature_data, dev_feature_data, test_feature_data = build_instances(
        "Dataset/SANER/{}".format(dataset), context_num, context_dict
    )
    data.rename_field('words', 'chars')
    vocab_size = len(data.get_vocab('chars'))
    feature_vocab_size = len(context_word2id)

    label_list = train_data.get_label()
    label2idx = defaultdict()
    label2idx.default_factory = label2idx.__len__
    idx2label = {}
    for label in label_list:
        idx = label2idx[label]
        idx2label[idx] = label
    metric = METRIC.get(args.task_type, NERMetric)
    if os.path.exists(args.cache_dir):
        config_ = AutoConfig.from_pretrained(args.cache_dir, hidden_size=args.bert_size)
        # tokenizer_ = AutoTokenizer.from_pretrained(args.cache_dir)
        tokenizer_ = AutoTokenizer.from_pretrained(args.cache_dir, do_lower_case=True)
    else:
        os.mkdir(args.cache_dir)
        config_ = AutoConfig.from_pretrained(args.model_name)
        tokenizer_ = AutoTokenizer.from_pretrained(args.model_name, fintuning_task=args.task_type, id2label=idx2label,
                                                   num_labels=len(idx2label))
        config_.save_pretrained(args.cache_dir)
        tokenizer_.save_pretrained(args.cache_dir)
    tagger_config = dict()
    tagger_config['hidden_size'] = args.tagger_size
    tagger_config['input_size'] = args.bert_size
    tagger_config['tagger_name'] = args.tagger_name
    tagger_config['use_context'] = eval(args.use_context)
    tagger_config['context_mechanism'] = args.context_mechanism
    tagger_config['bidirectional'] = eval(args.tagger_bidirectional)

    collate_fn_seq = CollateFnSeq(tokenizer=tokenizer_, label2idx=label2idx)

    train_loader = DataLoader(train_data, collate_fn=collate_fn_seq, batch_size=args.batch_size)
    eval_loader = DataLoader(valid_data, collate_fn=collate_fn_seq, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, collate_fn=collate_fn_seq, batch_size=4)
    setattr(config_, 'num_labels', len(idx2label))
    setattr(config_, 'use_tagger', eval(args.use_tagger))
    setattr(config_, 'tagger_config', tagger_config)
    setattr(config_, 'use_context', eval(args.use_context))
    setattr(config_, 'use_crf', eval(args.use_crf))
    setattr(config_, 'fix_pretrained', eval(args.fix_pretrained))
    setattr(config_, 'use_aug', eval(args.use_aug))
    device = torch.device(args.device)
    model = BertForSeqTask.from_pretrained(args.cache_dir, config=config_, vocab_size=vocab_size, feature_vocab_size=feature_vocab_size)
    model.to(device=device)
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'tagger' in n], 'lr': args.learning_rate_tagger},
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], },
        {'params': [p for n, p in model.named_parameters() if 'context' in n], 'lr': args.learning_rate_context},
        {'params': [p for n, p in model.named_parameters() if 'memory' in n], 'lr': args.learning_rate_context},
        {'params': [p for n, p in model.named_parameters() if 'fusion' in n], 'lr': args.learning_rate_context},
        {'params': model.classifier.parameters(), 'lr': args.learning_rate_classifier},
    ]
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, eps=args.adam_eps)
    all_step = len(train_loader) * args.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step,
                                                num_training_steps=all_step)

    if eval(args.train):
        best_model_path = train(model,
                                train_loader,
                                eval_loader,
                                optimizer,
                                args,
                                metric,
                                test_loader=test_loader,
                                warmup_strategy=scheduler,
                                num_labels=len(idx2label),
                                loss_fn=None,
                                lr_decay_scheduler=None,
                                idx2label=idx2label,
                                train_feature_data=train_feature_data,
                                dev_feature_data=dev_feature_data)

    else:
        best_model_path = args.best_model
    test(best_model_path, args, test_loader, device, metric(idx2label), dev_feature_data=test_feature_data)

def collate_fn(tokenizer, batch_data=None):
    """
    process batch data before feeding into the model
    """
    sentences = []
    labels = []
    sent_length = []
    word_length = []
    chars = []
    for data in batch_data:
        sentences.append(data['sentence'])
        labels.append(data['labels'])
        sent_length.append(data['sentence_length'])
        word_length.append(data['word_length'])
        chars.append(data['chars'])
    input_batch = tokenizer.encode(sentences, labels, chars)
    input_batch['sentence_length'] = sent_length
    for key, value in input_batch.items():
        # print(key, value)
        if key != 'label_ids_original':
            input_batch[key] = torch.tensor(value)
    return input_batch


def get_features(indices, seq_len, train_feature_data):
    context_num = 10
    ret_list = []
    for index in indices:
        feature_data = train_feature_data[index]
        temp_len = len(feature_data)
        if temp_len >= seq_len:
            ret_list.append(feature_data[:seq_len])
        else:
            ret_list.append(feature_data + [[0] * context_num for _ in range(seq_len - temp_len)])
    return torch.tensor(ret_list)

def train(model,
          train_loader,
          eval_loader,
          optimizer,
          args,
          metric,
          test_loader=None,
          num_labels=None,
          loss_fn=None,
          warmup_strategy=None,
          lr_decay_scheduler=None,
          idx2label=None,
          train_feature_data=None,
          dev_feature_data=None):
    """
    base train for all modes, return saved best model path
    """
    best_f1 = 0
    no_improve_step = 0
    all_step = len(train_loader) * args.num_epoch
    global_step = 0
    best_model_path = None

    for epoch in range(1, args.num_epoch + 1):
        local_step = 0
        epoch_loss = 0.
        epoch_step = len(train_loader)
        p_bar = tqdm.tqdm(train_loader)
        model.train()
        metric_ = metric(idx2label, use_crf=eval(args.use_crf))

        for batch in p_bar:
            batch_to_device(batch, args.device)
            if eval(args.use_crf):
                output, _ = model(**batch)
                loss =output[0]
            else:
                device = torch.device('cuda')
                features = get_features(batch.data['seq_ids'], seq_len=batch.data['input_ids'].shape[1], train_feature_data=train_feature_data).to(device=device)
                batch["features"] = features

                output, gate_weight = model(**batch)
                if args.mode == 'pretrained':
                    loss = output[0]
                else:
                    gold_labels = batch['label_ids']
                    loss = loss_fn(output.view(-1, num_labels), gold_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if warmup_strategy:
                warmup_strategy.step()
            local_step += 1
            global_step += 1
            epoch_loss += loss.item()
            p_bar.set_description(
                f"Epoch: {epoch}: Percentage: {local_step}/{epoch_step} Loss: {round(loss.item(), 2)}")
        eval_f1, fine_grained_f1 = evaluate(eval_loader, metric_, model, args.device,
                                            mode=args.mode, use_crf=eval(args.use_crf), dev_feature_data=dev_feature_data)
        if lr_decay_scheduler:
            lr_decay_scheduler.step()
        logger.info("-" * 50)
        logger.info(
            "Epoch: {}\t Global step: {} \t Avgloss: {:.2f}".format(epoch, global_step, epoch_loss / local_step))

        logger.info("Overall results: ")
        logger.info("precision {:.4f} recall {:.4f} f1 {:.4f}".format(eval_f1['precision'], eval_f1['recall'],
                                                                      eval_f1['f1']))
        logger.info("fine_grained_f1: ")
        for key, fined_f1 in fine_grained_f1.items():
            logger.info("{}: precision {:.4f} recall {:.4f} f1 {:.4f}".format(key, fined_f1['precision'],
                                                                              fined_f1['recall'],
                                                                              fined_f1['f1']))
        if eval_f1['f1'] > best_f1:
            logger.info(
                "This epoch has better f1 {:.4f} than current best f1 {:.4f}".format(eval_f1['f1'], best_f1))
            best_f1 = eval_f1['f1']
            logger.info(f"{global_step}/{all_step}")
            if args.mode == 'pretrained':
                save_name = "{}_{}.pth".format(args.model_name, epoch)
            else:
                save_name = "{}_{}.pth".format('lstm', epoch)
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
            best_model_path = os.path.join(args.model_dir, save_name)
            torch.save(model, best_model_path)
            no_improve_step = 0
        else:
            no_improve_step += 1
        logger.info('-' * 50)
        if no_improve_step > args.no_improve:
            logger.info("Early stop !!!!")
            break
    return best_model_path


def evaluate(data_loader, metrics, model, device, mode='pretrained', use_crf=False, dev_feature_data=None):
    """
    """
    model.eval()

    # metrics = NERMetric(id2token=idx2label)
    for batch in data_loader:

        # device = torch.device('cuda')
        # print(batch.data['seq_ids'], "999", batch.data['input_ids'].shape[1], len(dev_feature_data))
        features = get_features(batch.data['seq_ids'], seq_len=batch.data['input_ids'].shape[1],
                                train_feature_data=dev_feature_data).to(device=device)
        batch["features"] = features

        batch_to_device(batch, device)
        output, gate_weight = model(**batch)
        # torch.save(gate_weight[0].cpu(),'global.pt')
        # torch.save(gate_weight[1].cpu(), 'local.pt')
        # break
        if mode == 'pretrained':
            if use_crf:
                logits = output
                gold_label = batch['labels_original']
            else:
                logits = torch.argmax(output[1], dim=-1)
                gold_label = batch['labels']
        else:
            if use_crf:
                logits = output
                gold_label = batch['label_ids_original']
            else:
                logits = torch.argmax(output, dim=-1)
                gold_label = batch['label_ids']
        metrics(logits, gold_label)
    f1, fine_grained_f1 = metrics.calculate_f1()
    return f1, fine_grained_f1

def test(best_model_path, args, test_loader, device, metric, dev_feature_data=None):
    if best_model_path:
        logger.info("-" * 25 + 'test' + '-' * 25)
        logger.info(f'Best model: {os.path.basename(best_model_path)}')
        model_test = torch.load(best_model_path)
        best_file_dir = os.path.join(args.result_dir, args.dataset_name, args.mode)
        if not os.path.exists(best_file_dir):
            os.makedirs(best_file_dir)
        if args.mode == 'pretrained':
            prefix_ = args.model_name
            if eval(args.use_tagger):
                prefix_ = prefix_ + '_' + args.tagger_name
        else:
            prefix_ = 'LSTM'
        if eval(args.use_context):
            prefix_ = prefix_ + '_' + args.context_mechanism
        prefix_ = prefix_ + '_' + str(args.batch_size) + '_' + str(args.learning_rate)
        best_file_name = prefix_ + '.txt'
        besst_model_name = prefix_ + '.pth'

        f = open(os.path.join(best_file_dir, best_file_name), 'w')
        torch.save(model_test, os.path.join(best_file_dir, besst_model_name))
        f1, fined_f1 = evaluate(test_loader, metric, model_test,
                                device, args.mode, eval(args.use_crf), dev_feature_data=dev_feature_data)
        f.write(best_model_path + '\n')
        f.write("Overall results: \n")
        logger.info("Overall results: ")
        logger.info("precision {:.4f} recall {:.4f} f1 {:.4f}".format(f1['precision'], f1['recall'],
                                                                      f1['f1']))
        f.write("precision {:.4f} recall {:.4f} f1 {:.4f} \n".format(f1['precision'], f1['recall'],
                                                                     f1['f1']))
        logger.info("fine_grained_f1: ")
        f.write("fine_grained_f1: \n")
        for key, fined_f1 in fined_f1.items():
            logger.info("{}: precision {:.4f} recall {:.4f} f1 {:.4f}".format(key,
                                                                              fined_f1['precision'],
                                                                              fined_f1['recall'],
                                                                              fined_f1['f1']))
            f.write("{}: precision {:.4f} recall {:.4f} f1 {:.4f}\n".format(key,
                                                                            fined_f1['precision'],
                                                                            fined_f1['recall'],
                                                                            fined_f1['f1']))
        f.close()


def feed_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    arguments = init_args()
    log_name = arguments.model_name
    if eval(arguments.use_tagger):
        log_name += '-tagger'
        arguments.model_dir += '-tagger'
    if eval(arguments.use_context):
        arguments.model_dir += '-context'
        log_name += '-context'
    feed_seed(arguments.seed)
    log_path = os.path.join('resources/log', log_name)
    log_wrapper(logger, base_dir=log_path)
    pretrained_mode(arguments)

