import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertModel, BertTokenizer
from train_bert import convert_examples_to_features, MyPro

from Models import Bert_BiLSTM_Attention, Bert_Attention


class Config:
    def __init__(self):
        self.bert_model = "D:/QuickTest/pytorch_model.bin"
        self.bert_model_config = "D:/QuickTest/bert_config.json"
        self.bert_model_voc = "D:/QuickTest/bert-base-chinese-vocab.txt"
        self.data_dir = "D:/QuickTest/data"
        self.output_dir = "D:/QuickTest/output"
        # Config of Processor
        self.num_labels = 2
        self.max_seq_length = 32
        self.batch_size = 16
        self.do_lower_case = True
        self.do_train = False
        self.do_eval = True
        self.do_test = True
        # Config of LSTM
        self.hidden_dims = 256
        self.num_layers = 1  # 2
        self.dropout_prob = 0.1
        self.embedding_size = 768
        # Config of Training
        self.epochs = 3
        self.learning_rate = 1e-5


def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()  # 预测值flatten摊平 [32]
    labels_flat = labels.flatten()  # 真实值摊平
    return np.sum(pred_flat == labels_flat), len(labels_flat)


def Test(config, model, processors, tokenizer, labels, device):
    test_examples = processors.get_test_examples(data_dir=config.data_dir)
    test_features = convert_examples_to_features(test_examples, label_list=labels,
                                                 max_seq_length=config.max_seq_length,
                                                 tokenizer=tokenizer)
    print("***** Running Test_Set *****")
    print("  Num examples = {}".format(len(test_examples)))
    print("  Batch size = {}".format(config.batch_size))
    """加载数据"""
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    val_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    val_sampler = RandomSampler(val_data)  # 训练时需要shuffle数据 RandomSampler从迭代器里面随机取样本
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config.batch_size)

    model.eval()
    # Tracking variables
    right, total = 0, 0

    output_html = os.path.join(config.output_dir, "html.html")
    file_handle = open(output_html, mode='w', encoding='utf-8')
    # Evaluate data for one epoch
    for batch in val_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask, label_ids = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            output = model.forward(inputs_id=input_ids, attention_mask=input_mask)  # 不传labels
            best_path = output[0]
            attention_weights = output[1]
        batch_size_tokens = []
        if total <= 500:
            for seq in input_ids:
                tokens = tokenizer.convert_ids_to_tokens(seq.to('cpu').numpy())
                batch_size_tokens.append(tokens)
            text = mk_html(config=config, tokens=batch_size_tokens, attns=attention_weights.to('cpu').numpy())
            file_handle.write(text)
        best_path = best_path.detach().cpu().numpy().reshape(-1)
        label_ids = label_ids.to('cpu').numpy().reshape(-1)
        r, t = flat_accuracy(best_path, label_ids)
        right += r
        total += t
    accuracy = right / total
    file_handle.close()
    print("Validation Accuracy: {}".format(accuracy))
    return accuracy


def Eval(config, model, processors, tokenizer, labels, device, val_accuracy_set):
    val_examples = processors.get_dev_examples(data_dir=config.data_dir)
    val_features = convert_examples_to_features(val_examples, label_list=labels,
                                                max_seq_length=config.max_seq_length,
                                                tokenizer=tokenizer)
    print("***** Running Evaluation *****")
    print("  Num examples = {}".format(len(val_examples)))
    print("  Batch size = {}".format(config.batch_size))
    """加载数据"""
    all_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in val_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
    val_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    val_sampler = RandomSampler(val_data)  # 训练时需要shuffle数据 RandomSampler从迭代器里面随机取样本
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config.batch_size)

    model.eval()
    # Tracking variables
    right, total = 0, 0
    # Evaluate data for one epoch
    for batch in val_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask, label_ids = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            output = model.forward(inputs_id=input_ids, attention_mask=input_mask)  # 不传labels
            best_path = output[0]

        best_path = best_path.detach().cpu().numpy().reshape(-1)
        label_ids = label_ids.to('cpu').numpy().reshape(-1)
        r, t = flat_accuracy(best_path, label_ids)
        right += r
        total += t
    accuracy = right / total
    print("Validation Accuracy: {}".format(accuracy))
    val_accuracy_set.append(accuracy)
    return accuracy, val_accuracy_set


def Train(config, model, processors, tokenizer, labels, device, optimizer):
    train_loss_set = []  # Set for making loss graph
    train_loss_by_epochs = []
    val_accuracy_set = []
    train_examples = processors.get_train_examples(data_dir=config.data_dir)
    train_features = convert_examples_to_features(train_examples, label_list=labels,
                                                  max_seq_length=config.max_seq_length,
                                                  tokenizer=tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)  # 训练时需要shuffle数据 RandomSampler从迭代器里面随机取样本
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)

    total_steps = int(len(train_examples) / config.batch_size * config.epochs)
    print("***** Running Training *****")
    print("  Num examples = {}".format(len(train_examples)))
    print("  Batch size = {}".format(config.batch_size))
    print("  Total steps = {}".format(total_steps))
    num_steps = 0

    output_train_doc = os.path.join(config.output_dir, "train_doc.txt")
    file_handle = open(output_train_doc, mode='w')

    for i in range(config.epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        ac_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # 将数据加载到GPU上进行加速
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            input_ids, input_mask, label_ids = batch
            #  清除上一遍计算出的梯度
            optimizer.zero_grad()
            #  正向传递（通过网络输入数据）
            loss = model.forward(inputs_id=input_ids, attention_mask=input_mask, labels=label_ids)
            loss = loss[0]
            # 向后传递（反向传播）
            loss.backward()
            # 更新参数
            optimizer.step()
            epoch_loss += loss  # 累加loss
            ac_loss += loss
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            num_steps += 1
            if nb_tr_steps % 10 == 0:
                train_loss_set.append(ac_loss / 10)
                ac_loss = 0
            if nb_tr_steps % 10 == 0:
                print("********The Epoch is = {}, Train_Step = {} / {}********".format(i + 1, num_steps, total_steps))
        """验证模型"""
        accuracy, val_accuracy_set = Eval(config, model=model, processors=processors, tokenizer=tokenizer,
                                          labels=labels, device=device, val_accuracy_set=val_accuracy_set)
        epoch_time = time.time() - epoch_start_time
        epoch_loss_avg = epoch_loss / nb_tr_steps
        print("Epoch: {} , Train_Time: {}, Train loss: {}, Val_Accuracy: {}".format(i + 1, epoch_time, epoch_loss_avg,
                                                                                    accuracy))
        file_handle.write("Epoch:" + str(i + 1) + "\tTrain_time:" + str(epoch_time) +
                          "\tTrain_loss: " + str(epoch_loss_avg) + "\tVal_Accuracy: " + str(accuracy) + "\n")

        train_loss_by_epochs.append(epoch_loss_avg)

    file_handle.close()
    output_model_file = os.path.join(config.output_dir, "bert_attention_model.pth")
    torch.save(model.state_dict(), output_model_file)

    # 对训练过程进行绘图
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_loss_set)
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Step")
    ax1.set_title("Train loss by Every 10 Steps")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Loss & Accuracy by epochs")
    ax2.plot(train_loss_by_epochs, "r", label="Loss")
    ax2.legend(loc=1)
    ax2.set_ylabel('Loss')
    ax3 = ax2.twinx()
    ax3.plot(val_accuracy_set, 'g', label="Accuracy")
    ax3.legend(loc=2)
    ax3.set_ylabel('Accuracy')
    ax3.set_xlabel('Epoch')
    plt.show()


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255 * (1 - attn)), int(255 * (1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def mk_html(config, tokens, attns):
    html = ""

    for i in range(config.batch_size):
        attn_max = 0
        attn_min = 1
        for word, attn in zip(tokens[i], attns[i]):
            if word != '[CLS]' and word != '[SEP]':
                if attn > attn_max:
                    attn_max = attn
                if attn < attn_min:
                    attn_min = attn
        c = attn_max - attn_min
        for word, attn in zip(tokens[i], attns[i]):
            if word != '[PAD]' and word != '[CLS]' and word != '[SEP]':
                html += highlight(word, (attn - attn_min) / c)
        html += "<br>\n"
    return html + "<br><br>\n"


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Bert_Attention(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_voc, do_lower_case=config.do_lower_case)

    processors = MyPro()
    labels = processors.get_labels()

    if config.do_train:
        Train(config, model=model, processors=processors, tokenizer=tokenizer, labels=labels,
              device=device, optimizer=optimizer)
    if config.do_test:
        model_file = os.path.join(config.output_dir, "bert_attention_model.pth")
        model.load_state_dict(torch.load(model_file))
        Test(config, model=model, processors=processors, tokenizer=tokenizer, labels=labels, device=device)


if __name__ == '__main__':
    main()
