# -*- coding:utf-8 -*-
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from tqdm import tqdm, trange
import argparse
import numpy as np
import matplotlib.pyplot as plt

from Data_Processors import *


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()  # 预测值flatten摊平 [32]
    labels_flat = labels.flatten()  # 真实值摊平
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def main():
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='D:/1_EDU/GLUE/数据集/外卖评价数据集',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='D:/1_EDU/毕业设计/Bert_Model/bert-base-chinese/pytorch_model.bin',
                        type=str,
                        # required = True,
                        help="Choose bert mode which you need(.bin).")
    parser.add_argument("--bert_model_config",
                        default="D:/1_EDU/毕业设计/Bert_Model/bert-base-chinese/bert_config.json",
                        type=str,
                        # required = True,
                        help="Bert model config(.json).")
    parser.add_argument("--bert_model_voc",
                        type=str,
                        # required = True,
                        default="D:/1_EDU/毕业设计/Bert_Model/bert-base-chinese/bert-base-chinese-vocab.txt",
                        help="Bert_model_voc(.txt).")
    parser.add_argument("--output_dir",
                        default='D:/QuickTest',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--max_seq_length",
                        default=32,
                        type=int,
                        # required=True,
                        help="字符串最大长度")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        # required=True,
                        help="训练时batch大小")
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        # required=True,
                        help="训练的epochs次数")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        # required=True,
                        help="AdamW初始学习步长")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")

    args = parser.parse_args()
    """指定GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available():
        print("*****GPU加载成功*****")
    """加载数据"""
    processors = MyPro()
    # test_examples = MyPro.get_test_examples(data_dir=data_dir
    labels = processors.get_labels()
    """初始化分词器"""
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_voc, do_lower_case=args.do_lower_case)
    """加载Bert分类模型"""
    model = BertForSequenceClassification.from_pretrained(args.bert_model, config=args.bert_model_config)
    model.cuda()  # 将模型移入GPU
    """训练和保存模型"""
    if args.do_train:
        train_loss_set = []  # train_loss_set for making loss graph
        train_loss_by_epochs = []
        val_accuracy_set = []
        train_examples = processors.get_train_examples(data_dir=args.data_dir)
        train_features = convert_examples_to_features(train_examples, label_list=labels,
                                                      max_seq_length=args.max_seq_length,
                                                      tokenizer=tokenizer)
        num_train_steps = int(len(train_examples) / args.batch_size * args.epochs)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        """加载数据"""
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)  # 训练时需要shuffle数据 RandomSampler从迭代器里面随机取样本
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        """定义优化器，注意BertAdam、AdamW是不同版本的adam优化方法。  """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        """开始训练"""
        num_epoch = 1
        for _ in trange(args.epochs, desc="Epoch"):
            # 通过将模型设置为训练模式，模型需要计算梯度
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # Tracking variables
            # 准备数据输入和标签
            ac_loss = 0
            for step, batch in enumerate(train_dataloader):
                # 将数据加载到GPU上进行加速
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                input_ids, input_mask, segment_ids, label_ids = batch
                #  清除上一遍计算出的梯度
                optimizer.zero_grad()
                #  正向传递（通过网络输入数据）
                loss = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                             labels=label_ids)
                loss = loss[0]
                train_loss_set.append(loss)
                # 向后传递（反向传播）
                loss.backward()
                # 告诉网络使用optimizer.step（）更新参数
                optimizer.step()
                ac_loss += loss
                tr_loss += loss  # 累加loss
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if nb_tr_steps % 10 == 0:
                    train_loss_set.append(ac_loss / 10)
                    ac_loss = 0
                if nb_tr_steps % 10 == 0:
                    print("**********The Epoch is =", num_epoch, "***** Train_Step =", nb_tr_steps, "**********")
            print("Epoch: {} , Train loss: {}".format(num_epoch, tr_loss / nb_tr_steps))
            train_loss_by_epochs.append(tr_loss / nb_tr_steps)
            num_epoch += 1
            """验证模型"""
            if args.do_eval:
                val_examples = processors.get_dev_examples(data_dir=args.data_dir)
                val_features = convert_examples_to_features(val_examples, label_list=labels,
                                                            max_seq_length=args.max_seq_length,
                                                            tokenizer=tokenizer)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(val_examples))
                logger.info("  Batch size = %d", args.batch_size)
                """加载数据"""
                all_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in val_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in val_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
                val_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                val_sampler = RandomSampler(val_data)  # 训练时需要shuffle数据 RandomSampler从迭代器里面随机取样本
                val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)
                model.eval()
                # Tracking variables
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                # Evaluate data for one epoch
                for batch in val_dataloader:
                    # Add batch to GPU
                    batch = tuple(t.to(device) for t in batch)
                    # Unpack the inputs from our dataloader
                    input_ids, input_mask, segment_ids, label_ids = batch
                    # Telling the model not to compute or store gradients, saving memory and speeding up validation
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions
                        output = model(input_ids=input_ids, attention_mask=input_mask,
                                       token_type_ids=segment_ids)  # 不传labels
                        logits = output[0]  # [32,2]
                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()  # shape:[32,2]
                    label_ids = label_ids.to('cpu').numpy()
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                    eval_accuracy += tmp_eval_accuracy
                    nb_eval_steps += 1
                print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
                val_accuracy_set.append(eval_accuracy / nb_eval_steps)

        """保存模型"""
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(train_loss_set)
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Step/10")
        ax1.set_title("Train loss by every 10 Steps")

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


if __name__ == '__main__':
    main()
