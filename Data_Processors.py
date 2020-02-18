# coding: utf8
import codecs
import json
import logging
import os

import torch
from transformers import InputExample

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts


class MyPro(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        file_path = os.path.join(data_dir, 'data_train.txt')  # 读入数据
        f = open(file_path, 'r', encoding='utf-8')  # 指定读入数据格式及编码
        train_data = []
        index = 0
        for line in f.readlines():
            guid = "train-%d" % index
            line = line.replace('\n', '').split('\t')
            print(line)
            text_a = str(line[1])
            label = str(line[0])
            train_data.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
        return train_data

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        file_path = os.path.join(data_dir, 'data_val.txt')  # 读入数据
        f = open(file_path, 'r', encoding='utf-8')  # 指定读入数据格式及编码
        val_data = []
        index = 0
        for line in f.readlines():
            guid = "val-%d" % index
            line = line.replace('\n', '').split('\t')
            text_a = str(line[1])
            label = str(line[0])
            val_data.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
        return val_data

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        file_path = os.path.join(data_dir, 'data_test.txt')  # 读入数据
        f = open(file_path, 'r', encoding='utf-8')  # 指定读入数据格式及编码
        test_data = []
        index = 0
        for line in f.readlines():
            guid = "test-%d" % index
            line = line.replace('\n', '').split('\t')
            text_a = str(line[1])
            label = str(line[0])
            test_data.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
        return test_data

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ['0', '1']


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    """Loads a data file into a list of `InputBatch`s.

    Args:
        examples      : [List] 输入样本，包括question, label, index
        label_list    : [List] 所有可能的类别，可以是int、str等，如['book', 'city', ...]
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法

    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and show_exp:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features
