import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel


class Bert_BiLSTM_Attention(nn.Module):
    def __init__(self, config):
        super(Bert_BiLSTM_Attention).__init__()
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dims
        self.dropout = config.dropout_prob

        self.Bert = BertModel.from_pretrained(config.bert_model, config=config.bert_model_config)

        # BiLSTM
        # input of shape(batch, seq_len, input_size)
        self.BiLSTM = nn.LSTM(input_size=config.embedding_size,
                              hidden_size=config.hidden_dims,
                              num_layers=config.num_layers,
                              batch_first=True,
                              bidirectional=True)

        self.line = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.w = nn.Parameter(torch.rand(self.hidden_dim * 2), requires_grad=True)

        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_labels)

    def forward(self, inputs_id, attention_mask=None, labels=None):
        """
        :param inputs_id: inputs_id of shape (batch_size, sentence_length, input_size)
        :param attention_mask:
        :param labels: labels of shape (batch_size, labels)
        :return:
        """
        Embedding = self.Bert(inputs_id)
        # output of shape (batch, seq_len, num_directions * hidden_size)
        lstm_output, _ = self.BiLSTM(Embedding[0])
        lstm_output = lstm_output * attention_mask.unsqueeze(-1).repeat(1, 1, self.hidden_dim * 2).detach().float()
        # attention_weights of shape (batch_sizes, max_seq_length, 1)
        att_out, attention_weights = self.attention(lstm_output)

        att_out = self.dropout(att_out)
        att_out = torch.sum(att_out, 1)

        logits = self.classifier(att_out)

        best_path = logits.argmax(-1)

        outputs = (best_path, attention_weights)

        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def attention(self, H):
        M = torch.tanh(self.line(H))
        a = torch.matmul(M, self.w)
        a = F.softmax(a, dim=1).unsqueeze(-1)
        self.att_out = a.data
        out = H * a
        # out = torch.tanh(out)
        return out, a


class Bert_Attention(nn.Module):
    def __init__(self, config):
        super(Bert_Attention, self).__init__()
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_dims
        self.dropout = config.dropout_prob

        self.Bert = BertModel.from_pretrained(config.bert_model, config=config.bert_model_config)

        self.line = nn.Linear(config.embedding_size, self.hidden_dim)
        self.w = nn.Parameter(torch.rand(self.hidden_dim), requires_grad=True)

        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(config.embedding_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        #  The shape of Embedding is (batch_size, max_seq_length, Embedding_size)

        outputs = self.Bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        Embedding = outputs[0]

        out, attention_weights = self.attention(Embedding)

        out = self.dropout(out)
        out = torch.sum(out, 1)

        logits = self.classifier(out)

        best_path = logits.argmax(-1)

        outputs = (best_path, attention_weights)

        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def attention(self, H):
        M = torch.tanh(self.line(H))
        a = torch.matmul(M, self.w)
        a = F.softmax(a, dim=1).unsqueeze(-1)
        out = H * a
        return out, a


class BertRNN_Att(BertPreTrainedModel):

    def __init__(self, config, model_configs):
        super(BertRNN_Att, self).__init__(config)
        self.num_labels = model_configs.num_labels
        self.rnn_layers = model_configs.num_layers
        self.hidden_dim = model_configs.hidden_dims

        self.bert = BertModel(config)
        self.lstm = nn.GRU(input_size=config.hidden_size,
                           hidden_size=self.hidden_dim,
                           num_layers=self.rnn_layers,
                           bias=True,
                           batch_first=True,
                           # dropout = model_configs['train']['dropout_rate'],
                           bidirectional=True)

        self.line = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)

        self.w = nn.Parameter(torch.rand(self.hidden_dim * 2))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]
        lstm_output, _ = self.lstm(sequence_output)

        lstm_output = lstm_output * attention_mask.unsqueeze(-1).repeat(1, 1, self.hidden_dim * 2).detach().float()
        att_out, attention_weights = self.attention(lstm_output)

        att_out = self.dropout(att_out)
        att_out = torch.sum(att_out, 1)

        logits = self.classifier(att_out)

        best_path = logits.argmax(-1)

        outputs = (best_path, attention_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def attention(self, H):
        M = torch.tanh(self.line(H))
        a = torch.matmul(M, self.w)
        a = F.softmax(a, dim=1).unsqueeze(-1)
        out = H * a
        # out = torch.tanh(out)
        return out, a
