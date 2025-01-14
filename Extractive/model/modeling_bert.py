# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 21/2/9 -*-

from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch
from model.Layers import MultiLinearLayer
from transformers.modeling_bert import BertConfig
from model.BiLSTM import BiLSTM
from model.RTransformer import RTransformer
import torch.nn as nn


class BertForConversationalQuestionAnswering(BertPreTrainedModel):
    def __init__(
            self,
            config,
            n_layers=2,
            activation='relu',
            beta=100,
    ):
        super(BertForConversationalQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # self.bert = BertModel.from_pretrained(
        #     pretrained_model_name_or_path=r'E:\Internship\ConvQA\Reference\transformers-coqa\bert-base-chinese/',
        #     output_hidden_states=True
        # )

        # dynamic weight fusion
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dense_final = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(True))
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)), requires_grad=True)
        self.pool_weight = nn.Parameter(torch.ones((2, 1, 1, 1)), requires_grad=True)

        # middle layer
        hidden_size = config.hidden_size
        self.num_labels = 768
        self.bilstm = BiLSTM(self.num_labels, embedding_size=hidden_size, hidden_size=256, num_layers=2, dropout=0.3, with_ln=True)
        self.rtransformer = RTransformer(tag_size=self.num_labels, dropout=0.3, d_model=hidden_size, ksize=32, h=4)
        self.logits_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 2, activation)
        self.unk_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 1, activation)
        self.attention_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 1, activation)
        self.yn_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 2, activation)
        self.beta = beta
        self.init_weights()
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.dym_weight)

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            rational_mask=None,
            cls_idx=None,
            head_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        # print(outputs)
        # print(len(outputs))

        # last layer hidden-states of sequence, first token:classification token
        sequence_output, pooled_output = outputs  # sequence_output即为final_hidden
        # # bilstm
        # final_hidden = self.bilstm.get_lstm_features(sequence_output.transpose(1, 0), attention_mask.transpose(1, 0))

        # R-transformers
        final_hidden = self.rtransformer(sequence_output, attention_mask).transpose(1, 0)
        final_hidden = final_hidden.transpose(0, 1)  # [batch_size, seq_len, hidden_size]

        # final_hidden, pooled_output = outputs
        # attention layer to cal logits
        attention = self.attention_l(final_hidden).squeeze(-1)
        attention.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        attention = F.softmax(attention, dim=-1)
        attention_pooled_output = (attention.unsqueeze(-1) * final_hidden).sum(dim=-2)

        # on to find answer in the article
        segment_mask = token_type_ids.type(final_hidden.dtype)
        # rational_logits = rational_logits.squeeze(-1) * segment_mask

        # get span logits
        logits = self.logits_l(final_hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        # start_logits, end_logits = start_logits * rational_logits, end_logits * rational_logits
        start_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        end_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        # cal unkown/yes/no logits
        unk_logits = self.unk_l(pooled_output)
        yn_logits = self.yn_l(attention_pooled_output)
        yes_logits, no_logits = yn_logits.split(1, dim=-1)

        # start_positions and end_positions is None when evaluate
        # return loss during training
        # return logits during evaluate
        if start_positions is not None and end_positions is not None:

            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx

            new_start_logits = torch.cat((yes_logits, no_logits, unk_logits, start_logits), dim=-1)
            new_end_logits = torch.cat((yes_logits, no_logits, unk_logits, end_logits), dim=-1)

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)

            # # rational part
            # alpha = 0.25
            # gamma = 2.
            #
            # # use rational span to help calculate loss
            # rational_mask = rational_mask.type(final_hidden.dtype)
            # rational_loss = -alpha * ((1 - rational_logits) ** gamma) * rational_mask * torch.log(
            #     rational_logits + 1e-7) \
            #                 - (1 - alpha) * (rational_logits ** gamma) * (1 - rational_mask) * \
            #                 torch.log(1 - rational_logits + 1e-7)
            #
            # rational_loss = (rational_loss * segment_mask).sum() / segment_mask.sum()

            # total_loss = (start_loss + end_loss) / 2 + rational_loss * self.beta  # mutil-loss

            total_loss = (start_loss + end_loss) / 2
            return total_loss

        return start_logits, end_logits, yes_logits, no_logits, unk_logits


if __name__ == '__main__':
    bert_config = BertConfig.from_json_file(r'../bert-base-chinese/config.json')
    model = BertForConversationalQuestionAnswering(config=bert_config)
    for n, p in model.named_parameters():
        print(n)
