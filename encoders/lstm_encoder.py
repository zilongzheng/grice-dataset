import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dynamic_rnn import DynamicRNN

class LSTMEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()

        self.rnn_hidden_size = config["rnn_hidden_size"]
        self.word_embedding_size = config["word_embedding_size"]
        self.rnn_layers = config["rnn_num_layers"]
        
        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )

        self.dropout = nn.Dropout(p=config["dropout"])

        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["rnn_hidden_size"],
            config["rnn_num_layers"],
            batch_first=True,
            dropout=config["dropout"] if self.rnn_layers > 0 else 0
        )
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["rnn_hidden_size"],
            config["rnn_num_layers"],
            batch_first=True,
            dropout=config["dropout"] if self.rnn_layers > 0 else 0
        )
        # self.ans_rnn = nn.LSTM(
        #     config["word_embedding_size"],
        #     config["rnn_hidden_size"],
        #     config["rnn_num_layers"],
        #     batch_first=True,
        #     dropout=config["dropout"] if self.rnn_layers > 0 else 0
        # )

        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
        # self.ans_rnn = DynamicRNN(self.ans_rnn)

        fusion_size = 2 * self.rnn_hidden_size
        self.fusion = nn.Linear(fusion_size, self.rnn_hidden_size)


    def forward(self, batch):
        # (bs, num_rounds, 15)
        # ctx_ques = batch["ctx_ques"]
        # ctx_ans = batch["ctx_ans"]

        # (bs, num_rounds, 15 * 2)
        ctx_query = batch["ctx_ques_ans"]
        # (bs, num_rounds, 15 * 2 * 10)
        ctx_hist = batch["ctx_hist"]

        batch_size, num_rounds, max_seq_len = ctx_query.size()
        ctx_query = ctx_query.view(batch_size * num_rounds, max_seq_len)

        # (bs * 10, 15, embed_size)
        query_embed = self.word_embed(ctx_query)
        _, (query_embed, _) = self.ques_rnn(query_embed, batch["ctx_ques_len"])

        # ctx_ans = ctx_ans.view(batch_size * num_rounds, max_seq_len)
        # ans_embed = self.word_embed(ctx_ans)
        # _, (ans_embed, _) = self.ques_rnn(ans_embed, batch["ctx_ans_len"])

        ctx_hist = ctx_hist.view(batch_size * num_rounds, max_seq_len * 10)
        hist_embed = self.word_embed(ctx_hist)

        # shape: (batch_size * num_rounds, lstm_hidden_size)
        _, (hist_embed, _) = self.hist_rnn(hist_embed, batch["ctx_hist_len"])

        fused_vector = torch.cat((query_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)
        fused_embedding = torch.tanh(self.fusion(fused_vector))
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)
        return fused_embedding
