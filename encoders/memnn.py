import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dynamic_rnn import DynamicRNN

class MemNN(nn.Module):
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
        self.ans_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["rnn_hidden_size"],
            config["rnn_num_layers"],
            batch_first=True,
            dropout=config["dropout"] if self.rnn_layers > 0 else 0
        )

        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
        self.ans_rnn = DynamicRNN(self.ans_rnn)

        self.hist_att = nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size)
        self.q_att = nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size)
        self.ans_to_out = nn.Linear(2 * self.rnn_hidden_size, self.rnn_hidden_size)

        self.mask = torch.ones((10, 10)).byte()
        for i in range(10):
            for j in range(10):
                if j <= i:
                    self.mask[i][j] = 0


    def forward(self, batch, mask=None):
        # (bs, num_rounds, 15)
        # ctx_ques = batch["ctx_ques"]
        # (bs, num_rounds, 15)
        ctx_ques = batch["ctx_ques"]
        ctx_ques_len = batch["ctx_ques_len"]

        ctx_ans = batch["ctx_ans"]
        ctx_ans_len = batch["ctx_ans_len"]

        # (bs, num_rounds, 15 * 2)
        ctx_hist = batch["ctx_hist"]

        # (bs, num_rounds, 30)
        # ctx_query = torch.cat([ctx_ques, ctx_ans], dim=-1)

        batch_size, num_rounds, max_seq_len = ctx_ques.size()

        ctx_query = ctx_ques.view(batch_size * num_rounds, max_seq_len)

        # (bs * 10, 15, embed_size)
        query_embed = self.word_embed(ctx_query)
        _, (query_embed, _) = self.ques_rnn(query_embed, ctx_ques_len.view(-1))
        query_embed = query_embed.view(batch_size, num_rounds, self.rnn_hidden_size)


        ctx_ans = ctx_ans.view(batch_size * num_rounds, max_seq_len)
        ans_embed = self.word_embed(ctx_ans)
        _, (ans_embed, _) = self.ans_rnn(ans_embed, ctx_ans_len.view(-1))
        # ans_embed = ans_embed.view(batch_size, num_rounds, self.rnn_hidden_size)


        ctx_hist = ctx_hist.view(batch_size * num_rounds, max_seq_len * 2)
        hist_embed = self.word_embed(ctx_hist)

        # shape: (batch_size * num_rounds, lstm_hidden_size)
        _, (hist_embed, _) = self.hist_rnn(hist_embed, batch["ctx_hist_len"].view(-1))

        hist_embed = hist_embed.view(batch_size, num_rounds, self.rnn_hidden_size)
        # (batch_size, num_rounds, rnn_hidden_size) * (batch_size, rnn_hidden_size, num_rounds) -> (bs, num_rounds, num_rounds)
        qh = torch.bmm(query_embed, hist_embed.transpose(1, 2))
        mask = self.mask.unsqueeze(0).repeat(batch_size, 1, 1).to(ctx_query.device)
        # maskout hist[j] > 
        qh.masked_fill_(mask, -1e9)
        qh = F.softmax(qh, dim=-1)

        h_att = torch.bmm(qh, hist_embed)
        h_att = self.dropout(h_att)

        h_att = torch.tanh(self.hist_att(h_att.view(batch_size * num_rounds, self.rnn_hidden_size)))

        qh2 = torch.add(h_att, query_embed.view(-1, self.rnn_hidden_size))
        qh2 = torch.tanh(self.q_att(qh2))

        ans = self.ans_to_out(torch.cat([qh2, ans_embed], dim=-1))

        return ans.view(batch_size, num_rounds, -1)
