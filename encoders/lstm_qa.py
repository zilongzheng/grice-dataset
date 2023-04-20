import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dynamic_rnn import DynamicRNN

class LSTMQAEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()

        self.rnn_hidden_size = config["rnn_hidden_size"]
        self.word_embedding_size = config["word_embedding_size"]
        self.rnn_layers = config["rnn_num_layers"]
        self.vocab_size = len(vocabulary)
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
        # self.ans_rnn = DynamicRNN(self.ans_rnn)
        self.vocabulary = vocabulary

        # self.answer_fusion = nn.Linear(self.rnn_hidden_size * 2, self.rnn_hidden_size)
        self.context_fusion = nn.Linear(self.rnn_hidden_size * 2, self.rnn_hidden_size)

        fusion_size = self.rnn_hidden_size + self.rnn_hidden_size
        self.embed_to_output = nn.Linear(self.rnn_hidden_size, self.vocab_size)


    def forward(self, batch):
        # (bs, num_rounds, 15)
        # ctx_ques = batch["ctx_ques"]
        # ctx_ans = batch["ctx_ans"]

        # (bs, 3, max_seq_len)
        ctx_query = batch["qa_ques"]
        ctx_query_len = batch["qa_ques_len"]
        # (bs, num_rounds, 15 * 2 * 10)
        # ctx_hist = batch["ctx_ques_ans"]
        # ctx_hist_len = batch["ctx_ques_ans_len"]

        ctx_ques_ans = batch["ctx_ques_ans"]
        ctx_ques_ans_len = batch["ctx_ques_ans_len"]

        # (bs, num_rounds, max_seq_len)
        # ctx_ques = batch["ctx_ques"]
        # ctx_ques_len = batch["ctx_ques_len"]

        # (bs, num_rounds, max_seq_len)
        # ctx_ans = batch["ctx_ans"]
        # ctx_ans_len = batch["ctx_ans_len"]

        # gt_ans = batch["gt_ans_in"]
        # gt_ans_len = batch["gt_ans_len"]

        # full_ans = batch["full_ans"]
        # full_ans_len =  batch["full_ans_len"]

        # print(self.vocabulary.to_words(full_ans[0][0].cpu().data.numpy()))
        # for i in range(10):
        #     print(self.vocabulary.to_words(ctx_hist[0][i].cpu().data.numpy()))
        #     print(self.vocabulary.to_words(gt_ans[0][i].cpu().data.numpy()))
        # print(self.vocabulary.to_words(ctx_query.cpu().data.numpy()[0][0]))


        batch_size, num_qa, max_seq_len = ctx_query.size()
        ctx_query = ctx_query.view(batch_size * num_qa, max_seq_len)

        # (bs * 10, 15, embed_size)
        query_embed = self.word_embed(ctx_query)
        query_embed = self.dropout(query_embed)
        _, (query_embed, _) = self.ques_rnn(query_embed, ctx_query_len.view(-1))
        # query_embed = query_embed.view(batch_size, num_qa, self.rnn_hidden_size)

        num_rounds = ctx_ques_ans.size(1)
        # max_hist_len = ctx_hist.size(2)
        ctx_hist = ctx_ques_ans.view(batch_size * num_rounds, max_seq_len * 2)
        hist_embed = self.word_embed(ctx_hist)
        hist_embed = self.dropout(hist_embed)
        hist_embed = hist_embed.view(batch_size, num_rounds, max_seq_len * 2, self.word_embedding_size)
        hist_state = None
        for rnd in range(num_rounds):
            hist = hist_embed[:, rnd, :, :]
            hist_len = ctx_ques_ans_len[:, rnd]
            hist_state, (hist_enc, _) = self.hist_rnn(hist, hist_len, initial_state=hist_state)
            # print(hist_state[0].size())
        hist_embed = hist_enc
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_qa, 1)
        hist_embed = hist_embed.view(batch_size * num_qa, self.rnn_hidden_size)

        ques_att = query_embed * hist_embed
        fused_vector = self.context_fusion(torch.cat([query_embed, ques_att], dim=-1))
        fused_vector = self.dropout(fused_vector)
        fused_embedding = self.embed_to_output(fused_vector).view(batch_size, num_qa, self.vocab_size)

        # ctx_ques = ctx_ques.view(batch_size * num_rounds, max_seq_len)
        # ques_embed = self.word_embed(ctx_ques)
        # _, (ques_embed, _) = self.hist_rnn(ques_embed, ctx_ques_len.view(-1))

        # ctx_ans = ctx_ans.view(batch_size * num_rounds, max_seq_len)
        # ans_embed = self.word_embed(ctx_ans)
        # _, (ans_embed, _) = self.ans_rnn(ans_embed, ctx_ans_len.view(-1))


        # ctx_gt_ans = full_ans.view(batch_size * num_rounds, max_seq_len * 2)
        # gt_ans_embed = self.word_embed(ctx_gt_ans)
        # _, (gt_ans_embed, _) = self.ans_rnn(gt_ans_embed, full_ans_len.view(-1))
        # ans_embed = gt_ans_embed

        # ans_embed = torch.cat([ans_embed, gt_ans_embed], dim=-1)
        # ans_embed = self.answer_fusion(ans_embed)
        # ans_embed = gt_ans.view(batch_size * num_rounds, max_seq_len)
        # ctx_ans = ctx_ans.view(batch_size * num_rounds, max_seq_len)
        # ans_embed = self.word_embed(ans_embed)
        # print(ans_embed.size())
        # print(gt_ans_len.size())
        # (batch_size * num_rounds, rnn_hidden_size)
        # _, (ans_embed, _) = self.ans_rnn(ans_embed, gt_ans_len.view(-1))


        # ctx_hist = ctx_hist.view(batch_size * num_rounds, max_hist_len)
        # hist_embed = self.word_embed(ctx_hist)
        # hist_embed = self.dropout(hist_embed)

        # print(hist_embed.size())
        # print(ctx_hist_len.size())
        # shape: (batch_size * num_rounds, lstm_hidden_size)
        # _, (hist_embed, _) = self.hist_rnn(hist_embed, ctx_hist_len.view(-1))

        # print(hist_embed.size())
        # print(ans_embed.size())
        # hist_embed = torch.cat([hist_embed, ans_embed], dim=-1)
        # hist_embed = self.context_fusion(hist_embed)
        # hist_embed = self.context_fusion(torch.cat([ques_embed, gt_ans_embed], dim=-1))
        # hist_embed = self.dropout(hist_embed)
        # hist_embed = hist_embed.view(batch_size, num_rounds, self.rnn_hidden_size)

        # hist_embed = hist_embed.unsqueeze(1).repeat(1, num_qa, 1, 1)
        # hist_embed = hist_embed.view(batch_size * num_qa, num_rounds, self.rnn_hidden_size)

        # query_embed = query_embed.view(batch_size * num_qa, 1, self.rnn_hidden_size)

        # # (batch_size * num_qa, 1, num_rounds)
        # hist_att = torch.bmm(query_embed, hist_embed.transpose(1, 2))
        # hist_att = F.sigmoid(hist_att)
        # hist_att = torch.bmm(hist_att, hist_embed)


        # fused_vector = torch.cat((query_embed, hist_att), 2).squeeze(1)
        # fused_vector = self.dropout(fused_vector)
        # fused_embedding = self.fusion(fused_vector)
        # fused_embedding = F.log_softmax(fused_embedding.view(batch_size, num_qa, self.vocab_size), dim=-1)
        return fused_embedding
