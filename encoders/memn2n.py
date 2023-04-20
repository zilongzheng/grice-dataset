import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class MemN2N(nn.Module):

    def __init__(self, config, vocabulary):
        super(MemN2N, self).__init__()
        self.input_size = len(vocabulary)
        self.embed_size = config['word_embedding_size']
        self.memory_size = config['memory_size']
        self.num_hops = config['num_hops']
        self.use_bow = config['use_bow']
        self.use_lw = config['use_lw']
        self.use_ls = config['use_ls']
        self.vocab = vocabulary
        self.device = torch.device("cuda", 0)

        # create parameters according to different type of weight tying
        # pad = self.vocab.stoi['<pad>']
        self.A = nn.ModuleList([
            nn.Embedding(self.input_size, self.embed_size, padding_idx=vocabulary.PAD_INDEX)
        ])
        self.A[-1].weight.data.normal_(0, 0.1)
        self.C = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=vocabulary.PAD_INDEX)])
        self.C[-1].weight.data.normal_(0, 0.1)
        if self.use_lw:
            for _ in range(1, self.num_hops):
                self.A.append(self.A[-1])
                self.C.append(self.C[-1])
            self.B = nn.Embedding(self.input_size, self.embed_size, padding_idx=vocabulary.PAD_INDEX)
            self.B.weight.data.normal_(0, 0.1)
            self.out = nn.Parameter(
                I.normal_(torch.empty(self.input_size, self.embed_size), 0, 0.1))
            self.H = nn.Linear(self.embed_size, self.embed_size)
            self.H.weight.data.normal_(0, 0.1)
        else:
            for _ in range(1, self.num_hops):
                self.A.append(self.C[-1])
                self.C.append(nn.Embedding(self.input_size, self.embed_size, padding_idx=vocabulary.PAD_INDEX))
                self.C[-1].weight.data.normal_(0, 0.1)
            self.B = self.A[0]
            self.out = self.C[-1].weight

        # temporal matrix
        self.TA = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))
        self.TC = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))

    def forward(self, batch):

        # sen_size = query.shape[-1]
        # (bs, 3, max_seq_len)
        ctx_query = batch["qa_ques"]
        ctx_query_len = batch["qa_ques_len"]

        # (bs, num_rounds, max_seq_len)
        story = batch["ctx_ques_ans"]
        story_len = batch["ctx_ques_ans_len"]

        # (bs, num_rounds, max_seq_len)
        # ctx_ans = batch["ctx_ans"]
        # ctx_ans_len = batch["ctx_ans_len"]
        # print(ctx_query.size())
        batch_size, num_qa, sen_size = ctx_query.size()

        ctx_query = ctx_query.view(batch_size * num_qa, sen_size)

        story = story.unsqueeze(1).repeat(1, num_qa, 1, 1)
        num_rounds = story.size(2)
        story = story.view(batch_size * num_qa, num_rounds, sen_size * 2)
        # sen_size, embed_size
        query_w = self.compute_weights(sen_size)
        # (bs * num_qa, sen_size)
        state = (self.B(ctx_query) * query_w).sum(1)

        sen_size = story.shape[-1]
        story_w = self.compute_weights(sen_size)
        for i in range(self.num_hops):

            memory = (self.A[i](story.view(-1, sen_size)) * story_w).sum(1).view(
                *story.shape[:-1], -1)
            memory += self.TA
            output = (self.C[i](story.view(-1, sen_size)) * story_w).sum(1).view(
                *story.shape[:-1], -1)
            output += self.TC

            probs = (memory @ state.unsqueeze(-1)).squeeze()
            if not self.use_ls:
                probs = F.softmax(probs, dim=-1)
            response = (probs.unsqueeze(1) @ output).squeeze()
            if self.use_lw:
                state = self.H(response) + state
            else:
                state = response + state

        out = F.linear(state, self.out)
        print(out.size())
        return out

    def compute_weights(self, J):
        d = self.embed_size
        if self.use_bow:
            weights = torch.ones(J, d)
        else:
            func = lambda j, k: 1 - (j + 1) / J - (k + 1) / d * (1 - 2 * (j + 1) / J)    # 0-based indexing
            weights = torch.from_numpy(np.fromfunction(func, (J, d), dtype=np.float32))
        return weights.to(self.device) if torch.cuda.is_available() else weights