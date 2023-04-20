import torch
from torch import nn
from torch.distributions import Categorical


class GenerativeDecoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.word_embedding_size = config["word_embedding_size"]
        self.rnn_layers = config["rnn_num_layers"]
        self.rnn_hidden_size = config["rnn_hidden_size"]
        self.vocabulary = vocabulary
        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.answer_rnn = nn.LSTM(
            config["word_embedding_size"],
            self.rnn_hidden_size,
            self.rnn_layers,
            batch_first=True,
            dropout=config["dropout"],
        )

        self.lstm_to_words = nn.Linear(
            self.rnn_hidden_size, len(vocabulary)
        )

        self.dropout = nn.Dropout(p=config["dropout"])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    # def forward(self, encoder_output, batch):
    #     """Given `encoder_output`, learn to autoregressively predict
    #     ground-truth answer word-by-word during training.
    #     During evaluation, assign log-likelihood scores to all answer options.
    #     Parameters
    #     ----------
    #     encoder_output: torch.Tensor
    #         Output from the encoder through its forward pass.
    #         (batch_size, num_rounds, lstm_hidden_size)
    #     """

    #     if self.training:

    #         ans_in = batch["gt_ans_in"]
    #         batch_size, num_rounds, max_sequence_length = ans_in.size()

    #         ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)

    #         # shape: (batch_size * num_rounds, max_sequence_length,
    #         #         word_embedding_size)
    #         ans_in_embed = self.word_embed(ans_in)

    #         # reshape encoder output to be set as initial hidden state of LSTM.
    #         # shape: (lstm_num_layers, batch_size * num_rounds,
    #         #         lstm_hidden_size)
    #         init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
    #         init_hidden = init_hidden.repeat(
    #             self.rnn_layers, 1, 1
    #         )
    #         init_cell = torch.zeros_like(init_hidden)

    #         # shape: (batch_size * num_rounds, max_sequence_length,
    #         #         lstm_hidden_size)
    #         ans_out, (hidden, cell) = self.answer_rnn(
    #             ans_in_embed, (init_hidden, init_cell)
    #         )
    #         ans_out = self.dropout(ans_out)

    #         # shape: (batch_size * num_rounds, max_sequence_length,
    #         #         vocabulary_size)
    #         ans_word_scores = self.lstm_to_words(ans_out)
    #         return ans_word_scores

    #     else:

    #         ans_in = batch["opt_in"]
    #         batch_size, num_rounds, num_options, max_sequence_length = (
    #             ans_in.size()
    #         )

    #         ans_in = ans_in.view(
    #             batch_size * num_rounds * num_options, max_sequence_length
    #         )

    #         # shape: (batch_size * num_rounds * num_options, max_sequence_length
    #         #         word_embedding_size)
    #         ans_in_embed = self.word_embed(ans_in)

    #         # reshape encoder output to be set as initial hidden state of LSTM.
    #         # shape: (lstm_num_layers, batch_size * num_rounds * num_options,
    #         #         lstm_hidden_size)
    #         init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
    #         init_hidden = init_hidden.repeat(1, 1, num_options, 1)
    #         init_hidden = init_hidden.view(
    #             1, batch_size * num_rounds * num_options, -1
    #         )
    #         init_hidden = init_hidden.repeat(
    #             self.rnn_layers, 1, 1
    #         )
    #         init_cell = torch.zeros_like(init_hidden)

    #         # shape: (batch_size * num_rounds * num_options,
    #         #         max_sequence_length, lstm_hidden_size)
    #         ans_out, (hidden, cell) = self.answer_rnn(
    #             ans_in_embed, (init_hidden, init_cell)
    #         )

    #         # shape: (batch_size * num_rounds * num_options,
    #         #         max_sequence_length, vocabulary_size)
    #         ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))

    #         # shape: (batch_size * num_rounds * num_options,
    #         #         max_sequence_length)
    #         target_ans_out = batch["opt_out"].view(
    #             batch_size * num_rounds * num_options, -1
    #         )

    #         # shape: (batch_size * num_rounds * num_options,
    #         #         max_sequence_length)
    #         ans_word_scores = torch.gather(
    #             ans_word_scores, -1, target_ans_out.unsqueeze(-1)
    #         ).squeeze()
    #         ans_word_scores = (
    #             ans_word_scores * (target_ans_out > 0).float().cuda()
    #         )  # ugly

    #         ans_scores = torch.sum(ans_word_scores, -1)
    #         ans_scores = ans_scores.view(batch_size, num_rounds, num_options)

    #         return ans_scores

    def forward(self, encoder_output, batch, decode=False, **kwargs):
        """Given `encoder_output`, learn to autoregressively predict
        ground-truth answer word-by-word during training.

        During evaluation, assign log-likelihood scores to all answer options.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        if self.training:

            ans_in = batch["gt_ans_in"]
            batch_size, num_rounds, max_sequence_length = ans_in.size()

            ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            init_hidden = init_hidden.repeat(
                self.rnn_layers, 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )
            ans_out = self.dropout(ans_out)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         vocabulary_size)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores

        elif not decode:
            ans_in = batch["opt_in"]
            batch_size, num_rounds, num_options, max_sequence_length = (
                ans_in.size()
            )

            ans_in = ans_in.view(
                batch_size * num_rounds * num_options, max_sequence_length
            )

            # shape: (batch_size * num_rounds * num_options, max_sequence_length
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds * num_options,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
            init_hidden = init_hidden.repeat(1, 1, num_options, 1)
            init_hidden = init_hidden.view(
                1, batch_size * num_rounds * num_options, -1
            )
            init_hidden = init_hidden.repeat(
                self.rnn_layers, 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (init_hidden, init_cell)
            )

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, vocabulary_size)
            ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            target_ans_out = batch["opt_out"].view(
                batch_size * num_rounds * num_options, -1
            )

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            ans_word_scores = torch.gather(
                ans_word_scores, -1, target_ans_out.unsqueeze(-1)
            ).squeeze()
            ans_word_scores = (
                ans_word_scores * (target_ans_out > 0).float().cuda()
            )  # ugly

            ans_scores = torch.sum(ans_word_scores, -1)
            ans_scores = ans_scores.view(batch_size, num_rounds, num_options)

            return ans_scores
        else:
            return self.forward_decode(encoder_output, kwargs.get('max_seq_len', 15), kwargs.get('inference', 'sample'))

    
    def forward_decode(self, encoder_output, max_seq_len=20, inference='sample'):
        """
        Return:
            gen_samples: (batch_size, num_rounds, max_seq_len+1)
            sample_lens: (batch_size, num_rounds)
        """
        max_len = max_seq_len + 1 # Extra <END> token
        batch_size, num_rounds, _ = encoder_output.size()

        seq = torch.full((batch_size * num_rounds, max_len + 1), self.vocabulary.EOS_INDEX, dtype=torch.long)
        seq[:, 0] = self.vocabulary.SOS_INDEX
        
        sample_lens = torch.zeros(batch_size * num_rounds, dtype=torch.long)
        unit_cols = torch.ones(batch_size * num_rounds, dtype=torch.long)
        mask = torch.zeros(seq.size(), dtype=torch.uint8)

        device = encoder_output.get_device()
        seq = seq.to(device)
        sample_lens = sample_lens.to(device)
        unit_cols = unit_cols.to(device)
        mask = mask.to(device)

        # if self.use_cuda:
        #     seq = seq.cuda()
        #     sample_lens = sample_lens.cuda()
        #     unit_cols = unit_cols.cuda()
        #     mask = mask.cuda()

        # if enc_states is None:
        #     dec_hidden = self.init_hidden(enc_out, ques_hidden)
        # else:
        #     dec_hidden = enc_states

        init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
        init_hidden = init_hidden.repeat(
            self.rnn_layers, 1, 1
        )
        init_cell = torch.zeros_like(init_hidden)

        samples = []
        saved_log_probs = []

        (hidden, cell) = (init_hidden, init_cell)

        for t in range(max_seq_len):
            ans_in = seq[:, t:t+1]
            # shape: (batch_size * num_rounds, 1, word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # log_prob, dec_hidden = self.rnn_step(seq[:, t:t+1], dec_hidden, enc_out)
            # shape: (batch_size * num_rounds, 1, lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                ans_in_embed, (hidden, cell)
            )

            # shape: (batch_size * num_rounds, vocabulary_size)
            log_prob = self.logsoftmax(self.lstm_to_words(ans_out.squeeze(1)))

            # Explicitly removing padding token (index 0) and <S> token
            # (index 1) from logProbs so that they are never sampled.
            if t > 0:
                log_prob = log_prob[:, 2:]
            elif t == 0:
                # Additionally, remove </S> token from the first sample
                # to prevent the sampling of an empty sequence.
                log_prob = log_prob[:, 3:]

            # This also shifts end token index back by 1
            # END_TOKEN_IDX = self.vocabulary.EOS_INDEX - 1
            probs = torch.exp(log_prob)
            if inference == 'sample':
                categorical_dist = Categorical(probs)
                sample = categorical_dist.sample()
                saved_log_probs.append(categorical_dist.log_prob(sample))
                sample = sample.unsqueeze(-1)
            elif inference == 'greedy':
                sample = log_prob.topk(1)[1]
            else:
                raise ValueError(
                    "Invalid inference type: '{}'".format(inference))

            # Compensating for removed padding token and <S> prediction earlier
            # shape: (batch_size * num_rounds, 1)
            # (t == 0) W0 W1
            # (t > 0) </S> W0 W1
            sample = sample + 2
            if t == 0:
                sample = sample + 1

            samples.append(sample)
            seq.data[:, t+1] = sample.squeeze(1).data
            # Marking spots where <END> token is generated
            mask[:, t] = sample.squeeze(1).data.eq(self.vocabulary.EOS_INDEX)

            # Compensating for shift in <END> token index
            # sample.data.masked_fill_(mask[:, t].unsqueeze(1), self.vocabulary.EOS_INDEX)
        
        mask[:, max_seq_len].fill_(1)

        for t in range(max_len):
            # Zero out the spots where end token is reached
            unit_cols.masked_fill_(mask[:, t], 0)
            # Update mask
            mask[:, t] = unit_cols
            # Add +1 length to all un-ended sequences
            sample_lens = sample_lens + unit_cols

        # Adding <START> to generated answer lengths for consistency
        sample_lens = sample_lens + 1

        start_col = sample.data.new(sample.size()).fill_(self.vocabulary.SOS_INDEX)
        start_col.requires_grad = False
        # start_col = torch.tensor(start_col, requires_grad=False)

        gen_samples = [start_col] + samples
        gen_samples = torch.cat(gen_samples, dim=1).view(batch_size, num_rounds, -1)
        sample_lens = sample_lens.view(batch_size, num_rounds)
        
        return gen_samples, sample_lens, saved_log_probs
        