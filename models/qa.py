from torch import nn


class QAModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.
    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, batch):
        if self.config.get("roundwise", False):
            encoder_output = []
            batch_size, num_rounds, max_seq_len = batch["ctx_ans"].size()
            for round in range(num_rounds):
                rnd_info = {}
                rnd_info["ctx_ques"] = batch["ctx_ques"][:, round, :]
                rnd_info["ctx_ans"] = batch["ctx_ans"][:, round, :]
                rnd_info["ctx_hist"] = batch["ctx_hist"][:, :round+1, :].contiguous()
                rnd_info[""]
                enc_out = self.encoder(rnd_info)
                encoder_output.append(enc_out.unsqueeze(1))
            encoder_output = torch.cat(encoder_output, dim=1)

        else:
            encoder_output = self.encoder(batch)
        return encoder_output