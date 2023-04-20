from torch import nn


class EncoderDecoderModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.
    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch, decode=False, **kwargs):
        encoder_output = self.encoder(batch)
        if decode:
            gen_samples, sample_lens, _ = self.decoder(encoder_output, batch, decode=True, **kwargs)
            return gen_samples, sample_lens
        else:
            decoder_output = self.decoder(encoder_output, batch)

        # decoder_output = self.decoder(encoder_output, batch)
        return decoder_output