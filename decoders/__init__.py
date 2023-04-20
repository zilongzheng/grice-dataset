from decoders.decoder import LSTMDecoder
from decoders.gen import GenerativeDecoder

def Decoder(model_config, *args):
    name_dec_map = {"lstm": LSTMDecoder, "gen": GenerativeDecoder}
    return name_dec_map[model_config["decoder"]](model_config, *args)
