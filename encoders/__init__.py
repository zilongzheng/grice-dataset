from encoders.lstm_encoder import LSTMEncoder
from encoders.memnn import MemNN
from encoders.entnet import EntNet
from encoders.lstm_qa import LSTMQAEncoder
from encoders.entnet_qa import EntNetQA
from encoders.memn2n import MemN2N
from encoders.memnn_qa import MemNNQA

def Encoder(model_config, *args):
    name_dec_map = {"lstm": LSTMEncoder, "memnn": MemNN, "entnet": EntNet, "lstmqa": LSTMQAEncoder, "entnetqa": EntNetQA, "memn2n": MemN2N, "memnnqa": MemNNQA}
    return name_dec_map[model_config["encoder"]](model_config, *args)
