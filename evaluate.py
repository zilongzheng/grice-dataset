import torch
import argparse
from dataset import bAbiImpDataset
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import json
from models.enc2dec import EncoderDecoderModel
from models.rndenc2dec import RoundWiseEncoderDecoderModel
from encoders import Encoder
from decoders import Decoder
from utils.checkpointing import CheckpointManager
from utils.metrics import SparseGTMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--dialog-json', default='../data/impl_dial/world_large_nex_4000/impl_dial_test_v0.1.json')
parser.add_argument('--vocab-json', default='train_vocab.json')
parser.add_argument('--overfit', action='store_true')
parser.add_argument('--config-yml', default='./configs/lstm_qa.yml')

parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--mode', default='qa', choices=["qa", 'cancel'])

parser.add_argument('--cpu-workers', default=4, type=int)

parser.add_argument('--save-dirpath', default='./output')
parser.add_argument('--ckpt', default='checkpoint_best.pth')

args = parser.parse_args()

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

config = yaml.load(open(args.config_yml, 'r'))


val_dataset = bAbiImpDataset(
    args.dialog_json,
    args.vocab_json,
    max_sequence_length=config["model"]["max_seq_len"],
    num_examples=args.batch_size if args.overfit else None,
    return_explicit=True,
    return_options=args.mode == 'cancel',
    return_qa=args.mode == 'qa',
    concat_history=config.get("concat_history", True),
    add_boundary_toks= config["model"].get("decoder") == "gen"  # True if generate
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.cpu_workers,
)

device = (
    torch.device("cuda", args.gpu_id)
    if args.gpu_id >= 0
    else torch.device("cpu")
)
torch.cuda.set_device(device)

# args.save_dirpath = os.path.join(args.save_dirpath, config["name"], 'eval')
sparse_metrics = SparseGTMetrics()
print(config)

if args.mode == "qa":
    model = Encoder(config["model"], val_dataset.vocabulary).to(device)
else:
    encoder = Encoder(config["model"], val_dataset.vocabulary)
    decoder = Decoder(config["model"], val_dataset.vocabulary)
    decoder.word_embed = encoder.word_embed
    if config["model"]["model"] == "enc2dec":
        model = EncoderDecoderModel(encoder, decoder).to(device)
    else:
        model = RoundWiseEncoderDecoderModel(encoder, decoder).to(device)

if "inference_encoder" in config["model"]:
    new_config = config["model"].copy()
    new_config["encoder"] = config["model"]["inference_encoder"]
    new_config["decoder"] = config["model"]["inference_decoder"]
    inference_encoder = Encoder(new_config, val_dataset.vocabulary)
    inference_deocder = Decoder(new_config, val_dataset.vocabulary)
    inference_model = EncoderDecoderModel(inference_encoder, inference_deocder)
    inference_componet = torch.load(config["model"]["inference_path"])
    print('Loaded inference model {}'.format(config["model"]["inference_path"]))
    inference_model.load_state_dict(inference_componet["model"])
    model.inference_encoder = inference_encoder

load_pthpath = os.path.join(args.save_dirpath, config["name"], args.ckpt)
components = torch.load(load_pthpath)
model.load_state_dict(components['model'])
print('Loaded checkpoint {}'.format(load_pthpath))

model.eval()
if args.mode == "qa":
    correct = 0
    count = 0
for _, batch in enumerate(tqdm(val_dataloader)):
    for key in batch:
        batch[key] = batch[key].to(device)
    with torch.no_grad():
        output = model(batch)
    if args.mode == "qa":
        pred = output.view(-1, output.size(-1))

        pred_idx = pred.max(1)[1]
        gt_ans = batch["qa_ans"].view(-1)
        correct += torch.sum(pred_idx == gt_ans).item()
        count += int(pred.size(0))
    else:
        
        for i in range(len(batch["dialog_id"])):
            num_rounds = batch["num_rounds"][i]

        # if args.mode == "qa":
        # else:
            sparse_metrics.observe(output[i][:num_rounds].unsqueeze(0), batch["ans_ind"][i][:num_rounds].unsqueeze(0))
all_metrics = {}
if args.mode == "qa":
    all_metrics.update({"acc": float(correct)/count*100, "correct": correct, "total": count})
else:
    all_metrics.update(sparse_metrics.retrieve(reset=True))
for metric_name, metric_value in all_metrics.items():
    print("{}: {}".format(metric_name, metric_value))