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
parser.add_argument('--dialog-json', default='../data/impl_dial/world_large_nex_4000/impl_dial_dev_v0.1.json')
parser.add_argument('--vocab-json', default='train_vocab.json')
parser.add_argument('--overfit', action='store_true')
parser.add_argument('--config-yml', default='./configs/memnn.yml')

parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--mode', default='cancel', choices=["qa", 'cancel'])

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
    num_examples=100,
    return_explicit=True,
    return_options=args.mode == 'cancel',
    return_qa=args.mode == 'qa',
    concat_history=config.get("concat_history", True),
    add_boundary_toks=True # True if generate
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

load_pthpath = os.path.join(args.save_dirpath, config["name"], args.ckpt)
components = torch.load(load_pthpath)
model.load_state_dict(components['model'])
print('Loaded checkpoint {}'.format(load_pthpath))

model.eval()
output_json = []

def to_words(tensor):
    return ' '.join(val_dataset.vocabulary.to_words(tensor.detach().tolist()))

for _, batch in enumerate(tqdm(val_dataloader)):
    for key in batch:
        batch[key] = batch[key].to(device)
    with torch.no_grad():
        gen_samples, sample_lens = model(batch, decode=True, max_seq_len=15, inference='sample')
        for i in range(len(batch["dialog_id"])):
            num_rounds = batch["num_rounds"][i]
            new_dialog = {
                'dialog_id': int(batch['dialog_id'][i]),
                'dialogs': [None] * int(num_rounds)
            }
            for j in range(num_rounds):
                ques_len = batch["ctx_ques_len"][i][j].cpu().data
                ques = to_words(batch["ctx_ques"][i][j][:ques_len])
                ans_len = batch["ctx_ans_len"][i][j].cpu().data
                ans = to_words(batch["ctx_ans"][i][j][1:ans_len-1])
                gt_ans_len = batch["gt_ans_len"][i][j].cpu().data
                gt_ans = to_words(batch["gt_ans_out"][i][j][1:gt_ans_len-2])
                gen_ans_len = sample_lens[i][j].cpu().data
                gen_ans = to_words(gen_samples[i][j][2:gen_ans_len-1])
                new_dialog["dialogs"][j] = {
                    'question': ques,
                    'answer': ans,
                    'gen_explicit': gen_ans,
                    'gt_explicit': gt_ans,
                }
            output_json.append(new_dialog)
        # if args.mode == "qa":
        # else:
with open(os.path.join(args.save_dirpath, config["name"], 'gen.json'), 'w') as f:
    json.dump(output_json, f)