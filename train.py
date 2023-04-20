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
# from utils.checkpointing import CheckpointManager
from utils.metrics import SparseGTMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--dialog-json-train', default='../data/impl_dial/world_large_nex_4000/impl_dial_train_v0.1.json')
parser.add_argument('--dialog-json-dev', default='../data/impl_dial/world_large_nex_4000/impl_dial_dev_v0.1.json')
parser.add_argument('--vocab-json', default='train_vocab.json')
parser.add_argument('--overfit', action='store_true')
parser.add_argument('--config-yml', default='./configs/entnet.yml')
parser.add_argument('--validate', action='store_true')

parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--mode', default='cancel', choices=["qa", 'cancel'])

parser.add_argument('--cpu-workers', default=4, type=int)
parser.add_argument('--log_step', default=1, type=int)

parser.add_argument('--save-dirpath', default='./output')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--ckpt', default='checkpoint_latest.pth')

args = parser.parse_args()

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

config = yaml.load(open(args.config_yml, 'r'))

train_dataset = bAbiImpDataset(
    args.dialog_json_train,
    args.vocab_json,
    max_sequence_length=config["model"]["max_seq_len"],
    num_examples=args.batch_size if args.overfit else None,
    return_explicit=True,
    return_options=args.mode == 'cancel',
    return_qa=args.mode == 'qa',
    concat_history=config.get("concat_history", True),
    add_boundary_toks= config["model"].get("decoder") == "gen" # True if generate
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.cpu_workers,
    shuffle=True,
)

val_dataset = bAbiImpDataset(
    args.dialog_json_dev,
    args.vocab_json,
    max_sequence_length=config["model"]["max_seq_len"],
    num_examples=args.batch_size if args.overfit else None,
    return_explicit=True,
    return_options=args.mode == 'cancel',
    return_qa=args.mode == 'qa',
    concat_history=config.get("concat_history", True),
    add_boundary_toks= config["model"].get("decoder") == "gen" # True if generate
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

args.save_dirpath = os.path.join(args.save_dirpath, config["name"]) if not args.overfit else os.path.join(args.save_dirpath, 'overfit')


if args.mode == "qa":
    model = Encoder(config["model"], train_dataset.vocabulary).to(device)
else:
    encoder = Encoder(config["model"], train_dataset.vocabulary)
    decoder = Decoder(config["model"], train_dataset.vocabulary)
    decoder.word_embed = encoder.word_embed
    if config["model"]["model"] == "enc2dec":
        model = EncoderDecoderModel(encoder, decoder).to(device)
    else:
        model = RoundWiseEncoderDecoderModel(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

if "inference_encoder" in config["model"]:
    new_config = config["model"].copy()
    new_config["encoder"] = config["model"]["inference_encoder"]
    new_config["decoder"] = config["model"]["inference_decoder"]
    inference_encoder = Encoder(new_config, train_dataset.vocabulary)
    inference_deocder = Decoder(new_config, train_dataset.vocabulary)
    inference_model = EncoderDecoderModel(inference_encoder, inference_deocder)
    inference_componet = torch.load(config["model"]["inference_path"])
    print('Loaded inference model {}'.format(config["model"]["inference_path"]))
    inference_model.load_state_dict(inference_componet["model"])
    model.inference_encoder = inference_encoder

if args.resume:
    load_pthpath = os.path.join(args.save_dirpath, args.ckpt)
    components = torch.load(load_pthpath)
    model.load_state_dict(components['model'])
    optimizer.load_state_dict(components['optimizer'])
    print('Loaded checkpoint {}'.format(load_pthpath))

# checkpoint_manager = CheckpointManager(
#     model, optimizer, args.save_dirpath, config=config
# )
sparse_metrics = SparseGTMetrics()

if args.mode == "qa" or config["model"].get("decoder") == "gen":
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocabulary.PAD_INDEX)
else:
    criterion = nn.CrossEntropyLoss()

if args.validate:
    val_json = []

best_score = 0

if not os.path.exists(args.save_dirpath):
    os.makedirs(args.save_dirpath)

yaml.dump(
    config,
    open(os.path.join(args.save_dirpath,"config.yml"), "w"),
    default_flow_style=False,
)

for epoch in range(args.num_epochs):
    model.train()
    print("\nTraining for epoch {}:".format(epoch))
    if args.mode == "qa":
        correct = 0
        count = 0

    for i, batch in enumerate(tqdm(train_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        output = model(batch)
        # (batch_size, num_rounds)
        if args.mode == "qa":
            target = batch["qa_ans"]
        else:
            if config["model"].get("decoder") == 'gen':
                target = batch["gt_ans_out"]
            else:
                target = batch["ans_ind"]
        batch_loss = criterion(
            output.view(-1, output.size(-1)), target.view(-1)
        )
        batch_loss.backward(retain_graph=True)
        optimizer.step()
        if args.mode == "qa":
            pred = output.view(-1, output.size(-1))
            pred_idx = pred.max(1)[1]
            gt_ans = batch["qa_ans"].view(-1)
            correct += torch.sum(pred_idx == gt_ans).item()
            count += int(pred.size(0))
    if args.mode  == 'qa':
        print('Training Acc: {}/{} ({:.2f}%)'.format(correct, count, correct/count * 100))

    
    torch.cuda.empty_cache()
    # checkpoint_manager.step()
    if epoch % args.log_step == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.save_dirpath, "checkpoint_latest.pth")
        )
    if args.validate:
        model.eval()
        print("\nValidation for epoch {}:".format(epoch))
        if args.mode == "qa":
            correct = 0
            count = 0
        for i, batch in enumerate(tqdm(val_dataloader)):
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
                sparse_metrics.observe(output, batch["ans_ind"])
        all_metrics = {}
        if args.mode == "qa":
            all_metrics.update({"acc": float(correct)/count*100, "correct": correct, "total": count})
        else:
            all_metrics.update(sparse_metrics.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            print("{}: {}".format(metric_name, metric_value))
        all_metrics.update({"epoch": epoch})
        val_json.append(all_metrics)
        torch.cuda.empty_cache()
        with open(os.path.join(args.save_dirpath, 'val_res.json'), 'w') as f:
            for line in val_json:
                f.write(json.dumps(line, sort_keys=True) + '\n')
        if args.mode == "qa":
            target_score = all_metrics["acc"]
        else:
            target_score = all_metrics["mrr"]
        if target_score > best_score:
            best_score = target_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.save_dirpath, "checkpoint_best.pth")
            )
        print("Best score: {:.2f}".format(best_score))