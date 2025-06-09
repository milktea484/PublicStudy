import os
import wandb
import random
import argparse
import logging
import datetime
from tqdm import tqdm
import numpy as np
import torch

from dataset import create_batch_iter
from model import MLMModel, data2vecModel
from contents import ModelConfig, data2vecConfig


parser = argparse.ArgumentParser()

parser.add_argument("--framework", type=str, help="MLM or data2vec")
parser.add_argument("--train_partition_path", type=str, help="the path of train input pickle file")
parser.add_argument("--val_partition_path", type=str, help="the path of validation input pickle file")
parser.add_argument("--seed", default=None, type=int, help="setting of seed")

args = parser.parse_args()


# 使用するデバイスの設定
if torch.cuda.is_available():
    device=f"cuda:{torch.cuda.current_device()}"
else:
    device='cpu'
device_type = "cuda" if "cuda" in device else "cpu"
ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)

# 使用するフレームワークの識別
if args.framework == "data2vec":
    framework = "data2vec"
else:
    framework = "MLM"

# 出力ディレクトリの作成（上書き）
current_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
out_path = f"../data/pretrain_results/{framework}/{current_timestamp}"
os.makedirs(out_path, exist_ok=True)

# seed固定
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# logの設定
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(out_path, f'log_train.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

# モデルの設定
if framework == "data2vec":
    config = data2vecConfig()
    model = data2vecModel(config, device=device)
    wandb.init(
        project = "pretraining_data2vec",

        config = {
                "framework": args.framework,
                "train_partition_path": args.train_partition_path,
                "val_partition_path": args.val_partition_path,
                "max_iter": config.max_iter,
                "warmup_iter": config.warmup_iter,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "vocab_size": config.vocab_size,
                "max_length": config.max_length,
                "embed_dim": config.embed_dim,
                "ffn_embed_dim": config.ffn_embed_dim,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "learning_rate": config.learning_rate,
                "min_lr": config.min_lr,
                "eval_interval": config.eval_interval,
                "eval_iter": config.eval_iter,
                "k_layer": config.k_layer,
                "ema_decay": config.ema_decay,
                "ema_end_decay": config.ema_end_decay,
                "ema_anneal_end_step": config.ema_anneal_end_step,
                "loss_beta": config.loss_beta,
                "n_head_layer": config.n_head_layer,
                "seed": args.seed
        }
    )
elif framework == "MLM":
    config = ModelConfig()
    model = MLMModel(config, device=device)
    wandb.init(
        project = "pretraining",

        config = {
                "framework": args.framework,
                "train_partition_path": args.train_partition_path,
                "val_partition_path": args.val_partition_path,
                "max_iter": config.max_iter,
                "warmup_iter": config.warmup_iter,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "vocab_size": config.vocab_size,
                "max_length": config.max_length,
                "embed_dim": config.embed_dim,
                "ffn_embed_dim": config.ffn_embed_dim,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "learning_rate": config.learning_rate,
                "min_lr": config.min_lr,
                "eval_interval": config.eval_interval,
                "eval_iter": config.eval_iter,
                "seed": args.seed
        }
    )
else:
    ValueError(f"This framework is not recognized: {framework}")


# データローダーの設定
train_batch_iter = create_batch_iter(
    partition_path=args.train_partition_path,
    split="train",
    batch_size=config.batch_size // config.gradient_accumulation_steps,
    shuffle=True,
    seed=args.seed
)

if args.val_partition_path:
    eval_batch_iter = {
        "train": create_batch_iter(
            partition_path=args.train_partition_path,
            split="train",
            batch_size=config.batch_size // config.gradient_accumulation_steps,
            shuffle=True,
            seed=args.seed
        ),
        "val": create_batch_iter(
            partition_path=args.val_partition_path,
            split="val",
            batch_size=config.batch_size // config.gradient_accumulation_steps,
            shuffle=True,
            seed=args.seed
        )
    }


# トレーニングと結果の出力
logger.info(f"Run on {out_path}, with device {device}")
logger.info(f"Input for training: {args.train_partition_path}")
if args.val_partition_path:
    logger.info(f"Input for validation: {args.val_partition_path}")
logger.info(f"Selected framework: {framework}")
if args.seed is not None:
    logger.info(f"Setting seed: {args.seed}")

model.train()
step = 0
accumulated_loss = 0.0
with tqdm(
        range(config.max_iter * config.gradient_accumulation_steps)
    ) as pbar_iter:
    for it in pbar_iter:
        train_batch = next(train_batch_iter)
        with ctx:
            result = model._train(train_batch)
        loss = result["loss"] / config.gradient_accumulation_steps
        loss.backword()
        accumulated_loss += loss.item()

        if it+1 % config.gradient_accumulation_steps == 0:
            if args.val_partition_path:
                if step % config.eval_interval == 0:
                    model.eval()
                    loss_dict = {}
                    var_dict = {}
                    for split, batch_iter in eval_batch_iter.items():
                        losses = torch.zeros(config.eval_iter)
                        x_vars = torch.zeros(config.eval_iter)
                        y_vars = torch.zeros(config.eval_iter)

                        # gradient_accumulation_stepsに依らず一定回数回すので大丈夫だよね？
                        for eval_it in range(config.eval_iter):
                            batch = next(batch_iter)
                            with ctx, torch.inference_mode():
                                result = model._train(batch)
                            assert "loss" in result.keys() and "repr_vars" in result.keys()
                            losses[eval_it] = result["loss"].item()
                            if framework == "data2vec":
                                x_var, y_var = result["repr_vars"]
                                x_vars[eval_it] = x_var.item()
                                y_vars[eval_it] = y_var.item()
                            else:
                                x_var = result["repr_vars"]
                                x_vars[eval_it] = x_var.item()
                        loss_dict[split] = losses.mean()
                        var_dict[split] = (x_vars.mean(), y_vars.mean())
                    model.train()

            model._step()
            pbar_iter.set_postfix_str(f"Loss: {accumulated_loss:.3f}")

            if args.val_partition_path:
                if step % config.eval_interval == 0:
                    if framework == "data2vec":
                        train_x_var, train_y_var = var_dict["train"]
                        val_x_var, val_y_var = var_dict["val"]
                        wandb.log({
                            "train_loss": loss_dict["train"].item(), 
                            "val_loss": loss_dict["val"].item(), 
                            "train_student_var": train_x_var.item(), 
                            "train_teacher_var": train_y_var.item(), 
                            "val_student_var": val_x_var.item(), 
                            "val_teacher_var": val_y_var.item()}, step=step)
                    else:
                        train_x_var, _ = var_dict["train"]
                        val_x_var, _ = var_dict["val"]
                        wandb.log({
                            "train_loss": loss_dict["train"].item(), 
                            "val_loss": loss_dict["val"].item(), 
                            "train_representation_var": train_x_var.item(), 
                            "val_representation_var": val_x_var.item()}, step=step)
            else:
                wandb.log({"train_loss": accumulated_loss.item()})
            
            step += 1
            accumulated_loss = 0.0

wandb.finish()

# モデルの保存．ゆくゆくはモデルフレームワークに応じた場所に保存したい
torch.save(
    model.state_dict(),
    os.path.join(out_path, f"weight.pth")
)

