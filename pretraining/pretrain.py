import os
import wandb
import random
import argparse
import logging
import datetime
import h5py
import numpy as np
import torch

from dataset import create_dataloader
from model import MLMModel, data2vecModel
from contents import ModelConfig, data2vecConfig


parser = argparse.ArgumentParser()

parser.add_argument("--framework", type=str, help="MLM or data2vec")
parser.add_argument("--test_partition_path", type=str, help="the path of test file")
parser.add_argument("--weight_path", type=str, help="the path of weight.pth")
parser.add_argument("--seed", default=None, type=int, help="setting of seed")

args = parser.parse_args()


# 使用するデバイスの設定
if torch.cuda.is_available():
    device=f"cuda:{torch.cuda.current_device()}"
else:
    device='cpu'
device_type = "cuda" if "cuda" in device else "cpu"
ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)

# 出力ディレクトリ
out_path = os.path.dirname(args.weight_path)

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
        logging.FileHandler(os.path.join(out_path, f'log_pretrain.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)


# モデルの設定
if args.framework == "data2vec":
    logger.info("Selected framework: data2vec")
    config = data2vecConfig()
    model = data2vecModel(config, device=device)
    # loadにweights_pathを指定しているが，ゆくゆくはモデルフレームワークに応じて自動で選択できるようにしたい
    if args.weight_path:
        model.load_state_dict(torch.load(args.weight_path, map_location=model.device))
else:
    logger.info("Selected framework: MLM")
    config = ModelConfig()
    model = MLMModel(config, device=device)
    if args.weight_path:
        model.load_state_dict(torch.load(args.weight_path, map_location=model.device))

# # wandbの設定
# wandb.init(
#       project = "pretraining_test",

#       config = {
#             "framework": args.framework,
#             "test_partition_path": args.test_partition_path,
# 			"weights_path": args.weights_path,
#             "batch_size": config.batch_size,
#             "vocab_size": config.vocab_size,
#             "embed_dim": config.embed_dim,
#             "ffn_embed_dim": config.ffn_embed_dim,
#             "n_layer": config.n_layer,
#             "n_head": config.n_head,
#             "seed": args.seed
#       }
# )

# データローダーの設定
def get_batch_iter(loader):
    for batch in loader:
        batch["token_seqs"] = batch["token_seqs"].to(torch.long)
        batch["masked_token_seqs"] = batch["masked_token_seqs"].to(torch.long)
        yield batch

logger.info(f"Input for testing: {args.test_partition_path}")
test_loader = create_dataloader(
    args.test_partition_path,
    config.batch_size,
    True,
    args.seed
)
test_batch_iter = get_batch_iter(test_loader)

# embeddingの計算
logger.info(f"Run on {out_path}, with device {device}")
logger.info(f"Testing with file: {args.test_partition_path}")

emb_dict = {}
model.eval()
for batch in test_batch_iter:
    with ctx, torch.inference_mode():
        result = model.calculate_repr(batch)
    assert "representation" in result.keys() and "repr_var" in result.keys()
    # wandb.log({"test_loss": test_loss.item()})
    for idx in range(len(batch["seq_ids"])):
        seq_id = batch["seq_ids"][idx]
        seq_L = batch["Ls"][idx]
        emb_dict[seq_id] = result["representation"][idx][:seq_L].to('cpu').detach().numpy().copy()


# embeddingの保存
out_file = os.path.join(out_path, "embedding.h5")
with h5py.File(out_file, "w") as hdf:
    for key, value in emb_dict.items():
        hdf.create_dataset(name=key, data=value)

# wandb.finish()
