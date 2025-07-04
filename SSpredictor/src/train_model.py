import random
import numpy as np
import argparse
import torch
import logging
import os
import pandas as pd

from model import SecondaryStructurePredictor
from dataset import create_dataloader
from utils import get_embed_dim

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Applied workaround for CuDNN issue, install nvrtc.so
# Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR

parser = argparse.ArgumentParser()

parser.add_argument("--emb", type=str, help="The name of the desired LLM-dataset combination.")
parser.add_argument("--train_partition_path", type=str, help="The path of the train partition.")
parser.add_argument("--val_partition_path", type=str, help="The path of the validation partition.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--max_epochs", default=15, type=int, help="Maximum number of training epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for the training.")
parser.add_argument("--out_path", type=str, help="Path to write predictions (base pairs of test partition), weights and logs")
parser.add_argument("--seed", type=int, help="Random seed for this training.")
parser.add_argument("--iter", default=5, type=int, help="iterate")

args = parser.parse_args()

if torch.cuda.is_available():
    device=f"cuda:{torch.cuda.current_device()}"
else:
    device='cpu'

os.makedirs(args.out_path, exist_ok=True)

embeddings_path = f"data/embeddings/{args.emb}.h5"

# シードの固定
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    # pytorch CPU,GPU両方に対してシード固定できる
    torch.backends.cudnn.benchmark = False  # 再現性を無視してでも畳み込み演算速度を上げるオプション
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True   # pytorchで非決定的な操作を決定的なものにするオプション

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(args.out_path, f'log.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

train_loader = create_dataloader(
    embeddings_path,
    args.train_partition_path,
    args.batch_size,
    True,
    args.seed
)

#長さのみの情報を持つ配列embeddingのdataloader
ref_loader = create_dataloader(
    embeddings_path,
    args.train_partition_path,
    args.batch_size,
    True,
    args.seed,
    reference=True
)

if args.val_partition_path:
    val_loader = create_dataloader(
        embeddings_path,
        args.val_partition_path,
        args.batch_size,
        False,
        args.seed
    )

embed_dim = get_embed_dim(train_loader)

logger.info(f"Run on {args.out_path}, with device {device} and embeddings {embeddings_path}")
logger.info(f"Training with file: {args.train_partition_path}")
if args.val_partition_path:
    logger.info(f"Validation enabled, using file: {args.val_partition_path}")

for i in range(args.iter):
    net = SecondaryStructurePredictor(embed_dim=embed_dim, device=device, lr=args.lr)
    net_ref = SecondaryStructurePredictor(embed_dim=embed_dim, device=device, lr=args.lr)
    metrics_for_epoch = []
    
    for epoch in range(args.max_epochs):
        logger.info(f"Starting iteration {i+1}, epoch {epoch+1}")
        metrics_train = net.fit(train_loader)
        metrics = {"train_loss": metrics_train["loss"]}
        if epoch < 3:
            metrics_ref = net_ref.fit(ref_loader)
            metrics["ref_loss"] = metrics_ref["loss"]

        if args.val_partition_path:
            logger.info("Running validation inference")
            val_metrics = net.test(val_loader)
        
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            metrics.update(val_metrics)

        metrics_for_epoch.append(metrics)
        logger.info(" ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]))    

    pd.set_option('display.float_format','{:.3f}'.format)
    pd.DataFrame(metrics_for_epoch).to_csv(os.path.join(args.out_path, f"metrics_{i+1}.csv"), index=False)

    torch.save(
        net.state_dict(),
        os.path.join(args.out_path, f"weights/prior_{i+1}.pth")
    )
    torch.save(
        net_ref.state_dict(),
        os.path.join(args.out_path, f"weights/reference_{i+1}.pth")
    )
