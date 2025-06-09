import random
import numpy as np
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

from model import SecondaryStructurePredictor
from dataset import create_dataloader
from utils import get_embed_dim

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Applied workaround for CuDNN issue, install nvrtc.so
# Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR


parser = argparse.ArgumentParser()
parser.add_argument("--emb", type=str, help="The name of the desired LLM-dataset combination.")
parser.add_argument("--test_partition_path", type=str, help="The path of the test partition.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--out_path", default="results", type=str, help="Path to read model from, and to write predictions/metrics/logs")
parser.add_argument("--weights_path", type=str, help="Path to read model from, in cases it has to be read from a different place than `out_path`")
parser.add_argument("--seed", type=int, help="Random seed for this training.")
parser.add_argument("--iter", default=5, type=int, help="iterate")

args = parser.parse_args()

if torch.cuda.is_available():
    device=f"cuda:{torch.cuda.current_device()}"
else:
    device='cpu'

# Create results file with the name of input file
out_name = os.path.splitext(os.path.split(args.test_partition_path)[-1])[0]

embeddings_path = f"data/embeddings/{args.emb}.h5"

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
        logging.FileHandler(os.path.join(args.out_path, f'log-{out_name}.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Run on {args.out_path}, with device {device} and embeddings {embeddings_path}")
logger.info(f"Testing with file: {args.test_partition_path}")

test_loader = create_dataloader(
    embeddings_path,
    args.test_partition_path,
    args.batch_size,
    False,
    args.seed
)
embed_dim = get_embed_dim(test_loader)

y_probs, y_refs = [], []
test_loss = 0.0
for i in range(args.iter):
    try:
        prior_path = os.path.join(args.weights_path, f"prior_{i+1}.pth") if args.weights_path else os.path.join(args.out_path, f"weights/prior_{i+1}.pth")
        ref_path = os.path.join(args.weights_path, f"reference_{i+1}.pth") if args.weights_path else os.path.join(args.out_path, f"weights/reference_{i+1}.pth")
    except FileNotFoundError("this file does not exist: " + os.path.join(args.weights_path, f"prior_{i+1}.pth")):
        break
    best_model = SecondaryStructurePredictor(embed_dim=embed_dim, device=device)
    best_model.load_state_dict(torch.load(prior_path, map_location=torch.device(best_model.device)))
    outputs = best_model.test(test_loader)
    y_probs.append(outputs.pop("y_prob"))
    test_loss += outputs.pop("loss")

    ref_model = SecondaryStructurePredictor(embed_dim=embed_dim, device=device)
    ref_model.load_state_dict(torch.load(ref_path, map_location=torch.device(ref_model.device)))
    ref_outputs = ref_model.test(test_loader)
    y_refs.append(ref_outputs.pop("y_prob"))

#このあたりのコードが汚いので直したいところではある
y_prob, y_ref = {}, {}
seq_ids = dict.fromkeys(y_probs[0])
for seq_id in seq_ids.keys():
    y_prob_list = [y_probs[i].pop(seq_id).unsqueeze(0) for i in range(args.iter)]
    y_ref_list = [y_refs[i].pop(seq_id).unsqueeze(0) for i in range(args.iter)]
    y_prob[seq_id] = torch.mean(torch.cat(y_prob_list, dim=0), dim=0)
    y_ref[seq_id] = torch.mean(torch.cat(y_ref_list, dim=0), dim=0)
outputs["loss"] = test_loss/float(args.iter)
predictions, outputs["f1"], y_prob, y = best_model.pred(test_loader, y_prob, y_ref)

metrics = {f"test_{k}": v for k, v in outputs.items()}
logger.info(" ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]))
out_file = os.path.join(args.out_path, f"metrics_{out_name}.csv")
pd.set_option('display.float_format','{:.3f}'.format)
pd.DataFrame([metrics]).to_csv(out_file, index=False)

out_file = os.path.join(args.out_path, f"preds_{out_name}.csv")
predictions.to_csv(out_file, index=False)

##可視化結果プロット
for id, y_contact in y_prob.items():
	y_contact = y_contact.to('cpu').detach().numpy().copy()
	cordinate_size = y_contact.shape[0]
	cordinates = np.arange(0, cordinate_size+1)
	fig=plt.figure()
	#Python(numpy)では，配列のインデックスは行列の成分の番号(添え字)に対応するため，格子のインデックスに揃えるためには転置が必要
	plt.pcolormesh(cordinates, cordinates, y_contact.T)
	plt.colorbar()
	out_file = os.path.join(args.out_path, f"pred_{id}.png")
	plt.savefig(out_file)
    
	y[id] = y[id].to('cpu').detach().numpy().copy()
	fig=plt.figure()
	#Python(numpy)では，配列のインデックスは行列の成分の番号(添え字)に対応するため，格子のインデックスに揃えるためには転置が必要
	plt.pcolormesh(cordinates, cordinates, y[id].T)
	plt.colorbar()
	out_file = os.path.join(args.out_path, f"gt_{id}.png")
	plt.savefig(out_file)

	break