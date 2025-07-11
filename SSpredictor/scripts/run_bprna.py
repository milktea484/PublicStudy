import pandas as pd 
import os 
import shutil 
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--emb", type=str, help="The name of the desired LLM-dataset combination.", required=True)

args = parser.parse_args()

llm_and_dataset = args.emb.split("_")
llm = llm_and_dataset[0]
dataset = llm_and_dataset[1]
current_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

df = pd.read_csv(f'data/bpRNA.csv', index_col="id")
splits = pd.read_csv(f"data/bpRNA_splits.csv", index_col="id")

max_epochs = 5
seed = 1
iteration = 1

train = pd.concat((df.loc[splits.partition=="TR0"], df.loc[splits.partition=="VL0"])) 
test = df.loc[splits.partition=="TS0"]
data_path = f"data/bprna/"
out_path = f"results/{current_timestamp}/{dataset}/{llm}"
weight_path = f"results/{current_timestamp}/{dataset}/{llm}/weights"
os.makedirs(data_path, exist_ok=True)
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(weight_path, exist_ok=True)
train.to_csv(f"{data_path}train.csv")
test.to_csv(f"{data_path}test.csv")

print("+" * 80)
print(f"bpRNA TRAINING STARTED".center(80))
print("+" * 80)
os.system(f"python src/train_model.py --emb {args.emb} --train_partition_path {data_path}train.csv --max_epochs {max_epochs} --out_path {out_path} --seed {seed} --iter {iteration}")
# os.system(f"python src/train_model.py --emb {args.emb} --train_partition_path {data_path}train.csv --out_path {out_path}")
print(f"bpRNA TRAINING ENDED".center(80))
print("+" * 80)
print(f"bpRNA TESTING STARTED".center(80))
print("+" * 80)
os.system(f"python src/test_model.py --emb {args.emb} --test_partition_path {data_path}test.csv --out_path {out_path} --seed {seed} --iter {iteration}")
# os.system(f"python src/test_model.py --emb {args.emb} --test_partition_path {data_path}test.csv --out_path {out_path}")
print(f"bpRNA TESTING ENDED".center(80))
