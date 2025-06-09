import torch
from torch.utils.data import Dataset
import numpy as np
import random
import h5py
import pickle
import os

from utils import from_fastafile, from_csvfile, seq2token, masking
from contents import appended_tokens

class PretrainingDataset(Dataset):
    def __init__(self, dataset_path): 

        # データセットの読み込み
        if dataset_path.endswith(".h5"):
            seq_ids = []
            token_seqs = []
            with h5py.File(dataset_path, "r") as infile:
                # hdf5ファイルの階層構造は，一つのグループに全配列のdataset(name=(配列id), data=(numpy形式のtoken))があることを想定
                for id, token in infile.items():
                    seq_ids.append(id)
                    token_seqs.append(torch.from_numpy(token[()]))
            self.ids = seq_ids
            self.token_seqs = token_seqs
        elif dataset_path.endswith(".pickle"):
            seq_ids = []
            token_seqs = []
            with open(dataset_path, "rb") as infile:
                token_dict = pickle.load(infile)
                for id, token in token_dict.items():
                    seq_ids.append(id)
                    token_seqs.append(torch.from_numpy(token))
            self.ids = seq_ids
            self.token_seqs = token_seqs
            
        # 以下はトークンに変換していないファイルを使用した場合
        elif dataset_path.endswith(".fasta") or dataset_path.endswith(".csv"):
            # csvファイルはcolumsに"id"と"sequence"があることを想定
            if dataset_path.endswith(".fasta"):
                seq_ids, sequences = from_fastafile(dataset_path)
            else:
                seq_ids, sequences = from_csvfile(dataset_path)
            self.ids = seq_ids
            self.token_seqs = seq2token(sequences)
        else:
            raise ValueError("Unrecognized file format")

    
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # マスクされたトークン配列の作成
        masked_token_seq, mask_idx = masking(self.token_seqs[idx])
        return {"seq_id": self.ids[idx], "token_seq": self.token_seqs[idx], "masked_token_seq": masked_token_seq, "mask_idx": mask_idx, "L": len(self.token_seqs[idx])} 

def pad_batch(batch):
    """batch is a list of dicts with keys: seqid, seq_emb, Mc, L, sequence, mask"""
    seq_ids, token_seqs, masked_token_seqs, mask_idxes, Ls  = [[batch_elem[key] for batch_elem in batch] for key in batch[0].keys()]
    batch_size = len(batch)
    max_L = max(Ls)
    token_seqs_pad = torch.full((batch_size, max_L), appended_tokens.index("<pad>"), dtype=torch.uint8)
    masked_token_seqs_pad = torch.full((batch_size, max_L), appended_tokens.index("<pad>"), dtype=torch.uint8)
    for k in range(batch_size):
        token_seqs_pad[k, : Ls[k]] = token_seqs[k][:Ls[k]]
        masked_token_seqs_pad[k, : Ls[k]] = masked_token_seqs[k][:Ls[k]]

    return {"seq_ids": seq_ids, "token_seqs": token_seqs_pad, "masked_token_seqs": masked_token_seqs_pad, "mask_idxes": mask_idxes, "Ls": Ls}


def create_dataloader(partition_path, batch_size, shuffle, seed=None, collate_fn=pad_batch):
    dataset = PretrainingDataset(
        dataset_path=partition_path,
    )

    if seed is not None:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=collate_fn,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=collate_fn,
        )
    

def serch_file_path(partition_path, split):
    assert split == "train" or split == "val"
    dir_path = os.path.join(os.path.dirname(partition_path), split)
    file_list = [ f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    partition_path_list = []
    for num in range(len(file_list)):
        input_file = partition_path.replace(".pickle", f"_{num}.pickle")
        # 0から連番になっているファイルのみを受け付ける（別にこの条件はなくてもいいかも）
        if input_file not in file_list:
            # 指定された名称のファイルの0番がなければFileNotFoundErrorを返す
            if num == 0:
                raise FileNotFoundError(f"File {dir_path}/{partition_path} does not exist.")
            break
        partition_path_list.append(f"{dir_path}/{input_file}")
    return partition_path_list


def create_batch_iter(partition_path, split, batch_size, shuffle, seed=None):
    partition_path_list = serch_file_path(partition_path, split)
    while True:
        for input_path in partition_path_list:
            loader = create_dataloader(
                partition_path=input_path,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed
            )
            for batch in loader:
                batch["token_seqs"] = batch["token_seqs"].to(torch.long)
                batch["masked_token_seqs"] = batch["masked_token_seqs"].to(torch.long)
                yield batch
