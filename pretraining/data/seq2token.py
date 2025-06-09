import argparse
import numpy as np
import h5py
import pickle
import pandas as pd
import os
from tqdm import tqdm


VOCABLARY = ["A", "C", "G", "U", "N"]	# 事前学習で使用する塩基
NT = ["R","Y","K","M","S","W","B","D","H","V"]	# 事前学習で使用しない塩基
OTHER_NT = ["X","I","F","T"]	# IUPACに登録されていないが入力に存在している塩基

FILE_SPLIT = 2097152	# 保存する際の1つのファイルに保存する最大配列数

# fastaまたはcsvファイルから配列を読み込み，h5またはpickleに保存する

def main():
    parse = argparse.ArgumentParser()

	# まるでファイルフォーマットの選び方が4種類あるように見えるが，訓練データはfasta->pickle，テストデータはcsv->h5であることを前提とした実装になっている
    parse.add_argument("--input_file_path", type=str, help="Input file is expected to be .fasta or .csv.")
    parse.add_argument("--output_file_path", type=str, help="Output file is expected to be .pickle or .h5")

    args = parse.parse_args()


    if args.input_file_path.endswith(".fasta"):
        inputformat = "fasta"
    elif args.input_file_path.endswith(".csv"):
        inputformat = "csv"
    else:
        raise ValueError("Unrecognized file format")
    
    print(f"Loading {args.input_file_path} ...")
    seq_ids, seqs = from_file(args.input_file_path, inputformat)
    

    if args.output_file_path.endswith(".h5"):
        # 入力配列数が50万以下であればこっちがいい
        outputformat = "h5"
    elif args.output_file_path.endswith(".pickle"):
        # 入力配列数が50万を大きく超える場合，hdf5がうまく保存できない
        # cpuメモリの節約のため分割しなければならず，hdf5の階層構造の強みが活かせないのでpickleで保存
        outputformat = "pickle"
    else:
        raise ValueError("Unrecognized file format")

    print("Converting sequences into tokens...")
    seqs_tokens_dict = gen_token(seq_ids, seqs)

    print(f"Saving tokens into {args.output_file_path} file...")
    save_token(args.output_file_path, seqs_tokens_dict, outputformat)

    print("Finished")


def from_file(file, fileformat):
    '''fastaファイルまたはcsvファイルからidと塩基配列を取り出す'''

    if fileformat == "fasta":
        seq_ids, sequences = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            seq_ids.append(cur_seq_label.replace(" ", "_"))
            sequences.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.upper().strip())

        _flush_current_seq()

        assert len(set(seq_ids)) == len(sequences)

    elif fileformat == "csv":
        data = pd.read_csv(file)
        seq_ids = data.id.tolist()
        sequence_list = data.sequence.tolist()
        sequences = [seq.upper() for seq in sequence_list]

    return seq_ids, sequences


def gen_token(seq_ids, seqs):

    seqs_tokens_dict = {}
    for seq_id, seq in zip(seq_ids, seqs):
        seq_tokens = []
        for nt in seq:
            if nt in VOCABLARY:
                seq_tokens.append(VOCABLARY.index(nt))
            elif nt in NT or nt in OTHER_NT:
                #print(f"replace nucleotide {nt} -> N")
                seq_tokens.append(VOCABLARY.index("N"))
            else:
                raise ValueError(f"Unrecognized nucleotide {nt}")
            
        seqs_tokens_dict[seq_id] = np.array(seq_tokens, dtype=np.uint8)
    return seqs_tokens_dict


def save_token(file, seqs_tokens_dict, fileformat):
    if fileformat == "h5":
        out_path = "test"
        os.makedirs(out_path, exist_ok=True)
        out_file_path = f"{out_path}/{file}"
        with h5py.File(out_file_path, "w") as hdf:
            for key, value in tqdm(seqs_tokens_dict.items()):
                hdf.create_dataset(name=key, data=value)

    elif fileformat == "pickle":
        if file.startswith("train"):
            out_path = "pickle/train"
        elif file.startswith("val"):
            out_path = "pickle/val"
        else:
            out_path = "pickle/others"
        os.makedirs(out_path, exist_ok=True)

        mini_dict = {}
        num = 0
        file_num = 0
        for key, value in tqdm(seqs_tokens_dict.items()):
            num += 1
            mini_dict[key] = value
            if num % FILE_SPLIT == 0:
                file_name = file.replace(".pickle", f"_{file_num}.pickle")
                out_file_path = f"{out_path}/{file_name}"
                with open(out_file_path, "wb") as f:
                    pickle.dump(mini_dict, f)
                file_num += 1
                mini_dict = {}

        if mini_dict != {}:
            file_name = file.replace(".pickle", f"_{file_num}.pickle")
            out_file_path = f"{out_path}/{file_name}"
            with open(out_file_path, "wb") as f:
                pickle.dump(mini_dict, f)

if __name__ == "__main__":
    main()