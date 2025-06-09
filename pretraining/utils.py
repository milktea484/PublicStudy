import torch
import pandas as pd
from contents import rnaseq_tokens, appended_tokens


def masking(token_seq, sptoken_prob=0.15, mask_prob=0.8):
    '''
    入力をマスクする関数．デフォルトなら以下の通り．
    入力配列のうち15%をマスク対象(special token)，そのうち80%を"<mask>"トークン，10%をランダムな他トークン，残り10%はそのまま
    '''
    masked_token_seq = token_seq.clone().detach()
    idx_list = torch.randperm(len(masked_token_seq))
    sptoken_idxes = idx_list[:int(len(masked_token_seq)*sptoken_prob)].tolist()

    probs = torch.rand(len(sptoken_idxes)).tolist()
    for idx in sptoken_idxes:
        prob = probs.pop(0)
        if prob < mask_prob:
            masked_token_seq[idx] = appended_tokens.index("<mask>")
        elif prob > 0.5 + mask_prob/2.0:
            # other_idxesを5種のトークンを持つようランダムな順番で作成，もともとのと被るトークンは削除して先頭要素を選択
            other_idxes = torch.randperm(len(rnaseq_tokens), dtype=torch.uint8).tolist()
            other_idxes.remove(masked_token_seq[idx].item())
            masked_token_seq[idx] = other_idxes[0]

    assert len(token_seq) == len(masked_token_seq)
    return masked_token_seq, sptoken_idxes


def seq2token(seqs, pad_token="-"):
    '''
    塩基配列からトークンに直す関数．
    トークンは["A", "C", "G", "U", "N"]の対応するインデックス
    '''

    seqs_tokens = []
    for seq in seqs:
        seq_tokens = []

        for nt in seq:
            if nt in rnaseq_tokens:
                seq_tokens.append(rnaseq_tokens.index(nt))
            elif nt == "X" or nt == "I" or nt == "F":
                #print(f"replace nucleotide {nt} -> N")
                seq_tokens.append(rnaseq_tokens.index("N"))
            else:
                raise ValueError(f"Unrecognized nucleotide {nt}")
            
        seqs_tokens.append(torch.tensor(seq_tokens, dtype=torch.uint8))
    return seqs_tokens


def from_fastafile(fasta_file):
        '''fastaファイルからidと塩基配列を取り出す'''
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

        with open(fasta_file, "r") as infile:
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
        
        return seq_ids, sequences


def from_csvfile(csv_file):
    '''csvファイルからidと塩基配列を取り出す'''
    data = pd.read_csv(csv_file)
    sequences = data.sequence.tolist()
    seq_ids = data.id.tolist()
    
    return seq_ids, sequences