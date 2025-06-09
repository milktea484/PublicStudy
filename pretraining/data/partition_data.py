# total sequences: 19675767
# train sequences: 15739910
# validation sequences: 3935857

import random

def partition_fastafile(fasta_file, train_file, val_file, threshold=0.8):
    cur_seq_label = None
    buf = ""

    total_seq = 0
    n_train_seq = 0
    n_val_seq = 0
    with open(train_file, "w") as toutfile:
        with open(val_file, "w") as voutfile:
            with open(fasta_file, "r") as infile:
                for line_idx, line in enumerate(infile):
                    if line.startswith(">"):  # label line
                        total_seq += 1
                        if total_seq % 100000 == 1:
                            print("No." + str(total_seq) + "~ sequences is beening partitioned now ...")
                        if cur_seq_label is not None:
                            if random.random() < threshold:
                                toutfile.write(cur_seq_label + buf)
                                n_train_seq += 1
                            else:
                                voutfile.write(cur_seq_label + buf)
                                n_val_seq += 1
                            cur_seq_label = None
                            buf = ""
                        
                        if len(line) > 0:
                            cur_seq_label = line
                        else:
                            cur_seq_label = "seqnum{}".format(line_idx)
                        
                    else:  # sequence line
                        buf += line

            if cur_seq_label is not None:
                if random.random() < threshold:
                    toutfile.write(cur_seq_label + buf)
                    n_train_seq += 1
                else:
                    voutfile.write(cur_seq_label + buf)
                    n_val_seq += 1
                cur_seq_label = None
                buf = ""

    assert total_seq == n_train_seq + n_val_seq
    
    return total_seq, n_train_seq, n_val_seq

def main():
    random.seed(1)
    fasta_file = "/share03/nakamura/RNAcentral/ver24/rnacentral100_512_pretreated.fasta"
    train_file = "/share03/nakamura/github/pretraining/data/train_partition.fasta"
    val_file = "/share03/nakamura/github/pretraining/data/val_partition.fasta"
    total_seq, n_train_seq, n_val_seq = partition_fastafile(fasta_file, train_file, val_file)
    print("total sequences: " + str(total_seq))
    print("train sequences: " + str(n_train_seq))
    print("validation sequences: " + str(n_val_seq))

if __name__ == "__main__":
    main()