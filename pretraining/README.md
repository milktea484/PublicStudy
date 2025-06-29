# RNA特徴表現を計算する自己教師あり学習モデル

data2vec[1]を使用したRNA表現学習モデル

[1]. Baevski, A., et al., "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language," *Proceedings of the 39th International Conference on Machine Learning*, Vol. 162, PMLR, pp. 1298–1312 (2022). https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec



## 各ファイルの説明
 - contents.py  
    モデルの設定ファイル

 - dataset.py  
    入力データを整形するファイル

 - model.py  
    計算に使用するモデルクラスを定義したファイル．提案手法（data2vec）のほかに，比較手法（MLM）も定義している

 - module.py  
    model.pyで定義するモデルが使用する，モデル内部の細かなモジュールを定義したファイル

 - train_model.py  
    モデルを訓練する際に実行するファイル

 - pretrain.py  
    学習済みモデルを使用して，RNAの特徴表現の計算を実行するファイル

 - utils.py  
    その他の関数を定義するファイル

## 実行コマンド例

モデルを訓練する時
```bash
python train_model.py --framework data2vec --train_partition_path ../data/pretrain_data/train_partition.h5 --val_partition_path ../data/pretrain_data/val_partition.h5 --seed 1
```

学習済みモデルを使用してRNA特徴表現を得る時
```bash
python pretrain.py --framework data2vec --test_partition_path ../data/SS_data/ArchiveII.csv --weight_path ../data/pretrain_results/data2vec/{ディレクトリ指定}/weight.pth --seed 1
```