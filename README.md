# RNA pretraining and Secondary Structure Prediction

現段階での実行の仕方のメモ

## 表現学習を行う手順

以下，カレントディレクトリは`myrepo/pretraining/.`

表現学習モデルをトレーニングする場合：
```bash
python train_model.py --framework MLM --train_partition_path ../data/pretrain_data/train_partition.pickle --val_partition_path ../data/pretrain_data/val_partition.pickle --seed 1
```

表現学習モデルを用いて埋め込み表現を得る場合：
```bash
python pretrain.py --framework MLM --test_partition_path ../data/SS_data/ArchiveII.csv --weight_path ../data/pretrain_results/MLM/{ディレクトリ指定}/weight.pth --seed 1
```

その他やりたいこと
- トレーニングデータを分割せずに一つのファイルで入力する場合も対応したい


## 二次構造予測を行う手順

以下，カレントディレクトリは`myrepo/SSpredictor/.`

```bash
python scripts/run_archiveII_kfold.py --emb myrepr_ArchiveII
```