# RNA特徴表現を評価するためのRNA二次構造予測モデル
先行研究[2],[3]を参考に作成

[2]. Zablocki, L. I., et al., “Comprehensive benchmarking of large language models for RNA secondary structure prediction,” *Briefings in Bioinformatics*, Vol. 26, Issue 2 (2025). https://github.com/sinc-lab/rna-llm-folding

[3]. Gong, T., et al., “Accurate prediction of RNA secondary structure including pseudoknots through solving minimum-cost flow with learned potentials,” *Communications Biology*,  Vol.7, pp. 297 (2024). https://github.com/gongtiansu/KnotFold

## ディレクトリとファイルの説明
### src  
二次構造予測モデルの実装

- dataset.py  
    入力整形用ファイル

- model.py  
    二次構造予測モデル本体

- metrics.py  
    二次構造予測の精度を算出する関数を定義

- train_model.py  
    二次構造予測モデルの訓練に使用

- test_model.py  
    学習済みモデルを使用して二次構造を予測する時に使用

- utils.py  
    その他の関数を定義

- KnotFold_mincostflow.cc  
    二次構造の計算に使用．先行研究のソースコードをそのまま引用．

### scripts
srcで実装された二次構造予測モデルで，実験を行うためのディレクトリ．RNAデータセットごとにファイルが分けられている．

### notebooks
実験により得られた精度を，分かりやすいグラフに置き換えるためのnotebook

## 実行コマンド例

```bash
python scripts/run_archiveII_kfold.py --emb {data/embeddingsにあるファイル名}
```
