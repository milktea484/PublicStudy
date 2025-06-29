# 各ディレクトリの説明
- embeddings  
    評価するRNA特徴表現データを保存

- pretrain_data  
    RNA特徴表現を学習するモデル（pretrainingモデル）の学習に使用されるファイルを保存

- pretrain_results  
    学習済みpretrainingモデルの保存と，このモデルにより得られるRNA特徴表現を保存．pretrainingモデルを実行すると自動で結果が保存されていく．

- SS_data  
    pretrainingモデルに通し，評価する対象となるRNA配列データを保存

- SS_results  
    RNA特徴表現の評価結果を保存．SSpredictorモデルを実行すると，使用したRNAデータごとに自動で結果が保存されていく．