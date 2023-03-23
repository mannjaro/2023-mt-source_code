## Install

```shell
pip install -r requirements.txt
source .venv/bin/activate
```

## Run

### RQ1
BOW, TF-IDF, Word2Vecを用いた，不具合修正コミットの分類精度
```shell
python src/rq1
```

### RQ3
BOW, TF-IDF, Word2Vecを用いた，分類後の不具合修正コミットの分類精度
```shell
python src/rq3
```

## 各ディレクトリ

```
.
├── data -> コード中で用いるデータ．コミットメッセージやPRの本文など
│  ├── rq1
│  └── rq3
├── out -> 出力結果
│  ├── rq1
│  └── rq3
├── README.md -> このファイル
├── requirements.txt
├── setup.cfg
└── src -> ソースコード
   ├── preprocess.py
   ├── rq1.py
   └── rq3.py

```