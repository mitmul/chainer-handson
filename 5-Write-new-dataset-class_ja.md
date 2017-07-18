
# Chainer: ビギナー向けチュートリアル Vol.2

ここでは、Animeface-Characterデータセットを使って、146種類のキャラクター顔画像を90%以上の精度で分類するという課題を通して、以下のことを説明します。

- データセットオブジェクトを作る方法
- データセットを訓練用・検証用に分割する方法
- 訓練済み重みを持ってきて新しいタスクでFine-tuningする方法
- （おまけ：データセットクラスをフルスクラッチで書く方法）

使用した環境は以下です。

- NVIDIA Pascal TITAN X
- Ubuntu 16.04

使用するライブラリは以下です。

- Chainer 2.0.1
- CuPy 1.0.1
- Pillow
- tqdm

ここでは、**Chainerに予め用意されていないデータセットを外部から調達して**、Chainerで記述されたネットワークの訓練のために用いる方法を具体例をもって示します。基本的な手順はほぼ[Chainer: ビギナー向けチュートリアル Vol.1](http://qiita.com/mitmul/items/eccf4e0a84cb784ba84a)で説明したCIFAR10データセットクラスを拡張する章と変わりません。

今回はついでにChainerが用意するデータセットクラス用のユーティリティの一つである`split_dataset_random`を使い、**学習用データセットと検証用データセットへの分割を簡単に行う方法**も合わせて説明してみます。

また、**ターゲットとなるデータと似たドメインのデータセットを用いて予め訓練されたネットワーク重みを初期値として用いる方法**も説明します。Caffeの.caffemodelの形で配布されているネットワークをFine-tuningしたい場合、この記事とほぼ同様の手順が適用できると思います。

この記事は[もともとJupyter notebookで書いたもの](https://github.com/mitmul/chainer-handson/blob/master/5-Write-new-dataset-class_ja.ipynb)を `jupyter nbconvert --to markdown 5-Write-new-dataset-class_ja.ipynb` したものです。

## 1. データセットのダウンロード

まずは、データセットをダウンロードしてきます。今回は、Kaggle Grand Masterであらせられるnagadomiさんが[こちら](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/)で配布されているアニメキャラクターの顔領域サムネイルデータセットを使用します。


```python
%%bash
if [ ! -d animeface-character-dataset ]; then
    curl -L -O http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip
    unzip animeface-character-dataset.zip
    rm -rf animeface-character-dataset.zip
fi
```

使用するライブラリをpipで入れます。


```python
%%bash
pip install chainer
pip install cupy
pip install Pillow
pip install tqdm
```

    Requirement already satisfied: chainer in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages
    Requirement already satisfied: filelock in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages/filelock-2.0.8-py3.6.egg (from chainer)
    Requirement already satisfied: nose in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages (from chainer)
    Requirement already satisfied: six>=1.9.0 in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages (from chainer)
    Requirement already satisfied: protobuf>=2.6.0 in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages/protobuf-3.3.0-py3.6.egg (from chainer)
    Requirement already satisfied: numpy>=1.9.0 in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages (from chainer)
    Requirement already satisfied: setuptools in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages/setuptools-27.2.0-py3.6.egg (from protobuf>=2.6.0->chainer)
    Requirement already satisfied: cupy in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages
    Requirement already satisfied: nose in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages (from cupy)
    Requirement already satisfied: numpy>=1.9.0 in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages (from cupy)
    Requirement already satisfied: six>=1.9.0 in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages (from cupy)
    Requirement already satisfied: Pillow in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages
    Requirement already satisfied: olefile in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages (from Pillow)
    Requirement already satisfied: tqdm in /home/shunta/.pyenv/versions/anaconda3-4.3.1/lib/python3.6/site-packages


## 2. 問題設定の確認

今回は、animeface-character-datasetに含まれる様々なキャラクターの顔画像を用いて、未知のキャラクター顔画像が入力された際に、それが既知のクラス一覧の中のどのキャラクターの顔らしいかを出力するようなネットワークを訓練したいと思います。

その際に、**ランダムにパラメータを初期化したネットワークを訓練するのではなく、予め似たドメインで訓練済みのモデルを目的のデータセットでFine-tuningする**というやり方をしてみます。

今回学習に用いるデータセットは、以下のような画像を多数含むデータセットで、各キャラクターごとに予めフォルダ分けがされています。なので、今回もオーソドックスな画像分類問題となります。

#### 適当に抜き出したデータサンプル

| 000_hatsune_miku | 002_suzumiya_haruhi | 007_nagato_yuki | 012_asahina_mikuru |
|:-:|:-:|:-:|:-:|
| ![](animeface-character-dataset/thumb/000_hatsune_miku/face_128_326_108.png) | ![](animeface-character-dataset/thumb/002_suzumiya_haruhi/face_1000_266_119.png) | ![](animeface-character-dataset/thumb/007_nagato_yuki/face_83_270_92.png) | ![](animeface-character-dataset/thumb/012_asahina_mikuru/face_121_433_128.png) |

## 3. データセットオブジェクトの作成

ここでは、画像分類の問題でよく使われる`LabeledImageDataset`というクラスを使ったデータセットオブジェクトの作成方法を示します。まずは、Python標準の機能を使って下準備をします。

初めに画像ファイルへのパス一覧を取得します。画像ファイルは、`animeface-character-dataset/thumb`以下にキャラクターごとのディレクトリに分けられて入っています。下記のコードでは、フォルダ内に`ignore`というファイルが入っている場合は、そのフォルダの画像は無視するようにしています。


```python
import os
import glob
from itertools import chain

# 画像フォルダ
IMG_DIR = 'animeface-character-dataset/thumb'

# 各キャラクターごとのフォルダ
dnames = glob.glob('{}/*'.format(IMG_DIR))

# 画像ファイルパス一覧
fnames = [glob.glob('{}/*.png'.format(d)) for d in dnames
          if not os.path.exists('{}/ignore'.format(d))]
fnames = list(chain.from_iterable(fnames))
```

次に、画像ファイルパスのうち画像が含まれるディレクトリ名の部分がキャラクター名を表しているので、それを使って各画像にキャラクターごとに一意になるようなIDを作ります。


```python
# それぞれにフォルダ名から一意なIDを付与
labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
dnames = [os.path.basename(d) for d in dnames
          if not os.path.exists('{}/ignore'.format(d))]
labels = [dnames.index(l) for l in labels]
```

では、ベースとなるデータセットオブジェクトを作ります。やり方は簡単で、ファイルパスとそのラベルが並んだタプルのリストを`LabeledImageDataset`に渡せば良いだけです。これは `(img, label)` のようなタプルを返すイテレータになっています。


```python
from chainer.datasets import LabeledImageDataset

# データセット作成
d = LabeledImageDataset(list(zip(fnames, labels)))
```

次に、Chainerが提供している`TransformDataset`という便利な機能を使ってみます。これは、データセットオブジェクトと各データへの変換を表す関数を取るラッパークラスで、これを使うとdata augmentationや前処理などを行う部分をデータセットクラスの外に用意しておくことができます。


```python
from chainer.datasets import TransformDataset
from PIL import Image

width, height = 160, 160

# 画像のresize関数
def resize(img):
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height), Image.BICUBIC)
    return np.asarray(img).transpose(2, 0, 1)

# 各データに行う変換
def transform(inputs):
    img, label = inputs
    img = img[:3, ...]
    img = resize(img.astype(np.uint8))
    img = img - mean[:, None, None]
    img = img.astype(np.float32)
    # ランダムに左右反転
    if np.random.rand() > 0.5:
        img = img[..., ::-1]
    return img, label

# 変換付きデータセットにする
td = TransformDataset(d, transform)
```

こうすることで、`LabeledImageDataset`オブジェクトである`d`が返す `(img, label)` のようなタプルを受け取って、それを`transform`関数にくぐらせてから返すようなデータセットオブジェクトが作れました。

では、これを学習用と検証用の2つの部分データセットにsplitしましょう。今回は、データセット全体のうち80%を学習用に、残り20%を検証用に使うことにします。`split_dataset_random`を使うと、データセット内のデータを一度シャッフルしたのちに、指定した区切り目で分割したものを返してくれます。


```python
from chainer import datasets

train, valid = datasets.split_dataset_random(td, int(len(d) * 0.8), seed=0)
```

データセットの分割は他にも、交差検定をするための複数の互いに異なる訓練・検証用データセットペアを返すような`get_cross_validation_datasets_random`など、いくつかの関数が用意されています。こちらをご覧ください。：[SubDataset](https://docs.chainer.org/en/stable/reference/datasets.html#subdataset)

さて、変換の中で使っている`mean`は、今回使う学習用データセットに含まれる画像の平均画像です。これを計算しておきましょう。


```python
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook

# 平均画像が未計算なら計算する
if not os.path.exists('image_mean.npy'):
    # 変換をかまさないバージョンの学習用データセットで平均を計算したい
    t, _ = datasets.split_dataset_random(d, int(len(d) * 0.8), seed=0)

    mean = np.zeros((3, height, width))
    for img, _ in tqdm_notebook(t, desc='Calc mean'):
        img = resize(img[:3].astype(np.uint8))
        mean += img
    mean = mean / float(len(d))
    np.save('image_mean', mean)
else:
    mean = np.load('image_mean.npy')
```

試しに計算した平均画像を表示してみましょう。


```python
# 平均画像の表示
plt.imshow(mean.transpose(1, 2, 0) / 255)
plt.show()
```


![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_20_0.png)


なんか怖いですね…

平均を引くときはピクセルごとの平均にしてしまうので、この平均画像の平均ピクセルを計算しておきます。


```python
mean = mean.mean(axis=(1, 2))
```

## 4. モデルの定義とFine-tuningの準備

では次に、訓練を行うモデルの定義を行います。ここでは[Illustration2Vec](http://illustration2vec.net/)と呼ばれるモデルをベースとし、その最後の2層を削除してランダムに初期化された3つの全結合層を付け加えたものを新しいモデルとします。

学習時には、Illustration2Vec由来の部分（3層目以下の部分）の重みは固定しておきます。つまり、新たに追加した3つの全結合層だけを訓練します。

まず、配布されているIllustration2Vecモデルの訓練済みパラメータをダウンロードしてきます。


```python
%%bash
if [ ! -f illust2vec_ver200.caffemodel ]; then
    curl -L -O http://illustration2vec.net/models/illust2vec_ver200.caffemodel
fi
```

この訓練済みパラメータはcaffemodelの形式で提供されていますが、Chainerには非常に簡単にCaffeの訓練済みモデルを読み込む機能（`CaffeFunction`）があるので、これを使ってパラメータとモデル構造をロードします。ただし、読み込みには時間がかかるため、一度読み込んだ際に得られる`Chain`オブジェクトをPython標準の`pickle`を使ってファイルに保存しておきます。こうすることで次回からの読み込みが速くなります。

実際のネットワークのコードは以下のようになります。


```python
import pickle

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import Chain
from chainer.links.caffe import CaffeFunction


class Illust2Vec(Chain):

    CAFFEMODEL_FN = 'illust2vec_ver200.caffemodel'
    PKL_FN = 'illust2vec_ver200.pkl'

    def __init__(self, n_classes, unchain=True):
        w = chainer.initializers.HeNormal()        
        if not os.path.exists(self.PKL_FN):  # 変換済みのChainerモデル（PKLファイル）が無い場合
            model = CaffeFunction(self.CAFFEMODEL_FN)  # CaffeModelを読み込んで保存します。（時間がかかります）
            pickle.dump(model, open(self.PKL_FN, 'wb'))  # 一度読み込んだら、次回から高速に読み込めるようPickleします。
        else:
            model = pickle.load(open(self.PKL_FN, 'rb'))
        del model.encode1  # メモリ節約のため不要なレイヤを削除します。
        del model.encode2
        del model.forwards['encode1']
        del model.forwards['encode2']
        model.layers = model.layers[:-2]
        
        super(Illust2Vec, self).__init__()
        with self.init_scope():
            self.trunk = model  # 元のIllust2Vecモデルをtrunkとしてこのモデルに含めます。
            self.fc7 = L.Linear(None, 4096, initialW=w)
            self.bn7 = L.BatchNormalization(4096)
            self.fc8 = L.Linear(4096, n_classes, initialW=w)
            
        self.unchain = True

    def __call__(self, x):
        h = self.trunk({'data': x}, ['conv6_3'])[0]  # 元のIllust2Vecモデルのconv6_3の出力を取り出します。
        if self.unchain:
            h.unchain_backward()
        h = F.dropout(F.relu(self.bn7(self.fc7(h))))  # ここ以降は新しく追加した層です。
        return self.fc8(h)

n_classes = len(dnames)
model = Illust2Vec(n_classes)
model = L.Classifier(model)
model.to_gpu(0)
```




    <chainer.links.model.classifier.Classifier at 0x7f3d8c636390>



`__call__`の部分に`h.unchain_backward()`という記述が登場しました。`unchain_backward`は、ネットワークのある中間出力`Variable` などから呼ばれ、その時点より前のあらゆるネットワークノードの接続を断ち切ります。そのため、学習時にはこれが呼ばれた時点より前の層に誤差が伝わらなくなり、結果としてパラメータの更新も行われなくなります。

前述の

> 学習時には、Illust2Vec由来の部分（3層目以下の部分）の重みは固定しておきます

これを行うためのコードが、この`h.unchain_backward()`です。

このあたりの仕組みについて、さらに詳しくは、Define-by-RunによるChainerのautogradの仕組みを説明しているこちらの記事を参照してください。: [1-file Chainerを作る](http://qiita.com/mitmul/items/37d3932292cdd560d418)

## 5. 学習

それでは、このデータセットとモデルを用いて、学習を行ってみます。まず必要なモジュールをロードしておきます。


```python
from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer.training import extensions
from chainer.training import triggers
from chainer.dataset import concat_examples
```

次に学習のパラメータを設定します。今回は

- バッチサイズ64
- 学習率は0.01からスタートし、10エポック目で0.1倍にする
- 20エポックで学習終了

とします。


```python
batchsize = 64
gpu_id = 0
initial_lr = 0.01
lr_drop_epoch = [10]
lr_drop_ratio = 0.1
train_epoch = 20
```


```python
train_iter = iterators.MultiprocessIterator(train, batchsize)
valid_iter = iterators.MultiprocessIterator(
    valid, batchsize, repeat=False, shuffle=False)

optimizer = optimizers.MomentumSGD(lr=initial_lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

updater = training.StandardUpdater(
    train_iter, optimizer, device=gpu_id)

trainer = training.Trainer(updater, (train_epoch, 'epoch'), out='AnimeFace-result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.observe_lr())

# 標準出力に書き出したい値
trainer.extend(extensions.PrintReport(
    ['epoch',
     'main/loss',
     'main/accuracy',
     'validation/main/loss',
     'validation/main/accuracy',
     'elapsed_time',
     'lr']))

# ロスのプロットを毎エポック自動的に保存
trainer.extend(extensions.PlotReport(
        ['main/loss',
         'validation/main/loss'],
        'epoch', file_name='loss.png'))

# 精度のプロットも毎エポック自動的に保存
trainer.extend(extensions.PlotReport(
        ['main/accuracy',
         'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))

# モデルのtrainプロパティをFalseに設定してvalidationするextension
trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu_id))

# 指定したエポックごとに学習率を10分の1にする

def lr_drop(trainer):
    trainer.updater.get_optimizer('main').lr *= lr_drop_ratio

trainer.extend(
    lr_drop,
    trigger=triggers.ManualScheduleTrigger(lr_drop_epoch, 'epoch'))

trainer.run()
```

    epoch       main/loss   main/accuracy  validation/main/loss  validation/main/accuracy  elapsed_time  lr        
    [J1           1.57341     0.626759       0.596217              0.834188                  24.9535       0.01        
    [J2           0.592884    0.834851       0.529221              0.854108                  41.7383       0.01        
    [J3           0.398748    0.881312       0.487583              0.867564                  58.4842       0.01        
    [J4           0.33203     0.90418        0.427361              0.884571                  75.3357       0.01        
    [J5           0.227967    0.932119       0.433671              0.887747                  91.9539       0.01        
    [J6           0.207433    0.937707       0.426151              0.889243                  108.593       0.01        
    [J7           0.170046    0.948986       0.410441              0.890739                  125.667       0.01        
    [J8           0.142838    0.956436       0.406034              0.899671                  142.823       0.01        
    [J9           0.142185    0.958092       0.416965              0.894737                  159.578       0.01        
    [J10          0.117041    0.965956       0.408871              0.899111                  176.432       0.01        
    [J11          0.0944959   0.972889       0.388597              0.896906                  193.15        0.001       
    [J12          0.0734699   0.980753       0.355984              0.907484                  209.978       0.001       
    [J13          0.0731715   0.979719       0.376939              0.901167                  227.502       0.001       
    [J14          0.0722281   0.983651       0.364557              0.910362                  244.675       0.001       
    [J15          0.069801    0.982099       0.366278              0.90513                   261.601       0.001       
    [J16          0.0736745   0.97875        0.362352              0.904981                  278.607       0.001       
    [J17          0.0646118   0.982926       0.351286              0.90457                   295.928       0.001       
    [J18          0.0591635   0.984789       0.360433              0.907484                  312.508       0.001       
    [J19          0.0612314   0.985099       0.35836               0.907597                  330.281       0.001       
    [J20          0.0577607   0.985824       0.357249              0.90786                   347.66        0.001       


6分半くらいで学習が終わりました。標準出力に出る途中経過は上記のような感じでした。最終的に検証用データセットに対しても90%以上のaccuracyが出せていますね。では、画像ファイルとして保存されている学習経過でのロスカーブとaccuracyのカーブを表示してみます。


```python
from IPython.display import Image
Image(filename='AnimeFace-result/loss.png')
```




![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_34_0.png)




```python
Image(filename='AnimeFace-result/accuracy.png')
```




![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_35_0.png)



無事収束している感じがします。

最後に、いくつかvalidationデータセットから画像を取り出してきて個別の分類結果を見てみます。


```python
%matplotlib inline
import matplotlib.pyplot as plt

from PIL import Image
from chainer import cuda
```


```python
chainer.config.train = False
for _ in range(10):
    x, t = valid[np.random.randint(len(valid))]
    x = cuda.to_gpu(x)
    y = F.softmax(model.predictor(x[None, ...]))
    
    pred = os.path.basename(dnames[int(y.data.argmax())])
    label = os.path.basename(dnames[t])
    
    print('pred:', pred, 'label:', label, pred == label)

    x = cuda.to_cpu(x)
    x += mean[:, None, None]
    x = x / 256
    x = np.clip(x, 0, 1)
    plt.imshow(x.transpose(1, 2, 0))
    plt.show()
```

    pred: 010_izumi_konata label: 010_izumi_konata True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_1.png)


    pred: 134_nunnally_lamperouge label: 134_nunnally_lamperouge True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_3.png)


    pred: 120_asakura_yume label: 120_asakura_yume True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_5.png)


    pred: 081_yuzuhara_konomi label: 081_yuzuhara_konomi True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_7.png)


    pred: 185_akihime_sumomo label: 185_akihime_sumomo True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_9.png)


    pred: 001_kinomoto_sakura label: 001_kinomoto_sakura True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_11.png)


    pred: 159_andou_mahoro label: 159_andou_mahoro True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_13.png)


    pred: 105_yuno label: 105_yuno True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_15.png)


    pred: 097_kamikita_komari label: 097_kamikita_komari True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_17.png)


    pred: 051_houjou_reika label: 051_houjou_reika True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_19.png)


ランダムに10枚選んでみたところこの画像たちに対しては全て正解できました。

最後に、いつか何かに使うかもしれないので、一応snapshotを保存しておきます。


```python
from chainer import serializers

serializers.save_npz('animeface.model', model)
```

## 6. おまけ1：データセットクラスをフルスクラッチで書く方法

データセットクラスをフルスクラッチで書くには、`chainer.dataset.DatasetMixin`クラスを継承した自前クラスを用意すれば良いです。そのクラスは`__len__`メソッドと`get_example`メソッドを持つ必要があります。例えば以下のようになります。


```python
class MyDataset(chainer.dataset.DatasetMixin):
    
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.image_paths)
    
    def get_example(self, i):
        img = cv.imread(self.image_paths[i])
        img = img.transpose(2, 0, 1).astype(np.float32)
        label = self.labels[i]
        return img, label
```

これは、コンストラクタに画像ファイルパスのリストと、それに対応した順番でラベルを並べたリストを渡しておき、`[]`アクセサでインデックスを指定すると、対応するパスから画像を読み込んで、ラベルと並べたタプルを返すデータセットクラスになっています。例えば、以下のように使えます。

```python
image_files = ['images/hoge_0_1.png', 'images/hoge_5_1.png', 'images/hoge_2_1.png', 'images/hoge_3_1.png', ...]
labels = [0, 5, 2, 3, ...]

dataset = MyDataset(image_files, labels)

img, label = dataset[2]

#=> 読み込まれた 'images/hoge_2_1.png' 画像データと、そのラベル（ここでは2）が返る
```

このオブジェクトは、そのままIteratorに渡すことができ、Trainerを使った学習に使えます。つまり、

```python
train_iter = iterators.MultiprocessIterator(dataset, batchsize=128)
```

のようにしてイテレータを作って、UpdaterにOptimizerと一緒に渡せば、Trainerをいつも通りに使えます。

## 7. おまけ2：最もシンプルなデータセットオブジェクトの作り方

実はChainerのTrainerと一緒に使うためのデータセットは、**単なるPythonのリストでOK**です。どういうことかというと、`len()`で長さが取得でき、`[]`アクセサで要素が取り出せるものなら、**全てデータセットオブジェクトとして扱う事ができる**ということです。例えば、

```python
data_list = [(x1, t1), (x2, t2), ...]
```

のような`(データ, ラベル)`というタプルのリストを作れば、これはIteratorに渡すことができます。

```python
train_iter = iterators.MultiprocessIterator(data_list, batchsize=128)
```

ただこういったやりかたの欠点は、データセット全体を学習前にメモリに載せなければいけない点です。これを防ぐために、ImageDatasetとTupleDatasetを組み合わせる方法やLabaledImageDatasetといったクラスが用意されています。詳しくはドキュメントをご参照ください。
http://docs.chainer.org/en/stable/reference/datasets.html#general-datasets
