
# Chainer: ビギナー向けチュートリアル Vol.2

この記事からわかること：

- データセットオブジェクトを作る方法
- データセットを訓練用・検証用に分割する方法
- 訓練済み重みを持ってきて新しいタスクでFine-tuningする方法
- （おまけ：データセットクラスをフルスクラッチで書く方法）

ここでは、Chainerに予め用意されていないデータセットを外部から調達して、Chainerで記述されたネットワークの訓練のために用いる方法を具体例をもって示します。基本的な手順はほぼ[Chainer: ビギナー向けチュートリアル Vol.1](http://qiita.com/mitmul/items/eccf4e0a84cb784ba84a)で説明したCIFAR10データセットクラスを拡張する章と変わりません。

今回はついでにChainerが用意するデータセットクラス用のユーティリティの一つである`split_dataset_random`を使い、学習用データセットと検証用データセットへの分割を簡単に行う方法も合わせて説明してみます。

また、ターゲットとなるデータと似たドメインのデータセットを用いて予め訓練されたネットワーク重みを初期値として用いる方法も説明します。Caffeの.caffemodelの形で配布されているネットワークをFine-tuningしたい場合、この記事とほぼ同様の手順が適用できると思います。

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

Pillowとtqdmというパッケージを使用しますので、予めこれをインストールしておいてください。インストールは簡単で、pipを用いて以下のようにインストールできます。


```python
%%bash
pip install Pillow
pip install tqdm
```

    Requirement already satisfied: Pillow in /home/shunta/.pyenv/versions/anaconda3-4.4.0/lib/python3.6/site-packages
    Requirement already satisfied: olefile in /home/shunta/.pyenv/versions/anaconda3-4.4.0/lib/python3.6/site-packages (from Pillow)
    Requirement already satisfied: tqdm in /home/shunta/.pyenv/versions/anaconda3-4.4.0/lib/python3.6/site-packages


## 2. 問題設定の確認

今回は、animeface-character-datasetに含まれる様々なキャラクターの顔画像を用いて、未知のキャラクター顔画像が入力された際に、それが既知のクラス一覧の中のどのキャラクターの顔らしいかを出力するようなネットワークを訓練したいと思います。

その際に、**ランダムにパラメータを初期化したネットワークを訓練するのではなく、予め似たドメインで訓練済みのモデルを目的のデータセットでFine-tuningする**というやり方をしてみます。

今回学習に用いるデータセットは、以下のような画像を多数含むデータセットで、各キャラクターごとに予めフォルダ分けがされています。なので、今回もオーソドックスな画像分類問題となります。

#### 適当に抜き出したデータサンプル

| 000_hatsune_miku | 002_suzumiya_haruhi | 007_nagato_yuki | 012_asahina_mikuru |
|:-:|:-:|:-:|:-:|
| ![](animeface-character-dataset/thumb/000_hatsune_miku/face_128_326_108.png) | ![](animeface-character-dataset/thumb/002_suzumiya_haruhi/face_1000_266_119.png) | ![](animeface-character-dataset/thumb/007_nagato_yuki/face_83_270_92.png) | ![](animeface-character-dataset/thumb/012_asahina_mikuru/face_121_433_128.png) |

## 3. データセットオブジェクトの作成

ここでは、画像分類の問題でよく使われる`LabeledImageDataset`というクラスを使ったデータセットオブジェクトの作成を例示します。まずは、Python標準の機能を使ってサクッと下準備をしてしまいます。

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
import cv2 as cv

width, height = 160, 160

# 画像のresize関数
def resize(img):
    img = img.transpose(1, 2, 0)
    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
    return img.transpose(2, 0, 1)

# 各データに行う変換
def transform(inputs):
    img, label = inputs
    img = img[:3, ...]
    img = resize(img)
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
        img = resize(img)[:3]
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

では次に、訓練を行うモデルの定義を行います。ここではIllust2Vecと呼ばれるモデルをベースとし、その最後の2層を削除してランダムに初期化された3つの全結合層を付け加えたものを新しいモデルとします。

学習時には、Illust2Vec由来の部分（3層目以下の部分）の重みは固定しておきます。つまり、新たに追加した3つの全結合層だけを訓練します。

まず、配布されているIllust2Vecモデルの訓練済みパラメータをダウンロードしてきます。


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
```

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

- バッチサイズ128
- 学習率は40エポック目で0.1倍にする

ことにします。


```python
batchsize = 64
gpu_id = 0
lr_drop_epoch = [30, 60]
lr_drop_ratio = 0.1
train_epoch = 100
```


```python
train_iter = iterators.MultiprocessIterator(train, batchsize)
valid_iter = iterators.MultiprocessIterator(
    valid, batchsize, repeat=False, shuffle=False)

optimizer = optimizers.MomentumSGD(lr=0.01)
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
    [J1           1.5845      0.623241       0.647114              0.827381                  37.5897       0.01        
    [J2           0.573204    0.843129       0.542805              0.856575                  54.5251       0.01        
    [J3           0.400987    0.885141       0.497626              0.867564                  71.1841       0.01        
    [J4           0.306406    0.909354       0.449616              0.882218                  87.7798       0.01        
    [J5           0.256398    0.920944       0.443425              0.887712                  104.428       0.01        
    [J6           0.205854    0.938017       0.408577              0.889243                  121.368       0.01        
    [J7           0.172194    0.951055       0.438805              0.88416                   138.23        0.01        
    [J8           0.156446    0.952401       0.393488              0.898289                  155.064       0.01        
    [J9           0.137947    0.960679       0.407817              0.894177                  172.172       0.01        
    [J10          0.115862    0.963576       0.425544              0.889391                  189.335       0.01        
    [J11          0.0950843   0.972889       0.403892              0.896644                  206.608       0.01        
    [J12          0.0966766   0.972786       0.418247              0.889505                  223.521       0.01        
    [J13          0.0890769   0.975166       0.394598              0.898551                  240.946       0.01        
    [J14          0.0790933   0.977959       0.395257              0.897878                  258.083       0.01        
    [J15          0.0727327   0.979926       0.406979              0.896906                  275.095       0.01        
    [J16          0.0701191   0.980937       0.395499              0.899933                  292.157       0.01        
    [J17          0.0618324   0.983444       0.392069              0.901578                  309.212       0.01        
    [J18          0.0598441   0.984996       0.388061              0.897729                  326.36        0.01        
    [J19          0.0556599   0.984892       0.399634              0.902663                  343.529       0.01        
    [J20          0.0533907   0.984996       0.368317              0.906924                  360.742       0.01        
    [J21          0.050829    0.985099       0.389692              0.911298                  377.671       0.01        
    [J22          0.0526459   0.986962       0.383639              0.907597                  394.641       0.01        
    [J23          0.0464965   0.988411       0.403357              0.912007                  411.798       0.01        
    [J24          0.043239    0.989963       0.390547              0.904456                  428.64        0.01        
    [J25          0.0422824   0.990066       0.401114              0.905279                  445.703       0.01        
    [J26          0.0396545   0.991411       0.398242              0.900607                  462.689       0.01        
    [J27          0.0369248   0.990584       0.392404              0.906775                  479.699       0.01        
    [J28          0.0343193   0.992239       0.390253              0.905393                  497.061       0.01        
    [J29          0.035577    0.992239       0.384158              0.904868                  513.832       0.01        
    [J30          0.0343121   0.990066       0.386389              0.904719                  530.636       0.01        
    [J31          0.0288961   0.992964       0.366141              0.907335                  548.036       0.001       
    [J32          0.0247479   0.994896       0.374929              0.910773                  565.174       0.001       
    [J33          0.0253498   0.994205       0.375904              0.91324                   582.226       0.001       
    [J34          0.0235029   0.995447       0.388474              0.907335                  599.114       0.001       
    [J35          0.0227216   0.995551       0.373497              0.906512                  616.347       0.001       
    [J36          0.0241614   0.995137       0.364512              0.910624                  633.523       0.001       
    [J37          0.0228362   0.995964       0.371467              0.911298                  650.618       0.001       
    [J38          0.0246799   0.994205       0.369694              0.911858                  667.656       0.001       
    [J39          0.0226526   0.995757       0.37464               0.907186                  684.436       0.001       
    [J40          0.0197103   0.995861       0.385321              0.903634                  701.361       0.001       
    [J41          0.02149     0.995654       0.378063              0.908717                  718.29        0.001       
    [J42          0.0204457   0.995861       0.378268              0.907597                  735.235       0.001       
    [J43          0.0245075   0.994723       0.365067              0.912531                  752.288       0.001       
    [J44          0.0212647   0.995861       0.374565              0.909539                  769.143       0.001       
    [J45          0.0200397   0.995861       0.372792              0.907597                  786.034       0.001       
    [J46          0.0211206   0.995033       0.370955              0.911858                  803.11        0.001       
    [J47          0.018989    0.996585       0.368132              0.910064                  820.166       0.001       
    [J48          0.0208812   0.995938       0.355737              0.909951                  837.154       0.001       
    [J49          0.0231893   0.994205       0.389451              0.90898                   854.281       0.001       
    [J50          0.0210398   0.995757       0.357119              0.911447                  871.358       0.001       
    [J51          0.0187993   0.996792       0.362248              0.910624                  888.604       0.001       
    [J52          0.0208949   0.995757       0.372857              0.908831                  905.899       0.001       
    [J53          0.0206167   0.995344       0.370687              0.910887                  922.899       0.001       
    [J54          0.0214873   0.996792       0.364726              0.912007                  940.034       0.001       
    [J55          0.0211626   0.996068       0.367191              0.911035                  956.814       0.001       
    [J56          0.0197502   0.996482       0.361838              0.913503                  973.801       0.001       
    [J57          0.0207524   0.996689       0.369392              0.909653                  990.69        0.001       
    [J58          0.0193296   0.996275       0.374424              0.909539                  1007.79       0.001       
    [J59          0.0200938   0.996275       0.367831              0.912418                  1025.08       0.001       
    [J60          0.0211822   0.99493        0.375867              0.90842                   1042.11       0.001       
    [J61          0.0209016   0.995654       0.365966              0.913651                  1059.24       0.0001      
    [J62          0.0183563   0.996999       0.371247              0.910064                  1076.18       0.0001      
    [J63          0.0191735   0.995964       0.360612              0.909242                  1093.35       0.0001      
    [J64          0.0190406   0.996458       0.360736              0.910887                  1110.51       0.0001      
    [J65          0.0185012   0.99731        0.362603              0.910624                  1127.58       0.0001      
    [J66          0.021157    0.995344       0.368636              0.911035                  1144.86       0.0001      
    [J67          0.0208782   0.995447       0.366127              0.912269                  1161.73       0.0001      
    [J68          0.0194636   0.996171       0.366277              0.910476                  1178.91       0.0001      
    [J69          0.017752    0.996378       0.364377              0.908831                  1195.97       0.0001      
    [J70          0.0191976   0.996068       0.368126              0.910476                  1213.39       0.0001      
    [J71          0.0190953   0.996585       0.369903              0.911858                  1230.61       0.0001      
    [J72          0.0191564   0.995551       0.37065               0.910213                  1247.99       0.0001      
    [J73          0.0210153   0.995861       0.388312              0.908008                  1265.23       0.0001      
    [J74          0.019119    0.996275       0.363296              0.910624                  1282.13       0.0001      
    [J75          0.0190967   0.995861       0.370107              0.910362                  1299.23       0.0001      
    [J76          0.0174288   0.997103       0.367408              0.913091                  1316.29       0.0001      
    [J77          0.0178129   0.996585       0.379064              0.909951                  1333.37       0.0001      
    [J78          0.0202339   0.996171       0.357062              0.91212                   1350.45       0.0001      
    [J79          0.0186008   0.996482       0.358454              0.915147                  1367.56       0.0001      
    [J80          0.0184623   0.995833       0.36624               0.911035                  1384.7        0.0001      
    [J81          0.0205136   0.995344       0.360511              0.916792                  1401.67       0.0001      
    [J82          0.0196852   0.995964       0.387059              0.908306                  1418.57       0.0001      
    [J83          0.020936    0.995861       0.377162              0.910624                  1435.91       0.0001      
    [J84          0.0208509   0.995757       0.354316              0.913503                  1453          0.0001      
    [J85          0.0195411   0.996171       0.355391              0.909242                  1470.57       0.0001      
    [J86          0.0174101   0.996689       0.352271              0.917763                  1487.44       0.0001      
    [J87          0.0194651   0.995861       0.363054              0.914062                  1504.57       0.0001      
    [J88          0.018105    0.995757       0.369398              0.912943                  1522.05       0.0001      
    [J89          0.0173783   0.996171       0.37911               0.908717                  1539.22       0.0001      
    [J90          0.0193303   0.995964       0.356043              0.913651                  1556.1        0.0001      
    [J91          0.018608    0.996482       0.358634              0.913503                  1573.19       0.0001      
    [J92          0.0185521   0.996585       0.36978               0.909802                  1589.94       0.0001      
    [J93          0.0190667   0.995964       0.364025              0.916232                  1606.96       0.0001      
    [J94          0.0197768   0.995861       0.371049              0.90898                   1623.98       0.0001      
    [J95          0.0185012   0.996378       0.36765               0.910624                  1640.87       0.0001      
    [J96          0.0207783   0.996562       0.371219              0.907746                  1657.73       0.0001      
    [J97          0.0186797   0.995861       0.364412              0.911709                  1674.79       0.0001      
    [J98          0.0196398   0.996482       0.36925               0.908008                  1692.16       0.0001      
    [J99          0.0195398   0.995861       0.358516              0.911858                  1709.47       0.0001      
    [J100         0.0179573   0.996689       0.363717              0.909242                  1726.39       0.0001      


標準出力に出る途中経過は上記のような感じでした。最終的に検証用データセットに対しても90%以上のaccuracyが出せていますね。では、画像ファイルとして保存されている学習経過でのロスカーブとaccuracyのカーブを表示してみます。


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
for _ in range(15):
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

    pred: 071_nagase_minato label: 071_nagase_minato True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_1.png)


    pred: 014_hiiragi_kagami label: 014_hiiragi_kagami True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_3.png)


    pred: 030_aisaka_taiga label: 030_aisaka_taiga True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_5.png)


    pred: 004_takamachi_nanoha label: 004_takamachi_nanoha True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_7.png)


    pred: 064_amami_haruka label: 064_amami_haruka True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_9.png)


    pred: 091_komaki_manaka label: 091_komaki_manaka True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_11.png)


    pred: 086_tsuruya label: 086_tsuruya True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_13.png)


    pred: 060_ichinose_kotomi label: 060_ichinose_kotomi True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_15.png)


    pred: 107_chii label: 107_chii True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_17.png)


    pred: 001_kinomoto_sakura label: 001_kinomoto_sakura True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_19.png)


    pred: 054_horo label: 054_horo True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_21.png)


    pred: 008_shana label: 008_shana True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_23.png)


    pred: 061_furude_rika label: 061_furude_rika True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_25.png)


    pred: 023_hiiragi_tsukasa label: 023_hiiragi_tsukasa True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_27.png)


    pred: 093_yuuki_mikan label: 093_yuuki_mikan True



![png](5-Write-new-dataset-class_ja_files/5-Write-new-dataset-class_ja_38_29.png)


ランダムに15枚選んでみたところこの画像たちに対しては全て正解できました。

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
