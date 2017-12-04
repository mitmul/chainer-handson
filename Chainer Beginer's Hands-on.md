
Chainer Advent Calender 3日目です。以前書いた[Chainerビギナー向けチュートリアル](https://qiita.com/mitmul/items/eccf4e0a84cb784ba84a)がもう古くなってしまったので、内容を現時点の最新のstableバージョンである3.1.0向けに更新したものをおいておきます。一部内容的にも加筆・修正を加えています。

**注意：**

- 今回はニューラルネットワーク自体が何なのかといった説明は省きます。
- この記事はJupyter notebookを使って書かれていますので、コードは上から順番に実行できるようにチェックされています。元のJupyter notebookファイルはこちらにおいてあります。

Qiitaだとページ内リンクつきの目次が勝手に作成されるので、全体概要はそちらを眺めて把握してください。

# インストール

Chainerのインストールはとても簡単です。Chainer本体はすべてPythonコードのみからなるので、インストールも

```bash
pip install chainer
```

で完了です。ただ、これだけではGPUは使えません。GPUを使うためには、**別途CuPyをインストールする必要があります。**ただCuPyのインストールもとても簡単です。

```bash
pip install cupy
```

以上です。ただ、cuDNNやNCCLなどのNVIDIAライブラリを有効にしたい場合は、**CuPyをインストールする前に**事前にやっておくことがあります。cuDNNに関しては、cudnnenvを使うのがおすすめです。

```bash
pip install cudnnenv
cudnnenv install v7.0.3-cuda9
cudnnenv activate v7.0.3-cuda9
```

cudnnenvはいろいろなバージョンのcuDNNを簡単にホーム以下（`~/.cudnn`）にダウンロードしてくれるツールです。自分の環境に入っているCUDAのバージョンに合わせて入れましょう。（例：CUDA8の環境なら `cudnnenv install v7.0.3-cuda8` のようにする）cudnnenvでcuDNNを落としてきたあとは、**CuPyをインストールする前に以下の環境変数をセットします。**

```bash
LD_LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LD_LIBRARY_PATH
CPATH=~/.cudnn/active/cuda/include:$CPATH
LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LIBRARY_PATH
```

そのあと `pip install cupy` をすればcuDNNが有効になります。NCCLについてはcudnnenvのようなツールはないので自分でダウンロードしてきて設置する必要がありますが、あとの手順はどうようなので割愛します。詳しくはこちらを御覧ください：[Install CuPy with cuDNN and NCCL](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy-with-cudnn-and-nccl)

# 学習ループを書いてみよう

ここでは、有名な手書き数字のデータセットMNISTを使って、画像を10クラスに分類するネットワークを書いて訓練してみます。

## 1. データセットの準備

教師あり学習の場合、**データセットは「入力データ」と「それと対になるラベルデータ」を返すオブジェクトである必要があります。**
ChainerにはMNISTやCIFAR10/100のようなよく用いられるデータセットに対して、データをダウンロードしてくるところからそのような機能をもったオブジェクトを作るところまで自動的にやってくれる便利なメソッドがあるので、ここではひとまずこれを用いましょう。


```python
from chainer.datasets import mnist

# データセットがダウンロード済みでなければ、ダウンロードも行う
train, test = mnist.get_mnist(withlabel=True, ndim=1)
```

データセットオブジェクト自体は準備ができました。これは、例えば `train[i]` などとすると**i番目の `(data, label)` というタプルを返すリスト** と同様のものになっています（**実際ただのPythonリストもChainerのデータセットオブジェクトとして使えます**）。では0番目のデータとラベルを取り出して、表示してみましょう。


```python
# matplotlibを使ったグラフ描画結果がnotebook内に表示されるようにします。
%matplotlib inline
import matplotlib.pyplot as plt

# データの例示
x, t = train[0]  # 0番目の (data, label) を取り出す
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
print('label:', t)
```


![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_6_0.png)


    label: 5


## 2. Iteratorの作成

データセットの準備は完了しましたが、このままネットワークの学習に使うのは少し面倒です。なぜなら、ネットワークのパラメータ最適化手法として広く用いられているStochastic Gradient Descent (SGD)という手法では、一般的にいくつかのデータを束ねた**ミニバッチ**と呼ばれる単位でネットワークにデータを渡し、それに対する予測を作って、ラベルと比較するということを行います。そのため、**バッチサイズ分だけデータとラベルを束ねる作業が必要です。**

そこで、**データセットから決まった数のデータとラベルを取得し、それらを束ねてミニバッチを作ってくれる機能を持った`Iterator`を使いましょう。**`Iterator`は、先程作ったデータセットオブジェクトを渡して初期化してやったあとは、`next()`メソッドで新しいミニバッチを返してくれます。内部ではデータセットを何周なめたか（`epoch`）などの情報がどうように記録されているおり、学習ループを書いていく際に便利です。

データセットオブジェクトからイテレータを作るには、以下のようにします。


```python
from chainer import iterators

batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(
    test, batchsize, repeat=False, shuffle=False)
```

ここでは、学習に用いるデータセット用のイテレータ（`train_iter`）と、検証用のデータセット用のイテレータ（`test_iter`）を2つ作成しています。ここで、`batchsize = 128`としているので、ここで作成した訓練データ用の`Iterator`である`train_iter`およびテストデータ用の`Iterator`である`test_iter`は、それぞれ128枚の数字画像データを一括りにして返す`Iterator`ということになります。[^TrainingデータとValidationデータ]

#### NOTE: `SerialIterator`について

Chainerがいくつか用意している`Iterator`の一種である`SerialIterator`は、データセットの中のデータを順番に取り出してくる最もシンプルな`Iterator`です。コンストラクタの引数にデータセットオブジェクトと、バッチサイズを取ります。このとき、渡したデータセットオブジェクトから、何周も何周もデータを繰り返し読み出す必要がある場合は`repeat`引数を`True`とし、1周が終わったらそれ以上データを取り出したくない場合はこれを`False`とします。これは、主にvalidation用のデータセットに対して使うフラグです。デフォルトでは、`True`になっています。また、`shuffle`引数に`True`を渡すと、データセットから取り出されてくるデータの順番をエポックごとにランダムに変更します。`SerialIterator`の他にも、マルチプロセスで高速にデータを処理できるようにした`MultiprocessIterator`や`MultithreadIterator`など、複数の`Iterator`が用意されています。詳しくは以下を見てください。

- [Chainerで使えるIterator一覧](https://docs.chainer.org/en/latest/reference/iterators.html)

## 3. ネットワークの定義

では、学習させるネットワークを定義してみましょう。今回は、全結合層のみからなる多層パーセプトロンを作ってみます。中間層のユニット数は適当に100とし、今回は10クラス分類をしたいので、出力ユニット数は10とします。今回用いるMNISTデータセットは0〜9までの数字のいずれかを意味する10種のラベルを持つためです。では、ネットワークを定義するために必要な`Link`, `Function`, そして`Chain`について、簡単にここで説明を行います。

### LinkとFunction

Chainerでは、ニューラルネットワークの各層を、`Link`と`Function`に区別します。

- **`Link`は、パラメータを持つ関数です。**
- **`Function`は、パラメータを持たない関数です。**

これらを組み合わせてネットワークを記述します。パラメータを持つ層は、`chainer.links`モジュール以下にたくさん用意されています。パラメータを持たない層は、`chainer.functions`モジュール以下にたくさん用意されています。これらに簡単にアクセスするために、

```
import chainer.links as L
import chainer.functions as F
```

と別名を与えて、`L.Convolution2D(...)`や`F.relu(...)`のように用いる慣習がありますが、特にこれが決まった書き方というわけではありません。

### Chain

`Chain`は、**パラメータを持つ層（`Link`）をまとめておくためのクラス**です。パラメータを持つということは、基本的にネットワークの学習の際にそれらを更新していく必要があるということです（更新されないパラメータを持たせることもできます）。Chainerでは、モデルのパラメータの更新は、`Optimizer`という機能が担います。その際、更新すべき全てのパラメータを簡単に発見できるように、`Chain`で一箇所にまとめておきます。そうすると、`Chain.params()`メソッドを使って**更新されるパラメータ一覧が簡単に取得できます。**

### Chainを継承してネットワークを定義しよう

Chainerでは、ネットワークは`Chain`クラスを継承したクラスとして定義されることが一般的です。その場合、そのクラスのコンストラクタで、`self.init_scope()`で作られる`with`コンテキストを作り、その中でネットワークに登場する`Link`をプロパティとして登録しておきます。こうすると、自動的に`Optimizer`が最適化対象のパラメータを持つ層だな、と捉えてくれます。

もう一つ、一般的なのは、ネットワークの前進計算（データを渡して、出力を返す）を、`__call__`メソッドに書いておくという方法です。こうすると、ネットワーククラスをinstantiateして作ったオブジェクトを、関数のようにして使うことができます（例：`output = net(data)`）。

### GPUで実行するには

`Chain`クラスは`to_gpu`メソッドを持ち、この引数にGPU IDを指定すると、指定したGPU IDのメモリ上にネットワークの全パラメータを転送します。こうしておくと、前進計算も学習の際のパラメータ更新なども全部GPU上で行われるようになります。GPU IDとして-1を使うと、すなわちこれはCPUを意味します。

### 同じ結果を保証したい

ネットワークを書き始める前に、まずは乱数シードを固定して、本記事とほぼ同様の結果が再現できるようにしておきましょう。（より厳密に計算結果の再現性を保証したい場合は、`deterministic`というオプションについて知る必要があります。こちらの記事が役に立ちます：[ChainerでGPUを使うと毎回結果が変わる理由と対策](http://qiita.com/TokyoMickey/items/cc8cd43545f2656b1cbd)。


```python
import random
import numpy
random.seed(0)
numpy.random.seed(0)

import chainer

if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(0)
```

### ネットワークを表すコード

いよいよネットワークを書いてみます！


```python
import chainer
import chainer.links as L
import chainer.functions as F

class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        
        # パラメータを持つ層の登録
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

gpu_id = 0  # CPUを用いる場合は、この値を-1にしてください

net = MLP()

if gpu_id >= 0:
    net.to_gpu(gpu_id)
```

できました！疑問点はありませんか？ちなみに、Chainerにはたくさんの学習可能なレイヤやパラメータを持たないレイヤが用意されています。ぜひ一度以下の一覧のページを見てみましょう。

- [Chainerで使える関数(`Function`)一覧](https://docs.chainer.org/en/stable/reference/functions.html)
- [Chainerで学習できるレイヤ(`Link`)一覧](https://docs.chainer.org/en/stable/reference/links.html)

`Link`一覧には、ニューラルネットワークによく用いられる全結合層や畳み込み層、LSTMなどや、ReLUなどの活性化関数などなどだけでなく、有名なネットワーク全体も`Link`として載っています。ResNetや、VGGなどです。また、`Function`一覧には、画像の大きさをresizeしたり、サイン・コサインのような関数を始め、いろいろなネットワークの要素として使える関数が載っています。

#### NOTE

上のネットワーク定義で、`L.Linear`は全結合層を意味しますが、最初のLinear層は第一引数に`None`が渡されています。これは、実行時に、つまり**データがその層に入力された瞬間、必要な数の入力側ユニット数を自動的に計算する**ということを意味します。ネットワークが最初に計算を行う際に、初めて `(n_input)` $\times$ `n_mid_units` の大きさの行列を作成し、それを学習対象とするパラメータとして保持します。これは後々、畳み込み層を全結合層の前に配置する際などに便利な機能です。

様々な`Link`は、それぞれ学習対象となるパラメータを保持しています。それらの値は、NumPyの配列として簡単に取り出して見ることができます。例えば、上のモデル`MLP`は`l1`という名前の全結合層が登録されています。この全結合相は重み行列`W`とバイアス`b`という2つのパラメータを持ちます。これらには外から以下のようにしてアクセスすることができます：


```python
print('1つ目の全結合相のバイアスパラメータの形は、', net.l1.b.shape)
print('初期化直後のその値は、', net.l1.b.data)
```

    1つ目の全結合相のバイアスパラメータの形は、 (100,)
    初期化直後のその値は、 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]


しかしここで、`net.l1.W`にアクセスしようとすると、以下のようなエラーが出ると思います。

```
AttributeError: 'Linear' object has no attribute 'W'
```

なぜでしょうか？我々は`l1`をネットワークに登録するときに、`L.Linear`の第一引数に`None`を渡しましたね。そして、**まだネットワークに一度もデータを入力していません**。そのため、**まだ重み行列`W`は作成されていません。**そのため、そんなattributeはない、と言われているわけです。

## 4. 最適化手法の選択

では、上で定義したネットワークをMNISTデータセットを使って訓練してみましょう。学習時に用いる最適化の手法としてはいろいろな種類のものが提案されていますが、Chainerは多くの手法を同一のインターフェースで利用できるよう、`Optimizer`という機能でそれらを提供しています。`chainer.optimizers`モジュール以下に色々なものを見つけることができます。一覧はこちらにあります：

- [Chainerで使える最適化手法一覧](https://docs.chainer.org/en/stable/reference/optimizers.html)

ここでは最もシンプルな勾配降下法の手法である`optimizers.SGD`を用います。`Optimizer`のオブジェクトには、`setup`メソッドを使ってモデル（`Chain`オブジェクト）を渡します。こうすることで`Optimizer`に、何を最適化すればいいか把握させることができます。

他にもいろいろな最適化手法が手軽に試せるので、色々と試してみて結果の変化を見てみてください。例えば、下の`chainer.optimizers.SGD`のうち`SGD`の部分を`MomentumSGD`, `RMSprop`,  `Adam`などに変えるだけで、最適化手法の違いがどのような学習曲線（ロスカーブ）の違いを生むかなどを簡単に調べることができます。


```python
from chainer import optimizers

optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(net)
```

#### NOTE

今回はSGDのコンストラクタの`lr`という引数に $0.01$ を与えました。この値は学習率として知られ、モデルをうまく訓練して良いパフォーマンスを発揮させるために調整する必要がある重要な**ハイパーパラメータ**として知られています。

## 5. 学習する

いよいよ学習をスタートします！今回は分類問題なので、`softmax_cross_entropy`というロス関数を使って最小化すべきロスの値を計算します。

まず、ネットワークにデータを渡して、出てきた出力と、入力データに対応する正解ラベルを、`Function`の一種でありスカラ値を返す**ロス関数**に渡し、ロス（最小化したい値）の計算を行います。ロスは、`chainer.Variable`のオブジェクトになっています。そして、この`Variable`は、**今まで自分にどんな計算が施されたかを辿れるようになっています。**この仕組みが、Define-by-Run [Tokui 2015]とよばれる発明の実装における中心的な役割を果たしています。

ここでは誤差逆伝播法自体の説明は割愛しますが、**計算したロスに対する勾配をネットワークに逆向きに流していく**処理は、Chainerではネットワークが吐き出した`Variable`が持つ`backward()`メソッドを呼ぶだけでできます。これを呼ぶと、前述のようにこれまでの計算過程を逆向きに遡って**誤差逆伝播用の計算グラフを構築し**、途中のパラメータの勾配を連鎖率を使って計算してくれます。

こうして計算された各パラメータに対する勾配を使って、先程`Optimizer`を作成する際に指定したアルゴリズムを使ってネットワークパラメータの更新（＝学習）が行われるわけです。

まとめると、今回1回の更新処理の中で行うのは、以下の4項目です。

1. ネットワークにデータを渡して出力`y`を得る
2. 出力`y`と正解ラベル`t`を使って、最小化すべきロスの値を`softmax_cross_entropy`関数で計算する
3. `softmax_cross_entropy`関数の出力（`Variable`）の`backward()`メソッドを呼んで、ネットワークの全てのパラメータの勾配を誤差逆伝播法で計算する
4. Optimizerの`update`メソッドを呼び、3.で計算した勾配を使って全パラメータを更新する

パラメータの更新は、何度も何度も繰り返し行います。一度の更新に用いられるデータは、ネットワークに入力されたバッチサイズ分だけ束ねられたデータのみです。そのため、データセット全体のデータを使うために、次のミニバッチを入力して再度更新、その次のミニバッチを使ってまた更新、ということを繰り返すわけです。そのため、この過程を学習ループと呼んでいます。

#### NOTE: ロス関数

ちなみに、ロス関数は、例えば分類問題ではなく簡単な回帰問題を解きたいような場合、`F.softmax_cross_entropy`の代わりに`F.mean_squared_error`などを用いることもできます。他にも、いろいろな問題設定に対応するために様々なロス関数がChainerには用意されています。こちらからその一覧を見ることができます：

- [Chainerで使えるロス関数一覧](http://docs.chainer.org/en/latest/reference/functions.html#loss-functions)

### 学習ループのコード


```python
import numpy as np
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu

max_epoch = 10

while train_iter.epoch < max_epoch:
    
    # ---------- 学習の1イテレーション ----------
    train_batch = train_iter.next()
    x, t = concat_examples(train_batch, gpu_id)
    
    # 予測値の計算
    y = net(x)

    # ロスの計算
    loss = F.softmax_cross_entropy(y, t)

    # 勾配の計算
    net.cleargrads()
    loss.backward()

    # パラメータの更新
    optimizer.update()
    # --------------- ここまで ----------------

    # 1エポック終了ごとにValidationデータに対する予測精度を測って、
    # モデルの汎化性能が向上していることをチェックしよう
    if train_iter.is_new_epoch:  # 1 epochが終わったら

        # ロスの表示
        print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')

        test_losses = []
        test_accuracies = []
        while True:
            test_batch = test_iter.next()
            x_test, t_test = concat_examples(test_batch, gpu_id)

            # テストデータをforward
            y_test = net(x_test)

            # ロスを計算
            loss_test = F.softmax_cross_entropy(y_test, t_test)
            test_losses.append(to_cpu(loss_test.data))

            # 精度を計算
            accuracy = F.accuracy(y_test, t_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.data)
            
            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(test_losses), np.mean(test_accuracies)))
```

    epoch:01 train_loss:0.8150 val_loss:0.7960 val_accuracy:0.8360
    epoch:02 train_loss:0.4188 val_loss:0.4618 val_accuracy:0.8826
    epoch:03 train_loss:0.3282 val_loss:0.3753 val_accuracy:0.8988
    epoch:04 train_loss:0.3030 val_loss:0.3381 val_accuracy:0.9051
    epoch:05 train_loss:0.2858 val_loss:0.3121 val_accuracy:0.9123
    epoch:06 train_loss:0.2277 val_loss:0.2966 val_accuracy:0.9166
    epoch:07 train_loss:0.3285 val_loss:0.2830 val_accuracy:0.9203
    epoch:08 train_loss:0.3711 val_loss:0.2710 val_accuracy:0.9223
    epoch:09 train_loss:0.1856 val_loss:0.2624 val_accuracy:0.9249
    epoch:10 train_loss:0.1841 val_loss:0.2528 val_accuracy:0.9273


`val_accuracy`に着目してみると、最終的におおよそ93%程度の精度で手書きの数字が分類できるようになりました。

## 6. 学習済みモデルを保存する

学習が終わったら、その結果を保存します。Chainerには、2種類のフォーマットで学習済みネットワークをシリアライズする機能が用意されています。一つはHDF5形式で、もう一つはNumPyのNPZ形式でネットワークを保存するものです。今回は、追加ライブラリのインストールが必要なHDF5ではなく、NumPy標準機能で提供されているシリアライズ機能（`numpy.savez()`）を利用したNPZ形式でのモデルの保存を行います。


```python
from chainer import serializers

serializers.save_npz('my_mnist.model', net)
```


```python
# ちゃんと保存されていることを確認
%ls -la my_mnist.model
```

    -rw-rw-r-- 1 shunta shunta 333876 Dec  4 23:30 my_mnist.model


## 7. 保存したモデルを読み込んで推論する

学習したネットワークを、それを使って数字の分類がしたい誰かに渡して、使ってもらうにはどうしたら良いでしょうか。もっともシンプルな方法は、ネットワークの定義がかかれたPythonファイルと、今しがた保存したNPZファイルを渡して、以下のように使うことです。以下のコードの前に、渡したネットワーク定義のファイルからネットワークのクラス（ここでは`MLP`）が読み込まれていることを前提とします。


```python
# まず同じネットワークのオブジェクトを作る
infer_net = MLP()

# そのオブジェクトに保存済みパラメータをロードする
serializers.load_npz('my_mnist.model', infer_net)
```

以上で準備が整いました。それでは、試しにテストデータの中から一つ目の画像を取ってきて、それに対する分類を行ってみましょう。


```python
gpu_id = 0  # CPUで計算をしたい場合は、-1を指定してください

if gpu_id >= 0:
    infer_net.to_gpu(gpu_id)

# 1つ目のテストデータを取り出します
x, t = test[0]  #  tは使わない

# どんな画像か表示してみます
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()

# ミニバッチの形にする（複数の画像をまとめて推論に使いたい場合は、サイズnのミニバッチにしてまとめればよい）
print('元の形：', x.shape, end=' -> ')

x = x[None, ...]

print('ミニバッチの形にしたあと：', x.shape)

# ネットワークと同じデバイス上にデータを送る
x = infer_net.xp.asarray(x)

# モデルのforward関数に渡す
y = infer_net(x)

# Variable形式で出てくるので中身を取り出す
y = y.array

# 結果をCPUに送る
y = to_cpu(y)

# 予測確率の最大値のインデックスを見る
pred_label = y.argmax(axis=1)

print('ネットワークの予測:', pred_label[0])
```


![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_30_0.png)


    元の形： (784,) -> ミニバッチの形にしたあと： (1, 784)
    ネットワークの予測: 7


ネットワークの予測は7でした。画像を見る限り、当たっていそうですね！

# Trainerを使ってみよう

Chainerは、これまで書いてきたような学習ループを隠蔽する`Trainer`という機能を提供しています。これを使うと、学習ループを陽に書く必要がなくなり、またいろいろな便利なExtentionを使うことで、学習過程でのロスカーブの可視化や、ログの保存などが楽になります。

## 1. データセット・Iterator・ネットワークの準備

これらはループを自分で書く場合と同じなので、まとめてしまいます。


```python
random.seed(0)
numpy.random.seed(0)
if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(0)

train, test = mnist.get_mnist()

batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, False, False)

gpu_id = 0  # CPUを用いたい場合は、-1を指定してください

net = MLP()

if gpu_id >= 0:
    net.to_gpu(gpu_id)
```

## 2. Updaterの準備

ここからが学習ループを自分で書く場合と異なる部分です。ループを自分で書く場合には、データセットからバッチサイズ分のデータをとってきてミニバッチに束ねて、それをネットワークに入力して予測を作り、それを正解と比較し、ロスを計算してバックワード（誤差逆伝播）をして、`Optimizer`によってパラメータを更新する、というところまでを、以下のように書いていました。

```python
# ---------- 学習の1イテレーション ----------
train_batch = train_iter.next()
x, t = concat_examples(train_batch, gpu_id)

# 予測値の計算
y = net(x)

# ロスの計算
loss = F.softmax_cross_entropy(y, t)

# 勾配の計算
net.cleargrads()
loss.backward()

# パラメータの更新
optimizer.update()
```

これらの処理を、まるっと`Updater`はまとめてくれます。これを行うために、**`Updater`には`Iterator`と`Optimizer`を渡してやります。** `Iterator`はデータセットオブジェクトを持っていて、そこからミニバッチを作り、`Optimizer`は最適化対象のネットワークを持っていて、それを使って前進計算とロスの計算・パラメータのアップデートをすることができます。そのため、この2つを渡しておけば、上記の処理を`Updater`内で全部行ってもらえるというわけです。では、`Updater`オブジェクトを作成してみましょう。


```python
from chainer import training

gpu_id = 0  # CPUを使いたい場合は-1を指定してください

# ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
net = L.Classifier(net)

# 最適化手法の選択
optimizer = optimizers.SGD()
optimizer.setup(net)

# UpdaterにIteratorとOptimizerを渡す
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
```

#### NOTE

ここでは、ネットワークを`L.Classifier`で包んでいます。`L.Classifier`は一種の`Chain`になっていて、渡されたネットワーク自体を`predictor`というattributeに持ち、**ロス計算を行う機能を追加してくれます。**こうすると、`net()`はデータ`x`だけでなくラベル`t`も取るようになり、まず渡されたデータを`predictor`に通して予測を作り、それを`t`と比較して**ロスの`Variable`を返すようになります。**ロス関数として何を用いるかはデフォルトでは`F.softmax_cross_entropy`となっていますが、`L.Classifier`の引数`lossfunc`にロス計算を行う関数を渡してやれば変更することができるため、Classifierという名前ながら回帰問題などのロス計算機能の追加にも使うことができます。（`L.Classifier(net, lossfun=L.mean_squared_error, compute_accuracy=False)`のようにする）

`StandardUpdater`は前述のような`Updater`の担当する処理を遂行するための最もシンプルなクラスです。この他にも複数のGPUを用いるための`ParallelUpdater`などが用意されています。

## 3. Trainerの準備

実際に学習ループ部分を隠蔽しているのはUpdaterなので、これがあればもう学習を始められそうですが、TrainerはさらにUpdaterを受け取って学習全体の管理を行う機能を提供しています。例えば、**データセットを何周したら学習を終了するか(stop_trigger)** や、**途中のロスの値をどのファイルに保存したいか**、**ロスカーブを可視化した画像ファイルを保存するかどうか**など、学習全体の設定として必須・もしくはあると便利な色々な機能を提供しています。必須なものとしては学習終了のタイミングを指定する`stop_trigger`がありますが、これはTrainerオブジェクトを作成するときのコンストラクタで指定します。指定の方法は単純で、`(長さ, 単位)`という形のタプルを与えます。「長さ」には数字を、「単位」には`'iteration'`もしくは`'epoch'`のいずれかの文字列を指定します。こうすると、たとえば100 epoch（データセット100周）で学習を終了してください、とか、1000 iteration（1000回更新）で学習を終了してください、といったことが指定できます。Trainerを作るときに、`stop_trigger`を指定しないと、学習は自動的には止まりません。

では、実際にTrainerオブジェクトを作ってみましょう。


```python
max_epoch = 10

# TrainerにUpdaterを渡す
trainer = training.Trainer(
    updater, (max_epoch, 'epoch'), out='mnist_result')
```

`out`引数では、この次に説明する`Extension`を使って、ログファイルやロスの変化の過程を描画したグラフの画像ファイルなどを保存するディレクトリを指定しています。

Trainerと、その内側にあるいろいろなオブジェクトの関係は、図にまとめると以下のようになっています。このイメージを持っておくと自分で部分的に改造したりする際に便利だと思います。

![image](https://qiita-image-store.s3.amazonaws.com/0/17934/a751df31-b999-f692-d839-488c26b1c48a.png)

## 4. TrainerにExtensionを追加する

`Trainer`を使う利点として、

- ログを自動的にファイルに保存（`LogReport`)
- ターミナルに定期的にロスなどの情報を表示（`PrintReport`）
- ロスを定期的にグラフで可視化して画像として保存（`PlotReport`)
- 定期的にモデルやOptimizerの状態を自動シリアライズ（`snapshot`）
- 学習の進捗を示すプログレスバーを表示（`ProgressBar`）
- ネットワークの構造をGraphvizのdot形式で保存（`dump_graph`）
- ネットワークのパラメータの平均や分散などの統計情報を出力（`ParameterStatistics`）

などなどの様々な便利な機能を簡単に利用することができる点があります。これらの機能を利用するには、`Trainer`オブジェクトに対して`extend`メソッドを使って追加したい`Extension`のオブジェクトを渡してやるだけです。では実際に幾つかの`Extension`を追加してみましょう。


```python
from chainer.training import extensions

trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, net, device=gpu_id), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.ParameterStatistics(net.predictor.l1, {'std': np.std}))
trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))
```

### `LogReport`

`epoch`や`iteration`ごとの`loss`, `accuracy`などを自動的に集計し、`Trainer`の`out`引数で指定した出力ディレクトリに`log`というファイル名で保存します。

### `snapshot`

`Trainer`の`out`引数で指定した出力ディレクトリに`Trainer`オブジェクトを指定されたタイミング（デフォルトでは1エポックごと）に保存します。`Trainer`オブジェクトは上述のように`Updater`を持っており、この中に`Optimizer`とモデルが保持されているため、この`Extension`でスナップショットをとっておけば、学習の復帰や学習済みモデルを使った推論などが学習終了後にも可能になります。

### `dump_graph`

指定された`Variable`オブジェクトから辿れる計算グラフをGraphvizのdot形式で保存します。保存先は`Trainer`の`out`引数で指定した出力ディレクトリです。

### `Evaluator`

評価用のデータセットの`Iterator`と、学習に使うモデルのオブジェクトを渡しておくことで、学習中のモデルを指定されたタイミングで評価用データセットを用いて評価します。

### `PrintReport`

`Reporter`によって集計された値を標準出力に出力します。このときどの値を出力するかを、リストの形で与えます。

### `PlotReport`

引数のリストで指定された値の変遷を`matplotlib`ライブラリを使ってグラフに描画し、出力ディレクトリに`file_name`引数で指定されたファイル名で画像として保存します。

### `ParameterStatistics`

指定したレイヤ（Link）が持つパラメータの平均・分散・最小値・最大値などなどの統計情報を計算して、ログに保存します。パラメータが発散していないかなどをチェックするのに便利です。

---

これらの`Extension`は、ここで紹介した以外にも、例えば`trigger`によって個別に作動するタイミングを指定できるなどのいくつかのオプションを持っており、より柔軟に組み合わせることができます。詳しくは公式のドキュメントを見てください

- [ChainerのTrainer extension一覧](http://docs.chainer.org/en/latest/reference/extensions.html)

## 5. 学習を開始する

学習を開始するには、`Trainer`オブジェクトのメソッド`run`を呼ぶだけです！


```python
trainer.run()
```

    epoch       main/loss   main/accuracy  val/main/loss  val/main/accuracy  l1/W/data/std  elapsed_time
    [J1           1.59882     0.606693       0.7999         0.826246           0.035967       2.69623       
    [J2           0.601374    0.851196       0.458944       0.875396           0.036728       6.06142       
    [J3           0.427647    0.883612       0.371194       0.896361           0.037161       9.0557        
    [J4           0.368752    0.896384       0.334384       0.904767           0.0374263      12.3297       
    [J5           0.336899    0.904718       0.30967        0.912579           0.0376187      15.1991       
    [J6           0.315892    0.910048       0.294039       0.917425           0.0377735      18.5722       
    [J7           0.299324    0.913996       0.28063        0.921578           0.0379096      21.8253       
    [J8           0.285966    0.918336       0.269207       0.924941           0.0380317      24.9763       
    [J9           0.274382    0.921359       0.261144       0.927314           0.0381456      28.0358       
    [J10          0.264148    0.924624       0.249965       0.929193           0.0382543      31.1956       


初めに取り組んだ学習ループを自分で書いた場合よりもより短いコードで、リッチなログ情報とともに、下記で表示してみるようなグラフなども作りつつ、同様の結果を得ることができました。1層目の全結合層の重み行列の値の標準偏差が、学習の進行とともに大きくなっていっているのも見て取れて、面白いですね。

では、保存されているロスのグラフを確認してみましょう。


```python
from IPython.display import Image
Image(filename='mnist_result/loss.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_47_0.png)



精度のグラフも見てみましょう。


```python
Image(filename='mnist_result/accuracy.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_49_0.png)



もう少し学習を続ければ、まだ多少精度の向上が図れそうな雰囲気がありますね。

ついでに、`dump_graph`という`Extension`が出力した計算グラフを、`Graphviz`を使って画像化して見てみましょう。


```bash
%%bash
dot -Tpng mnist_result/cg.dot -o mnist_result/cg.png
```


```python
Image(filename='mnist_result/cg.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_52_0.png)



上から下へ向かって、データやパラメータがどのような`Function`に渡されて計算が行われ、ロスを表す`Variable`が出力されたかが分かります。

## 6. 学習済みモデルで推論する

それでは、Trainer Extensionのsnapshotが自動的に保存したネットワークのスナップショットから学習済みパラメータを読み込んで、学習ループを書いて学習したときと同様に1番目のテストデータで推論を行ってみましょう。

ここで注意すべきは、snapshotが保存するnpzファイルはTrainer全体のスナップショットであるため、extensionの内部のパラメータなども一緒に保存されています。これは、学習自体を再開するために必要だからです。しかし、今回はネットワークのパラメータだけを読み込めば良いので、`serializers.load_npz()`のpath引数にネットワーク部分までのパス（`updater/model:main/predictor/`）を指定しています。こうすることで、ネットワークのオブジェクトにパラメータだけを読み込むことができます。


```python
infer_net = MLP()
serializers.load_npz(
    'mnist_result/snapshot_epoch-10',
    infer_net, path='updater/model:main/predictor/')

if gpu_id >= 0:
    infer_net.to_gpu(gpu_id)

x, t = test[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.show()

x = infer_net.xp.asarray(x[None, ...])
y = infer_net(x)
y = to_cpu(y.array)

print('予測ラベル:', y.argmax(axis=1)[0])
```


![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_55_0.png)


    予測ラベル: 7


無事正解できていますね。

# 新しいネットワークを書いてみよう

ここでは、MNISTデータセットではなくCIFAR10という32x32サイズの小さなカラー画像に10クラスのいずれかのラベルがついたデータセットを用いて、いろいろなモデルを自分で書いて試行錯誤する流れを体験してみます。

| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|:--------:|:----------:|:----:|:---:|:----:|:---:|:----:|:-----:|:----:|:-----:|
| ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship4.png) | ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck4.png) |

## 1. ネットワークの定義

ここでは、さきほど試した全結合層だけからなるネットワークではなく、畳込み層を持つネットワークを定義してみます。3つの畳み込み層を持ち、2つの全結合層がそのあとに続いています。


```python
class MyNet(chainer.Chain):
    
    def __init__(self, n_out):
        super(MyNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, 3, 1)
            self.conv2 = L.Convolution2D(32, 64, 3, 3, 1)
            self.conv3 = L.Convolution2D(64, 128, 3, 3, 1)
            self.fc4 = L.Linear(None, 1000)
            self.fc5 = L.Linear(1000, n_out)
        
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        return h
```

## 2. 学習

ここで、あとから別のネットワークも簡単に同じ設定で訓練できるよう、`train`関数を作っておきます。これは、

- ネットワークのオブジェクト
- バッチサイズ
- 使用するGPU ID
- 学習を終了するエポック数
- データセットオブジェクト
- 学習率の初期値
- 学習率減衰のタイミング

などを渡すと、内部で`Trainer`を用いて渡されたデータセットを使ってネットワークを訓練し、学習が終了した状態のネットワークを返してくれる関数です。先程のMNISTでの例と違い、最適化手法にはMomentumSGDを用い、ExponentialShiftというExtentionを使って、指定したタイミングごとに学習率を減衰させるようにしてみます。

この`train`関数を用いて、上で定義した`MyModel`モデルを訓練してみます。


```python
from chainer.datasets import cifar


def train(network_object, batchsize=128, gpu_id=0, max_epoch=20, train_dataset=None, test_dataset=None, postfix='', base_lr=0.01, lr_decay=None):

    # 1. Dataset
    if train_dataset is None and test_dataset is None:
        train, test = cifar.get_cifar10()
    else:
        train, test = train_dataset, test_dataset

    # 2. Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)

    # 3. Model
    net = L.Classifier(network_object)

    # 4. Optimizer
    optimizer = optimizers.MomentumSGD(lr=base_lr)
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # 5. Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # 6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_cifar10_{}result'.format(network_object.__class__.__name__, postfix))
    
    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(test_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    if lr_decay is not None:
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_decay)
    trainer.run()
    del trainer
    
    return net
```


```python
net = train(MyNet(10), gpu_id=0)
```

    epoch       main/loss   main/accuracy  val/main/loss  val/main/accuracy  elapsed_time  lr        
    [J1           1.96016     0.29208        1.69692        0.39824            11.2051       0.01        
    [J2           1.61492     0.423593       1.51012        0.462025           21.3019       0.01        
    [J3           1.47277     0.473638       1.41357        0.49288            30.4257       0.01        
    [J4           1.39049     0.502278       1.34537        0.521262           40.1109       0.01        
    [J5           1.31663     0.528213       1.30922        0.535502           49.9053       0.01        
    [J6           1.25776     0.553005       1.27749        0.544996           59.1626       0.01        
    [J7           1.2042      0.569953       1.23783        0.560918           69.1643       0.01        
    [J8           1.15561     0.589163       1.19613        0.571697           78.5205       0.01        
    [J9           1.11053     0.606118       1.20992        0.570708           88.397        0.01        
    [J10          1.06659     0.622782       1.2239         0.560324           98.1943       0.01        
    [J11          1.02185     0.6372         1.17653        0.580103           108.195       0.01        
    [J12          0.984571    0.650076       1.15792        0.59108            117.212       0.01        
    [J13          0.939215    0.667679       1.1416         0.60265            126.884       0.01        
    [J14          0.894961    0.684255       1.16477        0.594838           136.712       0.01        
    [J15          0.85514     0.698669       1.14386        0.600475           146.424       0.01        
    [J16          0.813399    0.71252        1.16846        0.601365           155.925       0.01        
    [J17          0.772127    0.72868        1.17633        0.597903           166.023       0.01        
    [J18          0.723599    0.746623       1.18344        0.600376           175.892       0.01        
    [J19          0.675444    0.761659       1.24003        0.600475           185.295       0.01        
    [J20          0.626475    0.780271       1.21516        0.603145           195.041       0.01        


学習が20エポックまで終わりました。ロスと精度のプロットを見てみましょう。


```python
Image(filename='MyNet_cifar10_result/loss.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_63_0.png)




```python
Image(filename='MyNet_cifar10_result/accuracy.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_64_0.png)



学習データでの精度（`main/accuracy`)は87%程度まで到達していますが、テストデータでのロス（`val/main/loss`）はむしろIterationを進むごとに大きくなってしまっており、またテストデータでの精度（`val/main/accuracy`）も60%前後で頭打ちになってしまっています。学習データでは良い精度が出ているが、テストデータでは精度が良くないということなので、**モデルが学習データにオーバーフィッティングしている**と思われます。

## 3. 学習済みネットワークを使った予測

テスト精度は62%程度でしたが、試しにこの学習済みネットワークを使っていくつかのテスト画像を分類させてみましょう。あとで使いまわせるように`predict`関数を作っておきます。


```python
cls_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']

def predict(net, image_id):
    _, test = cifar.get_cifar10()
    x, t = test[image_id]
    net.to_cpu()
    y = net.predictor(x[None, ...]).data.argmax(axis=1)[0]
    print('predicted_label:', cls_names[y])
    print('answer:', cls_names[t])

    plt.imshow(x.transpose(1, 2, 0))
    plt.show()

for i in range(10, 15):
    predict(net, i)
```

    predicted_label: airplane
    answer: airplane



![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_67_1.png)


    predicted_label: automobile
    answer: truck



![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_67_3.png)


    predicted_label: dog
    answer: dog



![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_67_5.png)


    predicted_label: horse
    answer: horse



![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_67_7.png)


    predicted_label: truck
    answer: truck



![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_67_9.png)


うまく分類できているものもあれば、そうでないものもありました。ネットワークの学習に使用したデータセット上ではほぼ百発百中で正解できるとしても、未知のデータ、すなわちテストデータセットにある画像に対して高精度な予測ができなければ、意味がありません[^NN]。テストデータでの精度は、モデルの**汎化性能**に関係していると言われます。

どうすれば高い汎化性能を持つネットワークを設計し、学習することができるでしょうか？（そんなことが簡単に分かったら苦労しない。）

## 4. もっと深いネットワークを定義してみよう

では、上のネットワークよりもよりたくさんの層を持つネットワークを定義してみましょう。ここでは、1層の畳み込みネットワークを`ConvBlock`、1層の全結合ネットワークを`LinearBlock`として定義し、これをたくさんシーケンシャルに積み重ねる方法で大きなネットワークを定義してみます。

### 構成要素を定義する

まず、今目指している大きなネットワークの構成要素となる`ConvBlock`と`LinearBlock`を定義してみましょう。


```python
class ConvBlock(chainer.Chain):
    
    def __init__(self, n_ch, pool_drop=False):
        w = chainer.initializers.HeNormal()
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, n_ch, 3, 1, 1, nobias=True, initialW=w)
            self.bn = L.BatchNormalization(n_ch)
        self.pool_drop = pool_drop
        
    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        if self.pool_drop:
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)
        return h
    
class LinearBlock(chainer.Chain):
    
    def __init__(self, drop=False):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, 1024, initialW=w)
        self.drop = drop
        
    def __call__(self, x):
        h = F.relu(self.fc(x))
        if self.drop:
            h = F.dropout(h)
        return h
```

`ConvBlock`は`Chain`を継承した小さなネットワークとして定義されています。これは一つの畳み込み層とBatch Normalization層をパラメータありで持っているので、コンストラクタ内でこれらの登録を行っています。`__call__`メソッドでは、これらにデータを渡しつつ、活性化関数ReLUを適用して、さらに`pool_drop`がコンストラクタに`True`で渡されているときはMax PoolingとDropoutという関数を適用するようになっています。

Chainerでは、Pythonを使って書いたforward計算のコード自体がネットワークの構造を表します。すなわち、実行時にデータがどのような層をくぐっていったか、ということがネットワークそのものを定義します。これによって、上記のような分岐などを含むネットワークも簡単に書け、柔軟かつシンプルで可読性の高いネットワーク定義が可能になります。これが**Define-by-Run**と呼ばれる特徴です。

### 大きなネットワークの定義

次に、これらの小さなネットワークを構成要素として積み重ねて、大きなネットワークを定義してみましょう。


```python
class DeepCNN(chainer.ChainList):

    def __init__(self, n_output):
        super(DeepCNN, self).__init__(
            ConvBlock(64),
            ConvBlock(64, True),
            ConvBlock(128),
            ConvBlock(128, True),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256, True),
            LinearBlock(),
            LinearBlock(),
            L.Linear(None, n_output)
        )
    
    def __call__(self, x):
        for f in self:
            x = f(x)
        return x
```

ここで利用しているのが、`ChainList`というクラスです。このクラスは`Chain`を継承したクラスで、いくつもの`Link`や`Chain`を順次呼び出していくようなネットワークを定義するときに便利です。`ChainList`を継承して定義されるモデルは、親クラスのコンストラクタを呼び出す際に**キーワード引数ではなく普通の引数として**`Link`もしくは`Chain`オブジェクトを渡すことができます。そしてこれらは、**self.children()**メソッドによって**登録した順番に**取り出すことができます。

この特徴を使うと、forward計算の記述が簡単になります。**self.children()**が返す構成要素のリストから、for文で構成要素を順番に取り出していき、そもそもの入力である`x`に取り出してきた部分ネットワークの計算を適用して、この出力で`x`を置き換えるということを順番に行っていけば、一連の`Link`または`Chain`を、コンストラクタで親クラスに登録した順番と同じ順番で適用していくことができます。そのため、シーケンシャルな部分ネットワークの適用によって表される大きなネットワークを定義するのに重宝します。

それでは、学習を回してみます。今回はパラメータ数も多いので、学習を停止するエポック数を100に設定します。また、学習率を0.1から始めて、30エポックごとに10分の1にするように設定してみます。


```python
model = train(DeepCNN(10), max_epoch=100, base_lr=0.1, lr_decay=(30, 'epoch'))
```

    epoch       main/loss   main/accuracy  val/main/loss  val/main/accuracy  elapsed_time  lr        
    [J1           2.5835      0.167          1.99917        0.24377            21.7501       0.1         
    [J2           1.90847     0.269701       1.801          0.329411           41.9713       0.1         
    [J3           1.70557     0.350501       1.91915        0.307951           61.5891       0.1         
    [J4           1.54373     0.423913       1.68159        0.390131           81.445        0.1         
    [J5           1.38066     0.490689       1.48208        0.481408           101.239       0.1         
    [J6           1.19877     0.564183       1.13544        0.587816           121.357       0.1         
    [J7           1.07871     0.615769       0.974936       0.652789           141.293       0.1         
    [J8           0.959763    0.662941       1.13652        0.59731            162.255       0.1         
    [J9           0.879238    0.694254       0.84043        0.70896            182.452       0.1         
    [J10          0.827972    0.715233       1.05347        0.65002            203.096       0.1         
    [J11          0.762531    0.739944       0.835312       0.716278           223.883       0.1         
    [J12          0.733404    0.752038       0.794253       0.732595           244.697       0.1         
    [J13          0.686053    0.765925       0.744322       0.751286           264.843       0.1         
    [J14          0.657708    0.778626       0.677878       0.770669           284.672       0.1         
    [J15          0.631524    0.786645       0.914332       0.714003           304.889       0.1         
    [J16          0.61163     0.796214       0.71221        0.769778           325.536       0.1         
    [J17          0.595124    0.80163        0.793751       0.750396           346.612       0.1         
    [J18          0.569035    0.807545       1.01925        0.696005           368.311       0.1         
    [J19          0.558102    0.812921       1.74147        0.5714             389.245       0.1         
    [J20          0.549798    0.815217       1.18176        0.625692           410.47        0.1         
    [J21          0.539967    0.819793       0.606938       0.799446           431.402       0.1         
    [J22          0.524822    0.823598       0.689638       0.785206           452.378       0.1         
    [J23          0.523005    0.825707       0.732951       0.759889           474.294       0.1         
    [J24          0.501015    0.832833       0.728569       0.765823           494.66        0.1         
    [J25          0.49639     0.832681       0.685372       0.778283           515.899       0.1         
    [J26          0.481854    0.838535       1.17472        0.63212            536.841       0.1         
    [J27          0.487149    0.839924       0.596042       0.80271            558.435       0.1         
    [J28          0.468286    0.844849       0.580874       0.809335           579.213       0.1         
    [J29          0.461486    0.845868       0.885065       0.743572           599.586       0.1         
    [J30          0.461226    0.846635       0.804733       0.750396           620.015       0.1         
    [J31          0.289762    0.901914       0.365872       0.883604           640.39        0.01        
    [J32          0.206296    0.929046       0.370742       0.885285           660.644       0.01        
    [J33          0.179258    0.938039       0.353137       0.890131           681.459       0.01        
    [J34          0.155603    0.946072       0.358469       0.893987           702.702       0.01        
    [J35          0.14138     0.951382       0.354104       0.895273           723.44        0.01        
    [J36          0.13085     0.953764       0.36621        0.891713           744.74        0.01        
    [J37          0.116256    0.95872        0.362551       0.89468            765.024       0.01        
    [J38          0.106264    0.963642       0.401992       0.887164           785.83        0.01        
    [J39          0.0969507   0.966612       0.393622       0.894976           805.345       0.01        
    [J40          0.0901741   0.96849        0.395345       0.896064           823.356       0.01        
    [J41          0.0843399   0.970888       0.428838       0.890427           841.812       0.01        
    [J42          0.0811325   0.971567       0.412273       0.891812           861.034       0.01        
    [J43          0.0750897   0.973317       0.428453       0.888252           880.424       0.01        
    [J44          0.0715538   0.974225       0.435713       0.891812           901.881       0.01        
    [J45          0.0743785   0.973865       0.47753        0.881527           922.951       0.01        
    [J46          0.0700931   0.975621       0.461151       0.885087           944.207       0.01        
    [J47          0.0666815   0.976403       0.433557       0.889537           965.445       0.01        
    [J48          0.0638143   0.978145       0.434256       0.887955           985.621       0.01        
    [J49          0.0602312   0.97936        0.434696       0.889241           1006.84       0.01        
    [J50          0.0615239   0.978401       0.435663       0.893196           1027.14       0.01        
    [J51          0.0611749   0.979387       0.436939       0.889735           1048.3        0.01        
    [J52          0.0635055   0.978021       0.453827       0.884395           1069.08       0.01        
    [J53          0.0548479   0.981797       0.477223       0.887065           1089.35       0.01        
    [J54          0.0581961   0.980188       0.467385       0.884494           1110.04       0.01        
    [J55          0.0577169   0.980838       0.442252       0.887065           1131.06       0.01        
    [J56          0.0537893   0.981571       0.459286       0.886373           1151.61       0.01        
    [J57          0.0544636   0.981658       0.469634       0.886472           1172.86       0.01        
    [J58          0.056       0.980718       0.493179       0.880439           1193.33       0.01        
    [J59          0.0555124   0.980709       0.466303       0.886373           1213.74       0.01        
    [J60          0.058206    0.979699       0.499881       0.880142           1233.25       0.01        
    [J61          0.0351765   0.988631       0.399609       0.899328           1255.08       0.001       
    [J62          0.019923    0.994411       0.398297       0.902294           1275.65       0.001       
    [J63          0.0176245   0.994765       0.4026         0.902591           1296.4        0.001       
    [J64          0.0135215   0.996314       0.408149       0.903679           1316.43       0.001       
    [J65          0.0140328   0.995864       0.418805       0.903877           1335.65       0.001       
    [J66          0.0122992   0.996443       0.414804       0.904173           1356.38       0.001       
    [J67          0.0111444   0.996554       0.416321       0.904272           1378.04       0.001       
    [J68          0.0097971   0.997203       0.423815       0.903975           1397.77       0.001       
    [J69          0.00889979  0.997522       0.422069       0.906052           1416.84       0.001       
    [J70          0.00966005  0.997155       0.429041       0.904767           1436.99       0.001       
    [J71          0.00918675  0.997383       0.427221       0.90625            1457.23       0.001       
    [J72          0.00797029  0.997857       0.427777       0.906349           1476.48       0.001       
    [J73          0.00785823  0.997962       0.436813       0.905953           1496.4        0.001       
    [J74          0.00718227  0.997922       0.438096       0.906646           1516.5        0.001       
    [J75          0.0072844   0.998137       0.436073       0.905063           1536.99       0.001       
    [J76          0.00734126  0.997942       0.442533       0.906151           1557.52       0.001       
    [J77          0.00720541  0.998062       0.441812       0.904173           1578.02       0.001       
    [J78          0.00642986  0.998117       0.440529       0.904866           1596.88       0.001       
    [J79          0.00640991  0.998561       0.44013        0.905558           1617.16       0.001       
    [J80          0.00633529  0.998538       0.441952       0.905459           1637.8        0.001       
    [J81          0.00614588  0.998421       0.441848       0.907733           1657.59       0.001       
    [J82          0.00647109  0.998421       0.445399       0.906448           1677.41       0.001       
    [J83          0.0062315   0.998297       0.442615       0.906349           1696.9        0.001       
    [J84          0.00568081  0.998781       0.44344        0.904964           1716.51       0.001       
    [J85          0.00560865  0.998521       0.442075       0.906151           1734.59       0.001       
    [J86          0.00497888  0.998838       0.446387       0.905854           1752.64       0.001       
    [J87          0.00574452  0.998302       0.452482       0.904173           1770.8        0.001       
    [J88          0.00530873  0.998698       0.448742       0.906646           1788.89       0.001       
    [J89          0.00538916  0.998641       0.453755       0.906052           1806.57       0.001       
    [J90          0.00512664  0.998741       0.446427       0.906843           1825.06       0.001       
    [J91          0.00496505  0.998458       0.448768       0.906646           1843.66       0.0001      
    [J92          0.00504746  0.998841       0.448177       0.906646           1862.45       0.0001      
    [J93          0.00499313  0.998681       0.446119       0.907239           1879.89       0.0001      
    [J94          0.00450806  0.998858       0.451226       0.90625            1898.34       0.0001      
    [J95          0.00507836  0.998601       0.44974        0.90625            1916.19       0.0001      
    [J96          0.00484914  0.998598       0.447423       0.906448           1933.88       0.0001      
    [J97          0.00455869  0.999061       0.450153       0.906052           1952.23       0.0001      
    [J98          0.00513725  0.998641       0.446296       0.906547           1970.09       0.0001      
    [J99          0.00478506  0.998658       0.447933       0.905854           1988.7        0.0001      
    [J100         0.0043492   0.998961       0.446102       0.90625            2007.43       0.0001      


学習が終了しました。ロスカーブと精度のグラフを見てみましょう。


```python
Image(filename='DeepCNN_cifar10_result/loss.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_77_0.png)




```python
Image(filename='DeepCNN_cifar10_result/accuracy.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_78_0.png)



先程よりも大幅にテストデータに対する精度が向上したことが分かります。学習率を10分の1に下げるタイミングでロスががくっと減り、精度がガクッと上がっているのが分かります。最終的に、先程60%前後だった精度が、90%以上まで上がりました。しかし最新の研究成果では97%以上まで達成されています。さらに精度を上げるには、今回行ったようなネットワークの構造自体の改良ももちろんのこと、学習データを擬似的に増やす操作（Data augmentation）や、複数のモデルの出力を一つの出力に統合する操作（Ensemble）などなど、いろいろな工夫が考えられます。

# データセットクラスを書いてみよう

ここでは、Chainerにすでに用意されているCIFAR10のデータを取得する機能を使って、データセットクラスを自分で書いてみます。Chainerでは、データセットを表すクラスは以下の機能を持っていることが必要とされます。

- データセット内のデータ数を返す`__len__`メソッド
- 引数として渡される`i`に対応したデータもしくはデータとラベルの組を返す`get_example`メソッド

その他のデータセットに必要な機能は、`chainer.dataset.DatasetMixin`クラスを継承することで用意できます。ここでは、`DatasetMixin`クラスを継承し、Data augmentation機能のついたデータセットクラスを作成してみましょう。

## 1. CIFAR10データセットクラスを書く


```python
class CIFAR10Augmented(chainer.dataset.DatasetMixin):

    def __init__(self, train=True):
        train_data, test_data = cifar.get_cifar10()
        if train:
            self.data = train_data
        else:
            self.data = test_data
        self.train = train
        self.random_crop = 4

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        x, t = self.data[i]
        if self.train:
            x = x.transpose(1, 2, 0)
            h, w, _ = x.shape
            x_offset = np.random.randint(self.random_crop)
            y_offset = np.random.randint(self.random_crop)
            x = x[y_offset:y_offset + h - self.random_crop,
                  x_offset:x_offset + w - self.random_crop]
            if np.random.rand() > 0.5:
                x = np.fliplr(x)
            x = x.transpose(2, 0, 1)

        return x, t
```

このクラスは、CIFAR10のデータのそれぞれに対し、

- 32x32の大きさの中からランダムに28x28の領域をクロップ
- 1/2の確率で左右を反転させる

という加工を行っています。こういった操作を加えることで擬似的に学習データのバリエーションを増やすと、オーバーフィッティングを抑制することに役に立つということが知られています。これらの操作以外にも、画像の色味を変化させるような変換やランダムな回転、アフィン変換など、さまざまな加工によって学習データ数を擬似的に増やす方法が提案されています。

自分でデータの取得部分も書く場合は、コンストラクタに画像フォルダのパスとファイル名に対応したラベルの書かれたテキストファイルへのパスなどを渡してプロパティとして保持しておき、`get_example`メソッド内でそれぞれの画像を読み込んで対応するラベルとともに返す、という風にすれば良いことが分かります。

## 2. 作成したデータセットクラスを使って学習を行う

それではさっそくこの`CIFAR10`クラスを使って学習を行ってみましょう。先程使ったのと同じ大きなネットワークを使うことで、Data augmentationの効果がどの程度あるのかを調べてみましょう。`train`関数も含め、データセットクラス以外は先程とすべて同様です。


```python
model = train(DeepCNN(10), max_epoch=100, train_dataset=CIFAR10Augmented(), test_dataset=CIFAR10Augmented(False), postfix='augmented_', base_lr=0.1, lr_decay=(30, 'epoch'))
```

    epoch       main/loss   main/accuracy  val/main/loss  val/main/accuracy  elapsed_time  lr        
    [J1           2.41705     0.168578       1.96442        0.239616           21.0364       0.1         
    [J2           1.77832     0.311261       1.67194        0.363924           40.3804       0.1         
    [J3           1.58596     0.392949       1.62952        0.384889           59.8983       0.1         
    [J4           1.3736      0.48929        1.81392        0.416139           78.3853       0.1         
    [J5           1.16878     0.582181       1.34206        0.538469           97.7681       0.1         
    [J6           1.00387     0.648438       1.00019        0.639339           116.383       0.1         
    [J7           0.909489    0.686881       0.958338       0.66515            134.819       0.1         
    [J8           0.836435    0.717147       1.14386        0.650613           153.834       0.1         
    [J9           0.786064    0.736013       1.08884        0.631527           173.589       0.1         
    [J10          0.741895    0.75042        0.892615       0.71163            192.362       0.1         
    [J11          0.699808    0.764744       0.909562       0.693532           211.605       0.1         
    [J12          0.680101    0.772518       0.693694       0.768196           230.926       0.1         
    [J13          0.65564     0.780551       0.790436       0.74199            249.857       0.1         
    [J14          0.637646    0.788462       0.938974       0.712322           268.148       0.1         
    [J15          0.626624    0.792739       1.399          0.624011           287.32        0.1         
    [J16          0.626174    0.791566       1.05993        0.674446           306.526       0.1         
    [J17          0.589797    0.80189        0.733814       0.773042           324.806       0.1         
    [J18          0.597456    0.801511       0.635379       0.790348           344.435       0.1         
    [J19          0.582359    0.807452       0.67437        0.781448           364.575       0.1         
    [J20          0.578769    0.808983       0.594933       0.803699           383.932       0.1         
    [J21          0.575481    0.809363       0.899324       0.706388           402.81        0.1         
    [J22          0.56056     0.814824       0.765141       0.74288            420.17        0.1         
    [J23          0.559074    0.813899       0.653564       0.784612           439.749       0.1         
    [J24          0.5461      0.817268       0.645376       0.782338           459.755       0.1         
    [J25          0.548638    0.817216       0.781249       0.766614           479.531       0.1         
    [J26          0.547924    0.818314       0.74927        0.742583           499.529       0.1         
    [J27          0.540724    0.822937       0.684729       0.780459           518.619       0.1         
    [J28          0.538174    0.820352       0.766047       0.756131           537.466       0.1         
    [J29          0.525573    0.825687       1.07696        0.662381           556.891       0.1         
    [J30          0.522442    0.825381       0.729529       0.781448           576.545       0.1         
    [J31          0.374472    0.875799       0.322554       0.891713           596.116       0.01        
    [J32          0.295429    0.899379       0.309336       0.893196           616.043       0.01        
    [J33          0.274972    0.907289       0.292747       0.901701           635.045       0.01        
    [J34          0.26        0.910386       0.294528       0.901602           655.303       0.01        
    [J35          0.250123    0.914243       0.285186       0.904074           674.023       0.01        
    [J36          0.235666    0.918378       0.288161       0.903184           692.894       0.01        
    [J37          0.226342    0.922574       0.277501       0.90714            712.892       0.01        
    [J38          0.216918    0.925601       0.284357       0.906646           732.782       0.01        
    [J39          0.21181     0.92681        0.280313       0.90803            752.104       0.01        
    [J40          0.20146     0.93119        0.280123       0.909612           771.327       0.01        
    [J41          0.194855    0.932984       0.283762       0.907931           789.68        0.01        
    [J42          0.193677    0.933204       0.28565        0.908525           808.418       0.01        
    [J43          0.186472    0.933774       0.304119       0.901009           826.51        0.01        
    [J44          0.185192    0.935882       0.303901       0.907536           845.478       0.01        
    [J45          0.176464    0.938939       0.301849       0.902987           864.374       0.01        
    [J46          0.174909    0.939163       0.310574       0.904569           882.613       0.01        
    [J47          0.171831    0.939978       0.286953       0.908426           901.927       0.01        
    [J48          0.172587    0.939443       0.312076       0.9018             921.5         0.01        
    [J49          0.16893     0.941077       0.31851        0.898833           940.453       0.01        
    [J50          0.167457    0.942156       0.351041       0.894976           959.821       0.01        
    [J51          0.164177    0.942829       0.326005       0.902789           978.8         0.01        
    [J52          0.161436    0.943614       0.331878       0.901404           998.647       0.01        
    [J53          0.158322    0.944533       0.330473       0.899229           1017.47       0.01        
    [J54          0.157259    0.944531       0.340614       0.898635           1037.28       0.01        
    [J55          0.158028    0.944753       0.296676       0.909316           1057.27       0.01        
    [J56          0.157822    0.944992       0.307175       0.90447            1076.72       0.01        
    [J57          0.155367    0.946312       0.329747       0.897745           1095.03       0.01        
    [J58          0.149999    0.946871       0.312047       0.908623           1115.32       0.01        
    [J59          0.154755    0.945833       0.328333       0.904964           1134.18       0.01        
    [J60          0.153926    0.947351       0.331466       0.90269            1153.04       0.01        
    [J61          0.108315    0.962536       0.262127       0.920688           1172.78       0.001       
    [J62          0.0857237   0.970373       0.262611       0.923358           1192.26       0.001       
    [J63          0.0778385   0.973426       0.266739       0.924347           1211.19       0.001       
    [J64          0.0730304   0.97526        0.268768       0.924248           1229.3        0.001       
    [J65          0.0689073   0.976223       0.27424        0.924446           1248.87       0.001       
    [J66          0.0660777   0.976622       0.275464       0.925237           1268.56       0.001       
    [J67          0.0643725   0.978225       0.274544       0.925633           1286.57       0.001       
    [J68          0.061421    0.97886        0.284493       0.922567           1306.4        0.001       
    [J69          0.062867    0.97864        0.281003       0.925237           1326.03       0.001       
    [J70          0.059866    0.979006       0.282152       0.925732           1345.61       0.001       
    [J71          0.0559986   0.980639       0.288633       0.92504            1366          0.001       
    [J72          0.0562747   0.98133        0.289973       0.925633           1387.46       0.001       
    [J73          0.051901    0.981758       0.293774       0.925336           1407.7        0.001       
    [J74          0.0559888   0.980739       0.293734       0.925336           1428.96       0.001       
    [J75          0.0532844   0.981871       0.29023        0.924644           1448.57       0.001       
    [J76          0.0507484   0.983196       0.295357       0.923358           1469.59       0.001       
    [J77          0.0497761   0.982817       0.28886        0.926622           1488.75       0.001       
    [J78          0.0501891   0.983033       0.296822       0.924743           1507.39       0.001       
    [J79          0.0475229   0.983316       0.299419       0.923556           1524.96       0.001       
    [J80          0.0468904   0.983814       0.29781        0.925237           1544.22       0.001       
    [J81          0.0441662   0.984475       0.298029       0.924446           1561.95       0.001       
    [J82          0.045251    0.984375       0.299297       0.924743           1580.73       0.001       
    [J83          0.0430583   0.985216       0.298379       0.926028           1598.69       0.001       
    [J84          0.0442642   0.985154       0.300274       0.925831           1617.12       0.001       
    [J85          0.0406167   0.986293       0.305807       0.924248           1635.33       0.001       
    [J86          0.0425393   0.985617       0.303239       0.925831           1653.36       0.001       
    [J87          0.040143    0.986353       0.308565       0.923062           1671.75       0.001       
    [J88          0.0409943   0.985557       0.304132       0.925336           1690.03       0.001       
    [J89          0.0409428   0.985834       0.308335       0.925336           1708.28       0.001       
    [J90          0.0410388   0.985973       0.307658       0.923556           1726.77       0.001       
    [J91          0.0396118   0.986218       0.30318        0.92504            1744.19       0.0001      
    [J92          0.036719    0.987252       0.302251       0.925534           1762.65       0.0001      
    [J93          0.0360306   0.987892       0.304567       0.925831           1781.69       0.0001      
    [J94          0.0335703   0.988762       0.302379       0.926424           1801.04       0.0001      
    [J95          0.0366721   0.987652       0.302111       0.926028           1820.06       0.0001      
    [J96          0.0332877   0.988782       0.301645       0.925732           1837.54       0.0001      
    [J97          0.0337962   0.988491       0.302194       0.925732           1851.83       0.0001      
    [J98          0.0328447   0.98913        0.306731       0.925534           1866.64       0.0001      
    [J99          0.0341512   0.988321       0.30164        0.926919           1881.07       0.0001      
    [J100         0.0335156   0.988751       0.302511       0.927413           1895.03       0.0001      


先程のData augmentationなしの場合は90%程度で頭打ちになっていた精度が、学習データにaugmentationを施すことで92.5%以上まで向上させられることが分かりました。2.5%強の改善です。

ロスと精度のグラフを見てみましょう。


```python
Image(filename='DeepCNN_cifar10_augmented_result/loss.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_87_0.png)




```python
Image(filename='DeepCNN_cifar10_augmented_result/accuracy.png')
```




![png](Chainer%20Beginer%27s%20Hands-on_files/Chainer%20Beginer%27s%20Hands-on_88_0.png)



# もっと簡単にData Augmentationしよう

前述のようにデータセット内の各画像についていろいろな変換を行って擬似的にデータを増やすような操作をData Augmentationといいます。上では、オリジナルのデータセットクラスを作る方法を示すために変換の操作も`get_example()`内に書くという実装を行いましたが、実はもっと簡単にいろいろな変換をデータに対して行う方法があります。

それは、`TransformDataset`クラスを使う方法です。`TransformDataset`は、元になるデータセットオブジェクトと、そこからサンプルしてきた各データ点に対して行いたい変換を関数の形で与えると、変換済みのデータを返してくれるようなデータセットオブジェクトに加工してくれる便利なクラスです。かんたんな使い方は以下です。


```python
from chainer.datasets import TransformDataset

train_dataset, test_dataset = cifar.get_cifar10()


# 行いたい変換を関数の形で書く
def transform(inputs):
    x, t = inputs
    x = x.transpose(1, 2, 0)
    h, w, _ = x.shape
    x_offset = np.random.randint(4)
    y_offset = np.random.randint(4)
    x = x[y_offset:y_offset + h - 4,
          x_offset:x_offset + w - 4]
    if np.random.rand() > 0.5:
        x = np.fliplr(x)
    x = x.transpose(2, 0, 1)
    
    return x, t


# 各データをtransformにくぐらせたものを返すデータセットオブジェクト
train_dataset = TransformDataset(train_dataset, transform)
```

このようにすると、この新しい`train`は、上で自分でデータセットクラスごと書いたときと同じことをします。

## ChainerCVでいろいろな変換を簡単に行おう

さて、上では画像に対してランダムクロップと、ランダムに左右反転というのをやりました。もっと色々な変換を行いたい場合、上記の`transform`関数に色々な処理を追加していけばよいことになりますが、毎回使いまわすような変換処理をそのたびに書くのは面倒です。何かいいライブラリとか無いのかな、となります。そこで[ChainerCV](http://chainercv.readthedocs.io/en/latest) [Niitani 2017]です！今年のACM MultimediaのOpen Source Software CompetitionにWebDNN [Hidaka 2017]とともに出場していたChainerにComputer Vision向けの便利な機能を色々追加する補助パッケージ的なオープンソース・ソフトウェアです。今回はmaster版を使うため、以下のようにしてgithubから直接インストールします。

```bash
pip install git+git://github.com/chainer/chainercv
```

[ChainerCV](http://chainercv.readthedocs.io/en/latest)には、画像に対する様々な変換があらかじめ用意されています。

- [ChainerCVで使える画像変換一覧](http://chainercv.readthedocs.io/en/latest/reference/transforms.html#image)

そのため、上でNumPyを使ってごにょごにょ書いていたランダムクロップやランダム左右反転は、`chainercv.transforms`モジュールを使うと、それぞれ以下のように1行で書くことができます：

```python
x = transforms.random_crop(x, (28, 28))  # ランダムクロップ
x = chainercv.transforms.random_flip(x)  # ランダム左右反転
```

`chainercv.transforms`モジュールを使って、`transform`関数をアップデートしてみましょう。ちなみに、`get_cifar10()`で得られるデータセットでは、デフォルトで画像の画素値の範囲が`[0, 1]`にスケールされています。しかし、`get_cifar10()`に`scale=255.`を渡しておくと、値の範囲をもともとの`[0, 255]`のままにできます。今回`transform`の中で行う処理は、以下の5つです：

1. PCA lighting: これは大雑把に言えば、少しだけ色味を変えるような変換です
2. Standardization: 訓練用データセット全体からチャンネルごとの画素値の平均・標準偏差を求めて標準化をします
3. Random flip: ランダムに画像の左右を反転します
4. Random expand: `[1, 1.5]`からランダムに決めた大きさの黒いキャンバスを作り、その中のランダムな位置へ画像を配置します
5. Random crop: `(28, 28)`の大きさの領域をランダムにクロップします


```python
from functools import partial
from chainercv import transforms


train_dataset, test_dataset = cifar.get_cifar10(scale=255.)

mean = np.mean([x for x, _ in train_dataset], axis=(0, 2, 3))
std = np.std([x for x, _ in train_dataset], axis=(0, 2, 3))


def transform(inputs, train=True):
    img, label = inputs
    img = img.copy()
    
    # Color augmentation and Flipping
    if train:
        img = transforms.pca_lighting(img, 76.5)
        
    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]
    
    # Random crop
    if train:
        img = transforms.random_flip(img, x_random=True)
        img = transforms.random_expand(img, max_ratio=1.5)
        img = transforms.random_crop(img, (28, 28))
        
    return img, label

train_dataset = TransformDataset(train_dataset, partial(transform, train=True))
test_dataset = TransformDataset(test_dataset, partial(transform, train=False))
```

ちなみに、`pca_lighting`は、大雑把にいうと色味を微妙に変えた画像を作ってくれる関数です。

では、standardizationとChainerCVによるPCA Lightingを追加した`TransformDataset`を使って学習をしてみましょう。


```python
model = train(DeepCNN(10), max_epoch=100, train_dataset=train_dataset, test_dataset=test_dataset, postfix='augmented2_', base_lr=0.1, lr_decay=(30, 'epoch'))
```

    epoch       main/loss   main/accuracy  val/main/loss  val/main/accuracy  elapsed_time  lr        
    [J1           2.6137      0.129895       2.16087        0.189775           15.9238       0.1         
    [J2           2.06005     0.215333       1.87915        0.274624           28.9277       0.1         
    [J3           1.90142     0.2624         1.75637        0.33218            42.0564       0.1         
    [J4           1.78538     0.302829       1.6234         0.345827           55.205        0.1         
    [J5           1.61316     0.387248       1.45606        0.457476           68.171        0.1         
    [J6           1.4555      0.462119       1.36903        0.519284           81.1871       0.1         
    [J7           1.32249     0.526674       1.26702        0.551523           94.4554       0.1         
    [J8           1.21321     0.574659       1.38233        0.575653           107.044       0.1         
    [J9           1.13625     0.602182       1.46698        0.545392           120.164       0.1         
    [J10          1.08732     0.624361       1.21396        0.623813           132.999       0.1         
    [J11          1.03151     0.647396       0.974589       0.679688           146.278       0.1         
    [J12          0.978204    0.668179       1.37833        0.629549           159.298       0.1         
    [J13          0.946109    0.679927       0.994546       0.673358           172.214       0.1         
    [J14          0.904913    0.696154       0.834376       0.729233           185.179       0.1         
    [J15          0.894829    0.700368       0.804635       0.738232           198.777       0.1         
    [J16          0.860691    0.71262        0.963241       0.695411           212.039       0.1         
    [J17          0.846117    0.719489       0.718876       0.757516           225.282       0.1         
    [J18          0.823096    0.724225       0.655187       0.777987           238.509       0.1         
    [J19          0.819975    0.727684       0.619879       0.796974           251.969       0.1         
    [J20          0.800302    0.733676       0.739541       0.751384           265.801       0.1         
    [J21          0.784371    0.737892       0.715066       0.769482           280.13        0.1         
    [J22          0.773796    0.74369        0.739446       0.762263           293.7         0.1         
    [J23          0.767185    0.742727       0.743614       0.743275           306.943       0.1         
    [J24          0.76306     0.747336       1.03875        0.691456           320.305       0.1         
    [J25          0.759122    0.746044       0.715704       0.766515           333.653       0.1         
    [J26          0.745264    0.752558       0.866452       0.734771           347.184       0.1         
    [J27          0.743559    0.752684       0.980971       0.734771           360.633       0.1         
    [J28          0.735241    0.757133       0.651508       0.788074           374.045       0.1         
    [J29          0.732003    0.757333       0.735272       0.766416           387.933       0.1         
    [J30          0.730592    0.757812       0.640934       0.800633           401.4         0.1         
    [J31          0.549155    0.815437       0.369721       0.879945           414.765       0.01        
    [J32          0.468131    0.840385       0.326723       0.889834           428.13        0.01        
    [J33          0.444992    0.848565       0.329509       0.889438           441.333       0.01        
    [J34          0.42521     0.852801       0.332694       0.890229           454.816       0.01        
    [J35          0.413517    0.857873       0.320534       0.891416           467.961       0.01        
    [J36          0.404121    0.862812       0.304475       0.898932           481.188       0.01        
    [J37          0.396832    0.862912       0.311852       0.90002            495.033       0.01        
    [J38          0.390325    0.866747       0.300769       0.901701           508.68        0.01        
    [J39          0.378692    0.869825       0.298521       0.900613           522.005       0.01        
    [J40          0.373651    0.871795       0.300719       0.901305           535.313       0.01        
    [J41          0.365358    0.87486        0.318677       0.8929             548.654       0.01        
    [J42          0.352849    0.878357       0.293929       0.901602           562.143       0.01        
    [J43          0.352119    0.877224       0.290636       0.904074           575.363       0.01        
    [J44          0.350284    0.879716       0.300692       0.906547           589.204       0.01        
    [J45          0.34162     0.880894       0.287899       0.904272           602.849       0.01        
    [J46          0.34291     0.88143        0.283729       0.906349           616.871       0.01        
    [J47          0.336436    0.884031       0.297159       0.904964           630.645       0.01        
    [J48          0.327687    0.887019       0.302519       0.905261           645.006       0.01        
    [J49          0.331358    0.886209       0.288118       0.901899           659.13        0.01        
    [J50          0.330065    0.886749       0.284874       0.908228           673.029       0.01        
    [J51          0.323781    0.888061       0.290639       0.907041           686.453       0.01        
    [J52          0.324128    0.888067       0.319663       0.896954           700.152       0.01        
    [J53          0.319236    0.890245       0.312339       0.905162           713.566       0.01        
    [J54          0.317942    0.890905       0.322319       0.898141           727.184       0.01        
    [J55          0.318547    0.889906       0.301579       0.904767           740.278       0.01        
    [J56          0.315037    0.889924       0.305108       0.905953           754.289       0.01        
    [J57          0.310786    0.893422       0.293359       0.906547           767.587       0.01        
    [J58          0.314081    0.890645       0.288173       0.90803            781.544       0.01        
    [J59          0.311404    0.892047       0.328498       0.902195           795.144       0.01        
    [J60          0.304201    0.895161       0.324762       0.897844           809.221       0.01        
    [J61          0.257544    0.910266       0.24834        0.920985           822.853       0.001       
    [J62          0.234814    0.918269       0.251004       0.924545           836.326       0.001       
    [J63          0.222723    0.923493       0.245348       0.926325           849.858       0.001       
    [J64          0.217722    0.922937       0.243244       0.926424           863.784       0.001       
    [J65          0.211203    0.926151       0.248248       0.927017           876.722       0.001       
    [J66          0.211038    0.927689       0.243863       0.928105           890.21        0.001       
    [J67          0.209424    0.926302       0.240344       0.925534           903.314       0.001       
    [J68          0.203611    0.929588       0.24749        0.92682            916.759       0.001       
    [J69          0.202228    0.929648       0.24798        0.925138           930.411       0.001       
    [J70          0.1984      0.930288       0.250173       0.923952           943.638       0.001       
    [J71          0.197859    0.932385       0.253132       0.926523           956.579       0.001       
    [J72          0.192974    0.932913       0.250111       0.926226           970.071       0.001       
    [J73          0.194728    0.932505       0.246447       0.926919           983.381       0.001       
    [J74          0.194478    0.933184       0.247548       0.927611           997.03        0.001       
    [J75          0.190717    0.934014       0.250602       0.925732           1010.24       0.001       
    [J76          0.191612    0.933864       0.259754       0.925336           1023.51       0.001       
    [J77          0.194515    0.931706       0.250268       0.929984           1039.26       0.001       
    [J78          0.180834    0.937079       0.251139       0.926622           1057.64       0.001       
    [J79          0.179595    0.93702        0.259894       0.926127           1077.17       0.001       
    [J80          0.184892    0.934716       0.250286       0.92682            1096.33       0.001       
    [J81          0.17826     0.938          0.255567       0.925831           1115.8        0.001       
    [J82          0.176934    0.938279       0.259371       0.926919           1133.53       0.001       
    [J83          0.178327    0.937981       0.253367       0.926127           1153.29       0.001       
    [J84          0.177873    0.938059       0.252909       0.928797           1172.51       0.001       
    [J85          0.173131    0.938999       0.256215       0.926622           1192.59       0.001       
    [J86          0.178901    0.93782        0.254954       0.928303           1211.81       0.001       
    [J87          0.174844    0.938119       0.255952       0.927215           1229.67       0.001       
    [J88          0.173689    0.938001       0.258043       0.925534           1248.48       0.001       
    [J89          0.174803    0.939118       0.26418        0.924941           1267.49       0.001       
    [J90          0.171603    0.940657       0.258564       0.929193           1286.25       0.001       
    [J91          0.16517     0.942728       0.25537        0.929589           1305.62       0.0001      
    [J92          0.160067    0.944254       0.254748       0.930479           1325.02       0.0001      
    [J93          0.158133    0.944353       0.254335       0.928896           1344.39       0.0001      
    [J94          0.160589    0.94403        0.255143       0.928896           1363.9        0.0001      
    [J95          0.157108    0.945053       0.255221       0.929292           1383.7        0.0001      
    [J96          0.160223    0.94355        0.250388       0.92949            1403.22       0.0001      
    [J97          0.158621    0.945173       0.253497       0.929786           1423.33       0.0001      
    [J98          0.157769    0.944773       0.254209       0.930083           1442.77       0.0001      
    [J99          0.154349    0.947015       0.253928       0.928896           1461.64       0.0001      
    [J100         0.155439    0.946072       0.253269       0.930083           1475.87       0.0001      


わずかに93%を超えました。他にもネットワークにResNetと呼ばれる有名なアーキテクチャを採用するなど、簡単に試せる改善方法がいくつかあります。ぜひご自分で色々と試してみてください。

# おわりに

本記事では、[Chainer](http://chainer.org/)に関する

- 学習ループの書き方
- Trainerの使い方
- 自作モデルの書き方
- 自作データセットクラスの書き方

**Chainerの開発にコミットしてくれる方を歓迎します！Chainerはオープンソースソフトウェアですので、皆さんが自身で欲しい機能などを提案し、Pull requestを送ることで進化していきます。興味のある方は、こちらの[Contoribution Guide](http://docs.chainer.org/en/latest/contribution.html)をお読みになった後、ぜひIssueを立てたりPRを送ったりしてみてください。お待ちしております。**

chainer/chainer
[https://github.com/chainer/chainer](https://github.com/pfnet/chainer)

#### 参考文献

[Tokui 2015] Tokui, S., Oono, K., Hido, S. and Clayton, J., Chainer: a Next-Generation Open Source Framework for Deep Learning, Proceedings of Workshop on Machine Learning Systems(LearningSys) in The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS), (2015)

[Niitani 2017] Yusuke Niitani, Toru Ogawa, Shunta Saito, Masaki Saito, "ChainerCV: a Library for Deep Learning in Computer Vision", ACM Multimedia (ACMMM), Open Source Software Competition, 2017

[Hidaka 2017] Masatoshi Hidaka, Yuichiro Kikura, Yoshitaka Ushiku, Tatsuya Harada. WebDNN: Fastest DNN Execution Framework on Web Browser. ACM International Conference on Multimedia (ACMMM), Open Source Software Competition, pp.1213-1216, 2017.

#### 脚注

[^TrainingデータとValidationデータ]: 本記事では、Chainerの使い方の説明に主眼を置いているため、ValidationデータセットとTestデータセットを明確に区別していません。しかし実際にはこれらは区別されるべきです。普通、Trainingデータの一部をTrainingデータセットから取り除き、それらの取り除かれたデータでValidationデータセットを構成しておきます。その後、Trainingデータで訓練したモデルをまずValidationデータで評価し、Validationデータでの性能を向上させるようにモデルを改良していくというのが一般的な手順です。Testデータは全ての取り組みが終了したあとに、最終的なそのモデルの性能を（例えば他のモデルなどと比較する目的で）評価するためにだけ用いられます。偏ったデータを使ってモデル改良を行ってしまいオーバーフィッティングなどに陥ることを避けるなどの目的で、Training/Validationデータの構成を複数用意しておく場合もあります。

[^NN]: 学習データに対する予測精度は、もし学習データから抜き出されたあるデータをクエリとし、それが含まれている学習データセットから検索して発見することが必ずできるならば、そのデータについているラベルを答えることで、100%になってしまいます。
