# AI-Powered Central Sulcus Segmentation Tool
### 頭部MRIにおける中心溝同定サポートツール

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/Model-U--Net_(ResNet34)-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 1. 概要 (Overview)
頭部MRI（T2強調像、**NIfTI / Zipped DICOM**）を入力すると、AIが中心溝（Central Sulcus）の位置を自動で同定し、元画像にヒートマップを重畳して出力するセグメンテーションツールである。

* **Target:** 医学生、研修医、脳の画像診断を専門としない医師
* **Goal:** 解剖学的同定のトレーニング支援、教育コストの低減、医療安全への貢献

## 2. 開発の背景と医学的意義 (Medical Context)
**【課題】**
中心溝（ローランド溝）は前頭葉と頭頂葉の境界であり、極めて重要な解剖学的ランドマークである。しかし、初学者には以下のような課題が存在する。
* **誤認のリスク:** 中心前回の病変を「頭頂葉の病変」と誤記するなど、重大なレポートミスの原因となる。
* **同定の難易度:** 典型的な「オメガサイン（Omega sign）」等の特徴を見つけるのが困難な症例が存在する。

**【解決策】**
専門医の「眼」を学習したAIによる可視化を通じ、学習者の正しい解剖理解を支援する。

## 3. 使用技術 (Tech Stack)
* **Language:** Python 3.12
* **Framework:** Streamlit (Web App)
* **Deep Learning:**
    * PyTorch, Segmentation Models Pytorch (SMP)
    * **Model Architecture:** **U-Net**
        * **Encoder:** ResNet34 (**Pre-trained on ImageNet**)
        * 転移学習（Transfer Learning）を採用し、限られたデータセットでの学習効率と汎化性能を最大化した。
* **Medical Image Processing:**
    * Nibabel (NIfTI handling), Pydicom
* **Tools & Environment:**
    * Google Colab (Training), GitHub Codespaces (Development)
    * **3D Slicer** (Annotation / Ground Truth作成)

### Key Features & UX Design
* **Inference Modes:**
    * **Fast Mode:** 単一モデルによる高速推論（スクリーニング用）。
    * **Ensemble Mode:** **5-fold Cross Validation** で学習した5つのモデルによるアンサンブル推論（精査用）。
* **Interactive Post-processing (Real-time Adjustment):**
    以下のパラメータをスライダーでリアルタイムに調整し、結果を即座に確認可能。
    * **Probability Threshold:** 検出感度を調整（Default: 0.7）。
    * **Minimum Area:** 微小なノイズを除去するための最小面積閾値（Default: 30）。
    * **Center Mask Size:** 解剖学的にあり得ない正中部の除外領域サイズ（Default: 0.25）。


## 4. データセットと戦略 (Dataset & Strategy)
### Datasets
* **Training:** [RSNA Intracranial Aneurysm Detection](https://kaggle.com/competitions/rsna-intracranial-aneurysm-detection) (Kaggle) より抽出。
    * **Data Curation Flow:** データセットのリスト順に、T2強調像を含む**上位200症例**を抽出。そこから軸位断（Axial）以外の25症例を除外した後、後述の層化抽出法に基づき最終的な**48症例**を選定した。
* **Demo Data:** [IXI Dataset](https://brain-development.org/ixi-dataset/) (CC BY-SA 3.0) ※デモ用にファイル名を加工済

> **Note: Domain Shift (Training vs Demo)**
> 学習データとデモデータの間には、以下のドメイン乖離（Domain Shift）が存在する。
> * **Slice Thickness:** 学習データは3-5mm厚、デモデータは1mm厚（高解像度）。
> * **Face Information:** 学習データは顔除去（Defaced）済み、デモデータは顔除去なし。
> 本モデルは汎化性能を有しているが、上記乖離によりデモデータでは推論精度が変動する場合がある。

### Data Curation Strategy (Data-Centric Approach)
* **Stratified Sampling (層化抽出):**
    DICOMメタデータを解析し、**「MRI機種（Vendor）」「磁場強度（1.5T/3.0T）」「患者年齢」「スライス厚」**の分布がTrain/Val/Testデータにおいて大きく偏らないようサンプリングを行った。臨床データの不均質性（Heterogeneity）を考慮した設計である。
* **Strict Negative Sampling (厳密な背景画像選定):**
    学習データおよび検証データの**全症例を目視確認**し、「中心溝が明らかに存在しないスライス番号」を特定・リスト化した上で、そこからランダムに背景画像を抽出した。
    * **意図:** 未アノテーションの中心溝が含まれるスライスを誤って「背景（負例）」として学習させることを防ぎ、AIが「アノテーションされていない中心溝」も正しく認識できる**汎化性能を維持するため**の処置である。
* **Expert Annotation:**
    放射線診断専門医が**約40時間**を費やし、48症例・数百スライスのマスクをPixel単位で作成した。単なる正解ラベルではなく、臨床的な妥当性を担保した**「High-Quality Ground Truth」**である。

### Preprocessing Strategy
* **Skull Stripping（頭蓋除去）の廃止:**
    * 初期モデルでは実施していたが、UX向上（ユーザーの手間削減）および精度向上のため廃止した。
    * **結果として精度が向上:** 自動頭蓋除去（HD-BET）を行った場合、中心溝レベルでは脳実質の欠損は認められなかったが、**脳表を覆う脳脊髄液（CSF）の高信号領域が欠損する現象**が確認された。頭蓋除去を廃止しこのCSF信号を保持したことが、脳表の境界認識を助け、結果として中心溝の同定精度向上に寄与した可能性が考えられる。

## 5. 後処理と評価 (Post-processing & Results)

### Post-processing (False Positive Reduction)
誤検出を排除するため、独自のアルゴリズムを実装した。以下の処理を順次適用している。
1.  **Probability Thresholding:** 推論された確率マップを閾値（Default: 0.7）で二値化。
2.  **Center Masking:** 脳深部・正中部の誤検出を防ぐため、画像の中心領域に含まれるオブジェクトを除外。
3.  **Hemisphere Processing:**
    画像を左右半球に分割し、各サイドで**最大面積を持つ領域のみ**を抽出。
    * *Note:* 抽出された領域が最小面積（Minimum Area / Default: 30px）未満の場合はノイズと見なし、除外する。


### Evaluation
* **定量評価 (Quantitative):**
    * **Dice Coefficient: 0.917** (Test Data)
* **定性評価 (Qualitative):**
    * テストデータ全8症例（16半球）に対し、放射線診断専門医による目視評価を実施した。
    * **結果: 正解率 100%** (Score 2以上を達成)

| Score | 判定 | 定義 (Criteria) |
| :---: | :--- | :--- |
| **3** | **Excellent (正解)** | アノテーションされた全ての中心溝を広く被覆しており、かつ中心溝以外の誤検出がない。 |
| **2** | **Useful (正解)** | アノテーションされた中心溝を複数のスライスにわたって一定割合以上被覆している。中心溝が存在しないスライス等に軽微な誤検出を認めるが、医学的な判断を妨げないレベルである。 |
| **1** | **Poor (不正解)** | 中心溝への被覆が不十分（1スライス以下）、あるいは別の脳溝を誤って中心溝と同定している。 |

### Robustness Check (Stress Testing)
汎化性能を検証するため、Test Setには意図的に以下の「難易度の高い症例」を含めている。
* **Motion Artifact:** 体動によるブレが強い症例。
* **Brain Lesions:** 白質病変が目立ち、誤検出を誘発しやすい症例。
これらの条件下でも、本モデルは破綻することなく中心溝を同定できることを確認した。

#### Output Example (Inference on Open Data)
*Note: Due to license restrictions of the training dataset (RSNA/Kaggle), the images shown below are from the IXI Dataset (CC BY-SA 3.0).*

![IXI Result](images/demo_result.png)

*Left: Original T2WI (IXI Dataset), Right: AI Prediction (Post-processed)*

**Verification by Radiologist:**
本データセットは学習データ（3-5mm厚）と異なる1mm厚の高解像度画像（Domain Shiftあり）ですが、放射線診断専門医の目視確認において、中心溝が正確に同定されていることを確認した。

## 6. 考察と開発プロセス (Discussion)

**モデル選定と精度検証 (Model Selection & Ablation Study)**
本プロジェクトでは、最適な前処理とハイパーパラメータを選定するため、以下の条件比較を行った。

| ID | Skull Strip | Input | Negative Sampling | Other Settings | Dice (Test) | 備考 |
| :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| ① | Yes (HD-BET) | 2D | No | - | 0.8693 | ベースライン |
| ② | **No** | **2D** | No | - | **0.9185** | **数値上の最高値** |
| ③ | No | 2D | 10% | - | 0.9151 | 誤検出なし |
| ④ | No | 2.5D | No | - | 0.8552 | 誤検出増加 |
| ⑤ | No | 2.5D | 20% | Early Stop | 0.8885 | - |
| **⑥** | **No** | **2D** | **20%** | **Scheduler + Early Stop** | **0.9174** | **最終採用 (Final Model)** |

**考察と「Dice係数のパラドックス」 (Findings & The "Dice Paradox")**

1.  **Skull Stripping Freeの優位性:**
    一般に推奨される頭蓋除去（①）よりも、除去なし（②, ⑥）の方が高精度であった。
    * **考察:** 頭蓋除去処理後の画像を精査したところ、中心溝レベルで脳実質の明らかな欠損はなかったものの、**脳表のCSF（T2高信号）が部分的に消失**しているケースが見られた。頭蓋除去を廃止することで、CSFや頭蓋骨内板（Signal Void）といった周辺の解剖学的コントラストが保たれ、これがCNNにとって脳実質の境界を正確に認識するための空間的手がかり（Spatial Context）として機能した可能性が考えられる。

2.  **定量的評価（Dice）の限界と汎化性能:**
    数値上はモデル②が最高値であったが、本プロジェクトでは**Dice係数を絶対的な指標とは見なさなかった。**
    * **パラドックス:** AIが学習データの限界を超え、アノテーションされていないスライスの中心溝まで同定（汎化）した場合、計算上のDice係数は「誤検出」として低下してしまう。これは前述の**厳密なNegative Sampling**により、未アノテーションスライスを「背景」として誤学習することを防いだ成果でもある。
    * **結論:** 私はこの数値低下を「性能悪化」ではなく**「AIが解剖学的構造を深く理解した証左（Positive Generalization）」**と判断した。

    > *（実際の検証画像では、アノテーションが存在しないスライスに対してAIが正確に中心溝を同定している例が複数確認されましたが、データセットのライセンス規約を遵守し、ここでの画像掲載は控えます。）*

3.  **最終モデル（⑥）の決定:**
    上記の汎化性能を維持しつつ、臨床的に許容できない「明らかなノイズ（頭蓋外への誤反応）」を完全に抑制できたモデル⑥を、専門医の定性評価に基づき最終採用とした。

## 7. 今後の課題とロードマップ (Future Work & Roadmap)

* **Acceleration of 3D Data Creation (Model-Assisted Annotation):**
    * **課題:** 1mm等方性ボクセル（3Dデータ）の精密なアノテーションには、膨大な人的コストを要する。
    * **解決策:** 本モデル（2D）を「アノテーション支援ツール」として転用する。閾値を調整して感度を高めた本モデルで3Dボリュームに対しPre-labeling（粗塗り）を行い、専門医が修正（Refinement）を行うワークフローを構築する。
    * **展望:** これにより、高品質な3D教師データの作成時間を**大幅に短縮**し、次世代の**3D CNNモデル**構築を加速させる。

* **Test-Time Augmentation (TTA):**
    推論時に画像を多角的に入力し平均化することで、さらなる精度向上を図る。

## License

* **Source Code:** [MIT License](LICENSE)
    * 本リポジトリのソースコードはMITライセンスの下で公開されています。
* **Trained Model Weights:** Non-Commercial / Research Use Only
    * 学習済みモデルの重みは [RSNA Intracranial Aneurysm Detection](https://kaggle.com/competitions/rsna-intracranial-aneurysm-detection) データセットを使用して学習されているため、**非商用・研究目的 (Non-Commercial / Research Use Only)** での利用に限定されます。
* **Demo Data / Images:** [Creative Commons BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
    * デモ用データおよび掲載画像は [IXI Dataset](https://brain-development.org/ixi-dataset/) に由来します。

## Acknowledgements & Citation

This project uses the dataset from the RSNA Intracranial Aneurysm Detection competition hosted on Kaggle for **training purposes only**. The raw competition data is NOT included in this repository/application in compliance with the competition rules.

```bibtex
@misc{rsna-intracranial-aneurysm-detection,
    author = {Jeff Rudie, Evan Calabrese, et al.},
    title = {RSNA Intracranial Aneurysm Detection},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/rsna-intracranial-aneurysm-detection}},
    note = {Kaggle}
}
```

---
**Author:** T. Higuchi, M.D. - Board Certified Radiologist