# Nodule Classification

## ipynb Files
### NoduleClassification.ipynb
訓練模型以及測試結果所使用的主要檔案

### NoduleDatectionTestData.ipynb
Generate data in /DetectionTest

### make_detection_data.ipynb
Generate data in /DetectionData

## Folders
### LunaData
Nodule Classification 所使用的原始訓練資料

### previous_trials
過去各種訓練的測試結果。檔名格式：

`classification_{Used Model}_{CELoss * 10^4}_{F1 score * 10^4}_{Model Details}_{k-fold cross validation / batch size}_{optimizer / learning rate / weight decay}_{random seed}.pth`

### DetectionTest
單純生成 Nodule Detection 的 Testing Data

### DetectionData
生成類似 Nodule Detection 訓練使用的資料集
