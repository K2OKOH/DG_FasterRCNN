# DG_FasterRCNN
## 原始文件
### trainval_net.py
- FasterRCNN 的 baseline，测试用test_net.py
### da_trainval_net.py
- DA_FasterRCNN 的 baseline，测试用da_test_net.py 

## xmj 添加文件
### FasterRCNN_DAD_simple.py
- DG部分的尝试，尝试使用域对抗的方法，让 FasterRCNN 学到域共有特征

### FasterRCNN.py
- FasterRCNN 的 baseline
- 可完成 source 训练，target/source测试

### FasterRCNN_ML.py
- 使用了meta-learning的域泛化（并不标准，仅在meta—test部分反传了test的loss）
- Domain：source + 频率生成的两个域

### FasterRCNN_DG.py
- 用 FasterRCNN 的 baseline 训练 3个domain的混合域（代码除了读取的数据，都相同）
- domain3 在外部手动混合，混合域的样本数量不变
- Domain：source + 频率生成的两个域

### FasterRCNN_6DataSet.py
- 6个域混合的训练
- omain：source + 频率生成的两个域 + watercolor + comic + clipart