"""配置参数"""

import os

# 环境配置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 模型参数
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 512

# 训练参数
EPOCHS = 10

# 文件路径
PATH_TO_CMN = "data/zh.txt"  # 中文语料路径
PATH_TO_JPN = "data/ja.txt"  # 日文语料路径

# 模型保存路径
MODEL_PATH = './checkpoints/model'
CHECKPOINT_DIR = './checkpoints'
MODEL_PATH = os.path.join(MODEL_PATH, 'zh_jp_translator')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'ckpt')