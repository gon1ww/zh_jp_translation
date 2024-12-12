import io
import re
import tensorflow as tf
import unicodedata
from sklearn.model_selection import train_test_split
import MeCab
import jieba
import numpy as np
import config


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_chinese(w):
    """预处理中文文本"""
    w = ' '.join(jieba.cut(w.strip()))
    w = re.sub(r'[" "]+', " ", w)
    w = '<start> ' + w + ' <end>'
    return w


def preprocess_japanese(w):
    """预处理日语文本"""
    mecab = MeCab.Tagger("-Owakati")
    w = mecab.parse(w.strip()).strip()
    w = re.sub(r'[" "]+', " ", w)
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path_cmn, path_jpn, num_examples):
    """创建中日文数据集"""
    # 读取文件
    with open(path_cmn, 'r', encoding='utf-8') as f:
        cmn_lines = f.readlines()
    with open(path_jpn, 'r', encoding='utf-8') as f:
        jpn_lines = f.readlines()

    # 确保两个文件的行数相同
    min_len = min(len(cmn_lines), len(jpn_lines))
    if num_examples:
        min_len = min(min_len, num_examples)

    cmn_lines = cmn_lines[:min_len]
    jpn_lines = jpn_lines[:min_len]

    # 预处理文本
    cmn_processed = [preprocess_chinese(line.strip()) for line in cmn_lines]
    jpn_processed = [preprocess_japanese(line.strip()) for line in jpn_lines]

    return jpn_processed, cmn_processed


def tokenize(lang):
    """将文本转换为序列"""
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    
    # 计算最大长度
    max_length = max(len(t) for t in tensor)
    
    # 填充序列
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         maxlen=max_length,
                                                         padding='post')
    return tensor, lang_tokenizer


def load_dataset(path_cmn, path_jpn, num_examples=None):
    """加载数据集"""
    targ_lang, inp_lang = create_dataset(path_cmn, path_jpn, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    
    # 确保输入和目标张量长度相同
    max_length = max(input_tensor.shape[1], target_tensor.shape[1])
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                              maxlen=max_length,
                                                              padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                               maxlen=max_length,
                                                               padding='post')
    
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def prepare_data(num_examples=None):
    """准备训练数据"""
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
        config.PATH_TO_CMN, config.PATH_TO_JPN, num_examples)

    # 分割训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2)

    return (input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val,
            inp_lang, targ_lang)


def max_length(tensor):
    """计算张量的最大长度"""
    return max(len(t) for t in tensor)