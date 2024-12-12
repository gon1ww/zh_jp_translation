import tensorflow as tf
import os
import time

import config
from models import Encoder, Decoder
from preprocessing import prepare_data, max_length
from train import train_step
from translate import translate


def load_trained_model(checkpoint):
    """加载已训练的模型"""
    try:
        # 检查检查点文件是否存在
        if tf.train.latest_checkpoint(config.CHECKPOINT_DIR):
            # 恢复检查点
            checkpoint.restore(tf.train.latest_checkpoint(config.CHECKPOINT_DIR))
            print("成功加载检查点")
            
            # 如果检查点加载成功，尝试加载完整模型
            if (os.path.exists(config.MODEL_PATH + '.encoder') and 
                os.path.exists(config.MODEL_PATH + '.decoder')):
                print("成功加载完整模型")
                return checkpoint.encoder, checkpoint.decoder
                
        print("未找到已保存的模型文件，将创建新模型")
        return None, None
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None


def main():
    # 准备数据
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, inp_lang, targ_lang = prepare_data(
        num_examples=10000)
    
    # 计算最大长度
    max_length_inp = max_length(input_tensor_train)
    max_length_targ = max_length(target_tensor_train)
    
    # 确保输入和目标张量的长度相同
    max_seq_length = max(max_length_inp, max_length_targ)
    input_tensor_train = tf.keras.preprocessing.sequence.pad_sequences(
        input_tensor_train, maxlen=max_seq_length, padding='post')
    target_tensor_train = tf.keras.preprocessing.sequence.pad_sequences(
        target_tensor_train, maxlen=max_seq_length, padding='post')
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
    dataset = dataset.shuffle(config.BUFFER_SIZE)
    dataset = dataset.padded_batch(
        config.BATCH_SIZE,
        padded_shapes=([None], [None]),
        padding_values=(0, 0),
        drop_remainder=True
    )
    
    # 计算steps_per_epoch
    steps_per_epoch = len(input_tensor_train) // config.BATCH_SIZE
    
    # 创建优化器和检查点
    optimizer = tf.keras.optimizers.Adam()
    
    # 创建或加载模型
    print("正在加载模型...")
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    
    encoder = Encoder(vocab_inp_size, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                   encoder=encoder,
                                   decoder=decoder)
    
    encoder_loaded, decoder_loaded = load_trained_model(checkpoint)
    if encoder_loaded and decoder_loaded:
        encoder, decoder = encoder_loaded, decoder_loaded
    else:
        print("创建新模型并开始训练...")
        # 开始训练
        for epoch in range(config.EPOCHS):
            start = time.time()
            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0
            
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, targ_lang)
                total_loss += batch_loss
                
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                               batch,
                                                               batch_loss.numpy()))
            
            # 每 2 个周期保存一次检查点
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=config.CHECKPOINT_PATH)
                # 保存完整模型
                encoder.save(config.MODEL_PATH + '.encoder')
                decoder.save(config.MODEL_PATH + '.decoder')
                print(f'已保存模型到: {config.MODEL_PATH}')
            
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                              total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    print("模型准备就绪!")
    print("输入 'q' 或 'quit' 退出程序")
    
    while True:
        user_input = input("\n请输入要翻译的中文句子: ").strip()
        
        if user_input.lower() in ['q', 'quit']:
            print("谢谢使用!")
            break
        
        if not user_input:
            continue
        
        translate(user_input, encoder, decoder, inp_lang, targ_lang,
                 max_length_inp, max_length_targ)


if __name__ == "__main__":
    main()