import time
import tensorflow as tf
import config


def loss_function(real, pred):
    """计算损失函数"""
    # 创建掩码，过滤掉填充位置
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, targ_lang):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * config.BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def train(dataset, epochs, encoder, decoder, checkpoint, optimizer, steps_per_epoch, targ_lang):
    """训练模型"""
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, targ_lang)
            total_loss += batch_loss
            
            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
        # 每个epoch结束后保存检查点
        checkpoint.save(file_prefix=config.CHECKPOINT_PATH)
        
        # 每个epoch结束后保存完整模型
        encoder.save(config.MODEL_PATH + '.encoder')
        decoder.save(config.MODEL_PATH + '.decoder')
        
        print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
        print(f'Model saved to {config.MODEL_PATH}')
        print('-' * 50) 