import os
import sys
import tensorflow as tf
import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
sys.path.append(os.path.dirname(__file__))
from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
import modules.dataset as dataset

def step_decay(epoch):
    lr0 = 0.01
    if epoch < 20:
        return lr0
    if epoch < 40:
        return 0.3 * lr0
    if epoch < 60:
        return 0.1 * lr0
    if epoch < 80:
        return 0.01 * lr0
    if epoch < 100:
        return 0.001 * lr0
    return 0.0001 * lr0

def main(cfg):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel('FATAL')
    set_memory_growth()

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         use_pretrain=cfg['use_pretrain'],
                         training=True)
    model.summary(line_length=80)

    if cfg['train_dataset']:
        print("load ms1m dataset.")
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset.load_tfrecord_dataset(
            cfg['train_dataset'], cfg['batch_size'], cfg['binary_img'],
            is_ccrop=cfg['is_ccrop'])
    else:
        print("load fake dataset.")
        steps_per_epoch = 1
        train_dataset = dataset.load_fake_dataset(cfg['input_size'])

    loss_fn = SoftmaxLoss()

    if cfg['input_ckpt_path'] is not None:
        if os.path.isdir(cfg['input_ckpt_path']):
            ckpt_path = tf.train.latest_checkpoint(cfg['input_ckpt_path'])
        else:
            ckpt_path = cfg['input_ckpt_path']
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1

    if cfg['mode'] == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        summary_writer = tf.summary.create_file_writer(
            './logs/' + cfg['sub_name'])

        train_dataset = iter(train_dataset)

        while epochs <= cfg['epochs']:
            inputs, labels = next(train_dataset)

            with tf.GradientTape() as tape:
                logist = model(inputs, training=True)
                reg_loss = tf.reduce_sum(model.losses)
                pred_loss = loss_fn(labels, logist)
                total_loss = pred_loss + reg_loss

            learning_rate = tf.constant(cfg['base_lr'])
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9, nesterov=True)
                
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #if steps % 5 == 0:
            if True:
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      learning_rate.numpy()))

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'loss/total loss', total_loss, step=steps)
                    tf.summary.scalar(
                        'loss/pred loss', pred_loss, step=steps)
                    tf.summary.scalar(
                        'loss/reg loss', reg_loss, step=steps)
                    tf.summary.scalar(
                        'learning rate', optimizer.lr, step=steps)

            if steps % cfg['save_steps'] == 0:
                print('[*] save ckpt file!')
                output_ckpt_path = os.path.join(cfg['output_ckpt_path'], f'e_{epochs}_b_{steps % steps_per_epoch}.ckpt')
                model.save_weights(output_ckpt_path)

            steps += 1
            epochs = steps // steps_per_epoch + 1
    else:
        lr_schedule = LearningRateScheduler(step_decay)
        optimizer = tf.keras.optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

        model.compile(optimizer=optimizer, loss=loss_fn)
        output_ckpt_path = os.path.join(cfg['output_ckpt_path'], 'e_{epoch}_b_{batch}.ckpt')
        mc_callback = ModelCheckpoint(
            output_ckpt_path,
            save_freq=cfg['save_steps'], verbose=1,
            save_weights_only=True)
        tb_callback = TensorBoard(log_dir='logs/',
                                  update_freq=cfg['batch_size'] * 5,
                                  profile_batch=0)
        tb_callback._total_batches_seen = steps
        tb_callback._samples_seen = steps * cfg['batch_size']
        callbacks = [mc_callback, tb_callback, lr_schedule]

        model.fit(train_dataset,
                  epochs=cfg['epochs'],
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=epochs - 1)

    print("[*] training done!")

def parse_args(argv):
    parsor = argparse.ArgumentParser()
    parsor.add_argument('--cfg_path', type=str,
        help='config file path', default='./configs/arc_res50.yaml')
    parsor.add_argument('--gpu', type=str,
        help='which gpu to use', default='0')
    parsor.add_argument('--mode', type=str,
        help='fit: model.fit, eager_tf: custom GradientTape', default='fit')

    args = parsor.parse_args(argv)
    cfg = load_yaml(args.cfg_path)
    d = args.__dict__
    args_dict = {key: d[key] for key in d if key[0]!='_'}
    for k in args_dict:
        # if k in cfg:
        if True:
            cfg[k] = args_dict[k]
    return cfg

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
