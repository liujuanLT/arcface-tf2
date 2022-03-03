from absl import logging
import cv2
import os
import numpy as np
import tensorflow as tf
import argparse
import sys

sys.path.append(os.path.dirname(__file__))
from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm

def main(cfg):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()
    ckpt_path = tf.train.latest_checkpoint(cfg['model_path'])
    use_pretrain = (ckpt_path is None)
    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False,
                         use_pretrain=use_pretrain)

    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    if cfg['img_path']:
        print("[*] Encode {} to ./output_embeds.npy".format(cfg['img_path']))
        img = cv2.imread(cfg['img_path'])
        img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        embeds = l2_norm(model(img))
        np.save('./output_embeds.npy', embeds)
    else:
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(lfw_data_path=cfg['test_dataset'])

        if lfw:
            print("[*] Perform Evaluation on LFW...")
            acc_lfw, best_th = perform_val(
                cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
                is_ccrop=cfg['is_ccrop'])
            print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

        if agedb_30:
            print("[*] Perform Evaluation on AgeDB30...")
            acc_agedb30, best_th = perform_val(
                cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
                agedb_30_issame, is_ccrop=cfg['is_ccrop'])
            print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

        if cfp_fp:
            print("[*] Perform Evaluation on CFP-FP...")
            acc_cfp_fp, best_th = perform_val(
                cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
                is_ccrop=cfg['is_ccrop'])
            print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))

def parse_args(argv):
    parsor = argparse.ArgumentParser()
    parsor.add_argument('--cfg_path', type=str,
        help='config file path', default='./configs/arc_res50.yaml')
    parsor.add_argument('--gpu', type=str, 
        help='which gpu to use', default='0')
    parsor.add_argument('--img_path', type=str,
        help='path to input image', default='')
    parsor.add_argument('--model_path', type=str,
        help='path for ckpt directory', default='')
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
