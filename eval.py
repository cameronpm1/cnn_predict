import os
import sys
import glob
import yaml
import torch
import pprint
import shutil
import numpy as np
from tqdm import tqdm
from munch import munchify
from collections import OrderedDict
from models import VisDynamicsModel
from dataset import (DynamicImageDatasetStep,
                     DynamicImageDatasetConstant)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


def main(
    config_dir: str,
    ckpt_dir: str,
):
    config_filepath = config_dir
    checkpoint_filepath = ckpt_dir
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[-1]
    print(checkpoint_filepath)
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisDynamicsModel(lr=cfg.lr,
                             seed=cfg.seed,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             dataset=cfg.dataset,
                             lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)
    if 'refine' in cfg.model_name:
        ckpt = rename_ckpt_for_multi_models(ckpt)
        model.model.load_state_dict(ckpt)

        high_dim_checkpoint_filepath = str(sys.argv[3])
        high_dim_checkpoint_filepath = glob.glob(os.path.join(high_dim_checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(high_dim_checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        model.high_dim_model.load_state_dict(ckpt)

    else:
        model.load_state_dict(ckpt['state_dict'])

    model.eval()
    model.freeze()

    trainer = Trainer(deterministic=True,
                      default_root_dir=log_dir,
                      val_check_interval=1.0)

    trainer.test(model)
    model.test_save()

# gather latent variables by running training data on the trained high-dim models
def gather_latent_from_trained_high_dim_model(
    config_dir: str,
    ckpt_dir: str,
):
    config_filepath = config_dir
    checkpoint_filepath = ckpt_dir
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisDynamicsModel(lr=cfg.lr,
                             seed=cfg.seed,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             dataset=cfg.dataset,
                             lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    #model = model.to('cuda')
    model.eval()
    model.freeze()

    # prepare train and val dataset
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if cfg.if_cuda else {}
    train_dataset = DynamicImageDatasetStep(data_filepath=cfg.data_filepath,
                                      flag='train',
                                      seed=cfg.seed,
                                      object_name=cfg.dataset)
    val_dataset = DynamicImageDatasetStep(data_filepath=cfg.data_filepath,
                                    flag='val',
                                    seed=cfg.seed,
                                    object_name=cfg.dataset)
    # prepare train and val loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.train_batch,
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=cfg.val_batch,
                                             shuffle=False,
                                             **kwargs)

    # run train forward pass to save the latent vector for training the refine network later
    all_filepaths = []
    all_latents = []
    var_log_dir = os.path.join(log_dir, 'variables')
    for batch_idx, (data, target, filepath) in enumerate(tqdm(train_loader)):
        if cfg.model_name == 'encoder-decoder':
            output, latent = model.model(data.cuda())
        if cfg.model_name == 'encoder-decoder-64':
            #output, latent = model.model(data.cuda(), data.cuda(), False)
            output, latent = model.model(data.cpu(), data.cpu(), False)
        # save the latent vectors
        all_filepaths.extend(filepath)
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)

    mkdir(var_log_dir+'_train')
    np.save(os.path.join(var_log_dir+'_train', 'ids.npy'), all_filepaths)
    np.save(os.path.join(var_log_dir+'_train', 'latent.npy'), all_latents)

    # run val forward pass to save the latent vector for validating the refine network later
    all_filepaths = []
    all_latents = []
    var_log_dir = os.path.join(log_dir, 'variables')
    for batch_idx, (data, target, filepath) in enumerate(tqdm(val_loader)):
        if cfg.model_name == 'encoder-decoder':
            output, latent = model.model(data.cuda())
        if cfg.model_name == 'encoder-decoder-64':
            #output, latent = model.model(data.cuda(), data.cuda(), False)
            output, latent = model.model(data.cpu(), data.cpu(), False)
        # save the latent vectors
        all_filepaths.extend(filepath)
        for idx in range(data.shape[0]):
            latent_tmp = latent[idx].view(1, -1)[0]
            latent_tmp = latent_tmp.cpu().detach().numpy()
            all_latents.append(latent_tmp)

    mkdir(var_log_dir+'_val')
    np.save(os.path.join(var_log_dir+'_val', 'ids.npy'), all_filepaths)
    np.save(os.path.join(var_log_dir+'_val', 'latent.npy'), all_latents)

if __name__ == '__main__':
    dataset = 1
    sim = 'cube_2d'

    config_filepath = 'configs/'+sim+'/model64/config1.yaml'
    #ckpt_filepath = '/home/cameron/cnn_predict/logs/_cube_2d/video1_encoder-decoder-64_1/lightning_logs/checkpoints/{epoch}_{val_loss}/'
    ckpt_filepath = '/home/cameron/cnn_predict/logs/_cube_2d/video2_encoder-decoder-64_1/lightning_logs/checkpoints/{epoch}_{val_loss}/'
    gather_latent_from_trained_high_dim_model(config_filepath,ckpt_filepath)
    #main(config_filepath,ckpt_filepath)