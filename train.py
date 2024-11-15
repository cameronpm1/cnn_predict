import os
import sys
import yaml
import torch
import pprint
from munch import munchify
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.accelerators import find_usable_cuda_devices

from models import VisDynamicsModel

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


def main(config_filepath):
    config_filepath = config_filepath
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
                             if_test=False,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             dataset=cfg.dataset,
                             lr_schedule=cfg.lr_schedule,
                             input_type=cfg.input_type)

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + "/lightning_logs/checkpoints/{epoch}_{val_loss}",
        verbose=True,
        monitor='val_loss',
        mode='min')

    #setup tensorboard logger
    now = datetime.now()
    dt_string = now.strftime('%d'+'_'+'%m'+'_'+'%Y'+'_'+'%H:%M:%S')
    tb_log_dir = log_dir + '/tb_logs/'
    if not os.path.exists(tb_log_dir):
        mkdir(tb_log_dir)
    logger = TensorBoardLogger(tb_log_dir, name=dt_string)

    # define trainer
    trainer = Trainer(accelerator="cuda", 
                      devices=find_usable_cuda_devices(1),
                      logger=logger,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      callbacks=checkpoint_callback
                      )

    trainer.fit(model)

def main_latentpred(config_filepath,):
    config_filepath = config_filepath
    cfg = load_config(filepath=config_filepath)
    high_dim_checkpoint_filepath = str(sys.argv[3])
    refine_checkpoint_filepath = str(sys.argv[4])

    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.dataset,
                        cfg.model_name,
                        str(cfg.seed)])

    model = VisLatentDynamicsModel(lr=cfg.lr,
                                   seed=cfg.seed,
                                   if_cuda=cfg.if_cuda,
                                   if_test=False,
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

    model.load_high_dim_refine_model(high_dim_checkpoint_filepath, refine_checkpoint_filepath)

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir + "/lightning_logs/checkpoints/{epoch}_{val_loss}",
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='')

    # define trainer
    trainer = Trainer(max_epochs=cfg.epochs,
                      devices=find_usable_cuda_devices(1),
                      deterministic=True,
                      accelerator='ddp',
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      checkpoint_callback=checkpoint_callback)

    trainer.fit(model)


if __name__ == '__main__':
    torch.set_num_threads(13)

    dataset = 1
    sim = 'cube_2d'

    config_filepath = 'configs/'+sim+'/model64/config1.yaml'

    main(config_filepath)

    #/home/cameron/cnn_predict/logs/_cube_2d/video1_encoder-decoder-64_1/variables/latent.npy