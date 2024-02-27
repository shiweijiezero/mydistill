import logging
import os
from pprint import pprint

import pytorch_lightning
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from data_loader import PLDataModule
from wrapper import Wrapper

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    config_dir = 'config.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)
    # log config
    logger.info(config)
    pytorch_lightning.seed_everything(config["seed"])

    # dataset
    dataset = PLDataModule(batch_size=config["batch_size"],model_max_length=config["model_max_length"])
    dataset.setup()

    # print(dataset.train_dataloader())
    # print("训练数据格式：")
    # for item in dataset.train_dataloader():
    #     # print(item.shape)
    #     # print(item)
    #     print(list(item.keys()))
    #     pprint(item["text"])
    #     break

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="./log",
        version=None,
        name='lightning_logs'
    )

    # early_stopping = pytorch_lightning.callbacks.EarlyStopping(monitor='loss/loss',
    #                                                            patience=3,
    #                                                            mode='min')
    #
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='loss/loss',
        save_top_k=2,
        save_last=True,
        every_n_train_steps=300000,
        dirpath="./save_model",
    )
    #
    model = Wrapper(config)
    trainer = Trainer(precision="bf16-mixed",
                      max_epochs=config["n_epoch"],
                      # auto_scale_batch_size="power",
                      logger=tb_logger,
                      devices=config["CUDA_VISIBLE_DEVICES"],
                      # accelerator="gpu", devices=1,
                      accumulate_grad_batches=config["gradient_accumulation_steps"],
                      gradient_clip_val=config["max_grad_norm"],
                      # strategy="ddp",
                      callbacks=[checkpoint_callback,
                                 # early_stopping
                                 ]
                      )
    trainer.fit(model, dataset)
    # trainer.fit(model, dataset, ckpt_path="./save_model/last.ckpt")

if __name__ == '__main__':
    main()
