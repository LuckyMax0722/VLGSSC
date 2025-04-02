import os
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from configs.config import CONF

from core.datasets import SemanticKITTIBlipDataset, SemanticKITTIBlipDataModule
from core.model.blip2 import Blip2Module

def main():
    num_gpu = torch.cuda.device_count()

    dm = SemanticKITTIBlipDataModule(
        dataset=SemanticKITTIBlipDataset,
        data_root=CONF.PATH.DATA_ROOT,
        )

    model = Blip2Module()
    
    trainer = pl.Trainer(
        #devices=[i for i in range(num_gpu)],
        devices=1,
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=False
        ),
    )

    trainer.predict(model=model, datamodule=dm)


if __name__ == '__main__':
    main()

    # CUDA_VISIBLE_DEVICES=5 python /u/home/caoh/projects/MA_Jiachen/VLGSSC/core/pl_tools/pl_blip_predict.py