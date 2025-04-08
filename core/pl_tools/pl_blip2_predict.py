import torch

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from configs.config import CONF

from core.datasets import SemanticKITTIBlipDataModule
from core.model.blip2 import Blip2Module

def main():
    num_gpu = torch.cuda.device_count()

    dm = SemanticKITTIBlipDataModule(
        data_root=CONF.PATH.DATA_ROOT,
        )

    model = Blip2Module()
    
    trainer = pl.Trainer(
        logger=False,
        devices=[i for i in range(num_gpu)],
        strategy=DDPStrategy(
            accelerator='gpu',
            find_unused_parameters=False
        ),
    )

    trainer.predict(model=model, datamodule=dm)


if __name__ == '__main__':
    main()

    # python /u/home/caoh/projects/MA_Jiachen/VLGSSC/core/pl_tools/pl_blip_predict.py