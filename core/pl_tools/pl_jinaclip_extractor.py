import torch

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from configs.config import CONF

from core.datasets import SemanticKITTIBlipTextDataModule
from core.model.jinaclip.jina_clip_extractor import JinaCLIPExtractor

def main():
    num_gpu = torch.cuda.device_count()

    dm = SemanticKITTIBlipTextDataModule(
        data_root=CONF.PATH.DATA_ROOT,
        foundation_model='LLaVA',
        )

    model = JinaCLIPExtractor(
        truncate_dim=1024
    )

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

    # python /u/home/caoh/projects/MA_Jiachen/VLGSSC/core/pl_tools/pl_jinaclip_extractor.py