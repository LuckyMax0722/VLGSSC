import torch

from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from configs.config import CONF

from core.datasets import SemanticKITTIBlipDataset
from core.model.llava.LLaVA_net import LLaVAModule

def main():
    ds = SemanticKITTIBlipDataset(
        data_root=CONF.PATH.DATA_ROOT,
        split='predict04',
    )

    model = LLaVAModule(
        model_path="liuhaotian/llava-v1.6-34b",
        load_8bit=True
    )

    model.setup(stage="predict") 
    
    for batch in tqdm(ds, desc="Generate descriptions"):
        model(batch)


if __name__ == '__main__':
    main()

    # python /u/home/caoh/projects/MA_Jiachen/VLGSSC/core/pl_tools/pl_llava_predict.py