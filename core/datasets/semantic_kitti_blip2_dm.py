import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

class SemanticKITTIBlipDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        data_root,
        batch_size=1,
        num_workers=4,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.predict_dataset = self.dataset(self.data_root, 'predict')
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True)