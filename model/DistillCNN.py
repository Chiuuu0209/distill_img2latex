from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

# from ..data.utils import Tokenizer
# from ..models import ResNetTransformer,CnnRnn,Cnn
# from .metrics import CharacterErrorRate
# import torch.nn.functional as F


from .img2latex.data.utils import Tokenizer
from .img2latex.lit_models.metrics import CharacterErrorRate
from .student_model import Cnn
# from .img2latex.models import ResNetTransformer , ResNetTransformerLight
# from .img2latex.lit_models import LitResNetTransformer

class LitCnn(LightningModule):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        max_output_len: int,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        milestones: List[int] = [5],
        gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        vocab_file = Path(__file__).resolve().parents[0] / "img2latex" / "data" / "vocab.json"
        self.tokenizer = Tokenizer.load(vocab_file)
        self.model = Cnn(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            max_output_len=max_output_len,
            sos_index=self.tokenizer.sos_index,
            eos_index=self.tokenizer.eos_index,
            pad_index=self.tokenizer.pad_index,
            num_classes=len(self.tokenizer),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)
        self.val_cer = CharacterErrorRate(self.tokenizer.ignore_indices)
        self.test_cer = CharacterErrorRate(self.tokenizer.ignore_indices)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        #loss = self.loss_fn(logits, targets[:, 1:])
        
        targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))
        loss = self.loss_fn(logits, targets)
        
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        
        #loss = self.loss_fn(logits, targets[:, 1:])
        targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))
        loss = self.loss_fn(logits, targets)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = self.model.predict(imgs)
        val_cer = self.val_cer(preds, targets)
        self.log("val/cer", val_cer)

    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self.model.predict(imgs)

        #targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))

        test_cer = self.test_cer(preds, targets)
        self.log("test/cer", test_cer)
        return preds

    def test_epoch_end(self, test_outputs):
        with open("test_predictions.txt", "w") as f:
            for preds in test_outputs:
                for pred in preds:
                    decoded = self.tokenizer.decode(pred.tolist())
                    decoded.append("\n")
                    decoded_str = " ".join(decoded)
                    f.write(decoded_str)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
