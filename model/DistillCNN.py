from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .img2latex.data.utils import Tokenizer
from .img2latex.lit_models.metrics import CharacterErrorRate
from .img2latex.lit_models import LitResNetTransformer
from .student_model import Cnn

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
        teacher : bool = True,
        pretrained_weight = Path(__file__).resolve().parents[1] / "weights" / "model.ckpt",
        loss : str = "soft",
        embedding : bool = False,
        temperature : float = 1.0,
        r_target : float = 0.999,
        r_soft : float = 0.001,
        r_hard : float = 0.001,
        r_embedding : float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        vocab_file = Path(__file__).resolve().parents[0] / "img2latex" / "data" / "vocab.json"
        self.tokenizer = Tokenizer.load(vocab_file)
        self.pretrained_weight = pretrained_weight
        print(f'pretrained_weight : {self.pretrained_weight}')
        self.teacher = teacher
        
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

        """
        load model we want to distill
            -> freeze pretrained model
        """
        if self.teacher:
            teacher_model = LitResNetTransformer.load_from_checkpoint(self.pretrained_weight)
            for (name,para) in teacher_model.named_parameters():
                para.requires_grad = False
            self.teacher_model = teacher_model.model
        else:
            self.teacher_model = None
            self.r_target = 1.0
        
        # loss params
        """
        loss : str = "soft" or "hard"
            soft : soft target loss 
                    -> loss_ce(softmax(logits), targets) + 
                    -> loss_kl(softmax(logits/temperature), softmax(teacher_logits/temperature))
            hard : hard target loss
                    -> loss_ce(softmax(logits), targets) +
                    -> loss_ce(softmax(logits), softmax(teacher_logits))

        embedding : bool  -> if True, caluate add loss with embedding layer
                    -> loss_cos(student_embedding, teacher_embedding)

        temperature : float -> temperature for soft target loss to get smooth distribution
        """
        assert loss in ["soft","hard"] , "loss must be soft or hard"
        self.loss = loss
        self.embedding = embedding
        self.temperature = temperature
        self.r_target = r_target
        self.r_soft = r_soft
        self.r_hard = r_hard
        self.r_embedding = r_embedding

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)
        self.loss_cos = nn.CosineEmbeddingLoss(margin=0.05, reduction='mean')
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        
        self.val_cer = CharacterErrorRate(self.tokenizer.ignore_indices)
        self.test_cer = CharacterErrorRate(self.tokenizer.ignore_indices)

    def training_step(self, batch, batch_idx):
        """
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        #loss = self.loss_fn(logits, targets[:, 1:])
        
        targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))
        loss = self.loss_fn(logits, targets)
        
        self.log("train/loss", loss)
        return loss
        """
        imgs, targets = batch
        embedding_x , logits = self.model.cnn_forward(imgs) # (500 ,B ,E) , (B, num_classes, 500)
        targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))
        loss = self.loss_fn(logits, targets)
        self.log("train/loss_target", loss)

        if self.teacher:    
            # freeze pretrined model
            with torch.no_grad():
                teacher_embedding_x = self.teacher_model.encode(imgs) # (Sx, B, E)
                teacher_logits = self.teacher_model.decode(targets[:, :-1],teacher_embedding_x) # (Sy, B, num_classes)
                teacher_logits = teacher_logits.permute(1, 2, 0) # (B, num_classes, Sy)

            if self.loss == "soft":
                # soft target loss
                # print("student logits shape : ", logits.shape)
                # print("teacher logits shape : ", teacher_logits.shape)
                Sy = teacher_logits.shape[2]
                # print("Sy : ", Sy)
                # print("student logits shape : ", logits[:,:,:Sy].shape)
                # print("teacher logits shape : ", teacher_logits.shape)
                # assert logits[:,:,:Sy].shape == teacher_logits.shape , "logits shape must be same"
                loss_soft = self.loss_kl(F.log_softmax(logits[:,:,:Sy]/self.temperature, dim=-1), F.softmax(teacher_logits/self.temperature, dim=-1))
                loss = loss * self.r_target + self.r_soft * loss_soft
                self.log("train/loss_soft", loss_soft)

            if self.embedding:
                # embedding loss
                embedding_x = embedding_x.permute(1, 0, 2) # (B, Sx, E)
                embedding_x = embedding_x.reshape(embedding_x.shape[0],-1) # (B, Sx*E)
                teacher_embedding_x = teacher_embedding_x.permute(1, 0, 2) # (B, Sx, E)
                teacher_embedding_x = teacher_embedding_x.reshape(teacher_embedding_x.shape[0],-1) # (B, Sx*E)
                # print(embedding_x.shape, teacher_embedding_x.shape)
                assert embedding_x.shape == teacher_embedding_x.shape , "embedding_x and teacher_embedding_x must have same shape but got {} and {}".format(embedding_x.shape, teacher_embedding_x.shape)

                y = torch.ones(imgs.shape[0]).cuda()
                loss_embedding = self.loss_cos(embedding_x.cuda(), teacher_embedding_x.cuda(), y)
                # loss_embedding = self.loss_cos(embeding_x, teacher_embedding_x)
                loss = loss + self.r_embedding * loss_embedding
                self.log("train/loss_embedding", loss_embedding)
            
        self.log("train/loss", loss)
        return loss




    def validation_step(self, batch, batch_idx):
        """
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        
        #loss = self.loss_fn(logits, targets[:, 1:])
        targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))
        loss = self.loss_fn(logits, targets)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = self.model.predict(imgs)
        val_cer = self.val_cer(preds, targets)
        self.log("val/cer", val_cer)
        """
        """
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        #loss = self.loss_fn(logits, targets[:, 1:])
        
        targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))
        loss = self.loss_fn(logits, targets)
        
        self.log("train/loss", loss)
        return loss
        """
        imgs, targets = batch
        embedding_x , logits = self.model.cnn_forward(imgs) # (500 ,B ,E) , ()
        targets = F.pad(targets, pad=(0, 500 - targets.shape[1], 0, 0))
        loss = self.loss_fn(logits, targets)
        self.log("val/loss_target", loss)

        if self.teacher:    
            # freeze pretrined model
            with torch.no_grad():
                teacher_embedding_x = self.teacher_model.encode(imgs) # (Sx, B, E)
                teacher_logits = self.teacher_model.decode(targets[:, :-1],teacher_embedding_x) # (Sy, B, num_classes)
                teacher_logits = teacher_logits.permute(1, 2, 0) # (B, num_classes, Sy)

            if self.loss == "soft":
                # soft target loss
                # print("student logits shape : ", logits.shape)
                # print("teacher logits shape : ", teacher_logits.shape)
                Sy = teacher_logits.shape[2]
                # print("Sy : ", Sy)
                # print("student logits shape : ", logits[:,:,:Sy].shape)
                # print("teacher logits shape : ", teacher_logits.shape)
                loss_soft = self.loss_kl(F.log_softmax(logits[:,:,:Sy]/self.temperature, dim=-1), F.softmax(teacher_logits/self.temperature, dim=-1))
                loss = loss * self.r_target + self.r_soft * loss_soft
                self.log("val/loss_soft", loss_soft)

            if self.embedding:
                # embedding loss
                embedding_x = embedding_x.permute(1, 0, 2) # (B, Sx, E)
                embedding_x = embedding_x.reshape(embedding_x.shape[0],-1) # (B, Sx*E)
                teacher_embedding_x = teacher_embedding_x.permute(1, 0, 2) # (B, Sx, E)
                teacher_embedding_x = teacher_embedding_x.reshape(teacher_embedding_x.shape[0],-1) # (B, Sx*E)
                # print(embedding_x.shape, teacher_embedding_x.shape)
                assert embedding_x.shape == teacher_embedding_x.shape , "embedding_x and teacher_embedding_x must have same shape but got {} and {}".format(embedding_x.shape, teacher_embedding_x.shape)

                y = torch.ones(imgs.shape[0]).cuda()
                loss_embedding = self.loss_cos(embedding_x.cuda(), teacher_embedding_x.cuda(), y)
                # loss_embedding = self.loss_cos(embeding_x, teacher_embedding_x)
                loss = loss + self.r_embedding * loss_embedding
                self.log("val/loss_embedding", loss_embedding)

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

        if self.teacher:
            teacher_preds = self.teacher_model.predict(imgs)
            teacher_test_cer = self.test_cer(teacher_preds, targets)
            self.log("test/teacher_cer", teacher_test_cer)
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
