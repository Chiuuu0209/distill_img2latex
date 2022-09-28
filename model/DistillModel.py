from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from image_to_latex.data.utils import Tokenizer
from image_to_latex.models import ResNetTransformer
from image_to_latex.lit_models.metrics import CharacterErrorRate
from image_to_latex.lit_models import LitResNetTransformer



class DistillModel(LightningModule):
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
        pretrained_weight = Path(__file__).resolve().parents[1] / "weights" / "model.ckpt",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        # vocab_file = Path(__file__).resolve().parents[0] / "image-to-latex" / "data" / "vocab.json"
        vocab_file = Path(__file__).resolve().parents[0] / "img2latex" / "data" / "vocab.json"
        # print(vocab_file)
        self.tokenizer = Tokenizer.load(vocab_file)
        self.pretrained_weight = pretrained_weight
        # self.pretrained_weight = Path(__file__).resolve().parents[1] / "weights" / "model.ckpt"
        self.model = ResNetTransformer(
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
        for load distill model
        TODO: freeze pretrained model
        """
        
        teacher_model = LitResNetTransformer.load_from_checkpoint(self.pretrained_weight)
        for (name,para) in teacher_model.named_parameters():
            para.requires_grad = False
        self.teacher_model = teacher_model.model

        # print("type student : ", type(self.model), "teahcer : ", type(self.teacher_model),type(self.model)==type(self.teacher_model))
        # self.model = None

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)
        self.loss_emb = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.loss_cos = torch.nn.CosineEmbeddingLoss(margin=0.05, reduction='mean')
        self.val_cer = CharacterErrorRate(self.tokenizer.ignore_indices)
        self.test_cer = CharacterErrorRate(self.tokenizer.ignore_indices)

    def training_step(self, batch, batch_idx):
        """
        ResNetTransformer forward
        
        Args:
            x: (B, _E, _H, _W)
            y: (B, Sy) with elements in (0, num_classes - 1)

        Returns:
            (B, num_classes, Sy) logits

        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, num_classes)
        output = output.permute(1, 2, 0)  # (B, num_classes, Sy)
        """
        """
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        """
        imgs, targets = batch

        # train_model 
        embedding_x = self.model.encode(imgs) # (Sx, B, E)
        logits = self.model.decode(targets[:, :-1],embedding_x) # (Sy, B, num_classes)
        logits = logits.permute(1, 2, 0)  # (B, num_classes, Sy)

        # freeze pretrined model
        with torch.no_grad():
            teacher_embedding_x = self.teacher_model.encode(imgs) # (Sx, B, E)
            teacher_logits = self.teacher_model.decode(targets[:, :-1],teacher_embedding_x) # (Sy, B, num_classes)
            teacher_logits = teacher_logits.permute(1, 2, 0) # (B, num_classes, Sy)

        # transform embeding to proper shape
        embedding_x = embedding_x.permute(1, 0, 2) # (B, Sx, E)
        embedding_x = embedding_x.reshape(embedding_x.shape[0],-1) # (B, Sx*E)
        teacher_embedding_x = teacher_embedding_x.permute(1, 0, 2) # (B, Sx, E)
        teacher_embedding_x = teacher_embedding_x.reshape(teacher_embedding_x.shape[0],-1) # (B, Sx*E)
        
        # Loss with teacher embeding        
        # loss_embedding = self.loss_emb(embedding_x, teacher_embedding_x)# (B, Sx*E)
        # y = 2*torch.empty(batch.shape[0]).random_(2) - 1
        y = torch.ones(imgs.shape[0]).cuda()
        loss_embedding = self.loss_cos(embedding_x.cuda(), teacher_embedding_x.cuda(), y)


        # loss with target
        loss_target = self.loss_fn(logits, targets[:, 1:]) # (B, num_classes, Sy)

        # loss with teacher logits
        logits = logits.reshape(logits.shape[0],-1) # (B, num_classes*Sy)
        teacher_logits = teacher_logits.reshape(teacher_logits.shape[0],-1) # (B, num_classes*Sy)
        # loss_logits_teacher = self.loss_emb(logits, teacher_logits)# (B, num_classes*Sy)
        loss_logits_teacher = self.loss_cos(logits.cuda(), teacher_logits.cuda(), y)

        # mean every batch
        # loss_embedding = loss_embedding.mean()
        # loss_logits_teacher = loss_logits_teacher.mean()
        
        # loss = loss_target - loss_embedding - loss_logits_teacher
        loss = loss_target + loss_embedding + loss_logits_teacher
        
        # log loss
        self.log("train/loss_target", loss_target)
        self.log("train/loss_logits_teacher", loss_logits_teacher)
        self.log("train/loss_embedding", loss_embedding)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = self.model.predict(imgs)
        val_cer = self.val_cer(preds, targets)
        self.log("val/cer", val_cer)

        """
        
        imgs, targets = batch
        # model output 
        embedding_x = self.model.encode(imgs) # (Sx, B, E)
        logits = self.model.decode(targets[:, :-1],embedding_x) # (Sy, B, num_classes)
        logits = logits.permute(1, 2, 0)  # (B, num_classes, Sy)

        # teacher output
        with torch.no_grad():
            teacher_embedding_x = self.teacher_model.encode(imgs) # (Sx, B, E)
            teacher_logits = self.teacher_model.decode(targets[:, :-1],teacher_embedding_x) # (Sy, B, num_classes)
            teacher_logits = teacher_logits.permute(1, 2, 0) # (B, num_classes, Sy)
        
        # transform embeding to proper shape
        embedding_x = embedding_x.permute(1, 0, 2) # (B, Sx, E)
        embedding_x = embedding_x.reshape(embedding_x.shape[0],-1) # (B, Sx*E)
        teacher_embedding_x = teacher_embedding_x.permute(1, 0, 2) # (B, Sx, E)
        teacher_embedding_x = teacher_embedding_x.reshape(teacher_embedding_x.shape[0],-1) # (B, Sx*E)
        
        # Loss with teacher embeding        
        # loss_embedding = self.loss_emb(embedding_x, teacher_embedding_x)# (B, Sx*E)
        y = torch.ones(imgs.shape[0]).cuda()
        loss_embedding = self.loss_cos(embedding_x.cuda(), teacher_embedding_x.cuda(), y)
        
        # loss with target
        loss_target = self.loss_fn(logits, targets[:, 1:]) # (B, num_classes, Sy)

        # loss with teacher logits
        logits = logits.reshape(logits.shape[0],-1) # (B, num_classes*Sy)
        teacher_logits = teacher_logits.reshape(teacher_logits.shape[0],-1) # (B, num_classes*Sy)
        # loss_logits_teacher = self.loss_emb(logits, teacher_logits)# (B, num_classes*Sy)
        loss_logits_teacher = self.loss_cos(logits.cuda(), teacher_logits.cuda(), y)

        # mean every batch
        # loss_embedding = loss_embedding.mean()
        # loss_logits_teacher = loss_logits_teacher.mean()

        # loss = loss_target - loss_embedding - loss_logits_teacher
        loss = loss_target + loss_embedding + loss_logits_teacher

        # log loss
        self.log("val/loss_target", loss_target, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_logits_teacher", loss_logits_teacher, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_embedding", loss_embedding, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # predict the target
        preds = self.model.predict(imgs)
        val_cer = self.val_cer(preds, targets)
        self.log("val/cer", val_cer)


    def test_step(self, batch, batch_idx):
        """
        imgs, targets = batch
        preds = self.model.predict(imgs)
        test_cer = self.test_cer(preds, targets)
        self.log("test/cer", test_cer)
        """
        
        imgs, targets = batch
        preds = self.model.predict(imgs)
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
