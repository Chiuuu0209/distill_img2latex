from argparse import Namespace
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

# from image_to_latex.data import Im2Latex
# from image_to_latex.lit_models import LitResNetTransformer

from model import Im2Latex
from model.DistillModel import DistillModel
import os
# os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = Im2Latex(**cfg.data)
    datamodule.setup()

    # lit_model = LitResNetTransformer(**cfg.lit_model)
    lit_model = DistillModel(**cfg.lit_model)
    # lit_model = DistillModel.load_from_checkpoint("/home/chris/git/Distill/outputs/2022-09-22/17-02-15/wandb/latest-run/files/Distill-im2latex/kp4xwxk7/checkpoints/epoch=13-val/loss=-1.54-val/cer=0.17.ckpt")

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    logger: Optional[WandbLogger] = None
    if cfg.logger:
        logger = WandbLogger(**cfg.logger)

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
