Knowledge Distillation of image-to-latex models
----------------------------------------------------------------

Inspire by [kingyiusuen/image-to-latex](https://github.com/kingyiusuen/image-to-latex)

# Architecture
```
distill_img2latex
├── distill_img2latex (main folder)
│   ├── conf
│   	├── __config.yaml                   # config file template
│   	├── experiments                     # config file for experiments
│   	    ├── Original.yaml
│   	    ├── Target_Soft.yaml
│   	    ├── Target_Soft_Embde.yaml
│   	    ├── Beamsearch.yaml
│   	    ├── BS_Soft.yaml
│   	    └── BS_Soft_Embed.yaml
│   ├── data                                # data folder
│   	├── ...
│   ├── model                               # model folder
│   	├── __init__.py
│   	├── DistillModel.py                 # DistillModel class
│   	├── student_model.py                # student model class
│   	├── img2latex                       # function from image-to-latex
│   	    ├── data                        # data utils
│   	    ├── lit_model                   # lightning model flow
│   	    └── models                      # model architecture
│   ├── weights                             # weights folder
│       ├── model.ckpt                      # pretrained weights
├── run.py                                  # main file
├── BLEU.py                                 # calculate BLEU score
├── Beamsearch.py                           # Produce beamsearch result dataset
├── requirements.txt                        
└── README.md
```

# Setup
## Data
- available in last repository
- put all data in `data` folder
## Teacher model weight
- available in last repository training result
- put all weight in `weights` folder
## Preprocessing
- Produce Beamsearch predictions
```bash
python Beamsearch.py [teacher_model_path]
```
- Then `new_BS_data.lst` will be generated in `data` folder


# How To Use
## Requirements and Setup
- Python 3.6

Clone the repository to your computer and position your command line inside the repository folder:

Then, create a virtual environment named venv and install required packages:
```bash
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Experiment
```bash
python run.py trainer.gpus=1 data.batch_size=8
```
- Training Result will be generated in `outputs` folder
- Your model weight will be generated in where you set in config `logger.save_dir` 
- Configurations can be modified in `conf/config.yaml` or in command line. See [Hydra's documentation](https://hydra.cc/docs/intro/) to learn more.

- Note that the default configuration is `conf/config.yaml`. To use another configuration, please specify which config file you chose in `run.py`:
```python
@hydra.main(config_path="./conf", config_name="config") 
# specify your configuration path and name here
```

- We provide a config template in `conf/__config.yaml` for you to start with. You can also use `conf/__config.yaml` as a reference to create your own config file.
- And we have all ours experiments' config in `conf/experiments` folder.Please specify `pretrained_weight` and `save_dir` and replace default configuration if your want to reproduce our experiments.

## Config setting
```yaml
seed: 1234

trainer:
  gpus: 1
  overfit_batches: 0.0
  check_val_every_n_epoch: 10
  fast_dev_run: false
  max_epochs: 100
  min_epochs: 1
  num_sanity_val_steps: 0
  auto_lr_find: false
  checkpoint_callback: true

callbacks:
  model_checkpoint:
    save_top_k: 1
    save_weights_only: true
    mode: "min"
    monitor: "val/loss"
    filename: "{epoch}-{val/loss:.2f}-{val/cer:.2f}"
  early_stopping:
    patience: 100
    mode: "min"
    monitor: "val/loss"
    min_delta: 0.001

data:
  batch_size: 8
  num_workers: 4
  pin_memory: false
  BS: False             # IF use beamsearch dataset

lit_model:
  # Optimizer
  lr: 0.001
  weight_decay: 0.0
  # Scheduler
  milestones: [10]
  gamma: 0.5
  # Model
  d_model: 128
  dim_feedforward: 256
  nhead: 4
  dropout: 0.3
  num_decoder_layers: 3
  max_output_len: 500
  # pretrained weight
  teacher: False        # IF need teacher model
  pretrained_weight:    # your pretrained_weight path
  # loss
  loss: "soft"          # "soft" or "hard"
  embedding: False      # True or False
  temperature: 5        # knowledge distillation temperature -> recommanded 3~20
  # ratio of different loss
  r_target : 1.0
  r_soft : 0.0
  r_hard : 0.0
  r_embedding : 0.0
logger:
  project: "Distill-im2latex"
  save_dir : # where you save your model
```

## Evaluation
- BLEU score
```bash
python BLEU.py [your_prediction.txt_path]
```
