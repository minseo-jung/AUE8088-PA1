import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 128
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 100
OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4}
SCHEDULER_PARAMS    = {'type': 'CosineAnnealingLR', 'T_max': NUM_EPOCHS}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'efficientnet_b7'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
WANDB_NAME         += f'feature_extractor_added'
