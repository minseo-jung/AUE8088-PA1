# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting


# [TODO: Optional] Rewrite this class if you want
import torch
from torch import nn
from torchvision.models.alexnet import AlexNet


class MyNetwork(AlexNet):
    def __init__(self, num_classes=200, dropout=0.5):
        super().__init__()

        # 수정된 feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # padding 없이
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            # 사전학습 가중치 로드
            # 1) ImageNet-1k 가중치로 백본 로드 (num_classes=1000)
            self.model = models.get_model(model_name, weights='DEFAULT')
    
            # 2) 백본 종류별로 마지막 레이어 찾아 200-class로 교체
            in_features = None
            if hasattr(self.model, "classifier"):                
                if isinstance(self.model.classifier, nn.Sequential):
                    in_features = self.model.classifier[-1].in_features
                    self.model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:  
                    in_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(self.model, "fc"):                 
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(self.model, "head"):              
                in_features = self.model.head.in_features
                self.model.head = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"모델 {model_name} 의 분류 층을 찾을 수 없습니다.")

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Metric
        self.train_accuracy = MyAccuracy()
        self.train_f1score = MyF1Score(num_classes=num_classes)

        self.val_accuracy = MyAccuracy()
        self.val_f1score = MyF1Score(num_classes=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        # 각 metric 업데이트
        self.train_accuracy.update(scores, y)
        self.train_f1score.update(scores, y)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': self.train_accuracy,
            'f1/train': self.train_f1score,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.val_accuracy.update(scores, y)
        self.val_f1score.update(scores, y)
        self.log_dict({
            'loss/val': loss,
            'accuracy/val': self.val_accuracy,
            'f1/val': self.val_f1score,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
