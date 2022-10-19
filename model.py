import torch
import omegaconf
import pytorch_lightning as pl
import thop
from torchsummary import summary
from copy import deepcopy


class Conv1dNet(pl.LightningModule):
    def __init__(
        self, 
        conf: omegaconf.dictconfig.DictConfig
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.NLLLoss()
        self.conf = conf

        in_features=conf.features.n_mels
        n_classes=len(conf.idx_to_keyword)
        activation = getattr(torch.nn, conf.model.activation)()
        features = in_features
        module_list = []
        for kernel_size, stride, channels in zip(conf.model.kernels, conf.model.strides, conf.model.channels):
            module_list.extend([
                torch.nn.Conv1d(
                    in_channels=features, out_channels=channels, kernel_size=kernel_size,
                    stride=stride, 
                    groups=channels
                ),
                activation,
                torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1),
                torch.nn.BatchNorm1d(num_features=channels),
                activation,
                torch.nn.MaxPool1d(kernel_size=stride)
            ])
            features = channels
        module_list.extend([
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            # torch.nn.Dropout(p=0.5), # <--- changed

            # torch.nn.Linear(channels, conf.model.hidden_size), 
            # torch.nn.BatchNorm1d(num_features=conf.model.hidden_size), # <--- changed
            # torch.nn.Dropout(p=0.25), # <--- changed
            # activation, # <--- changed

            # torch.nn.Linear(conf.model.hidden_size, conf.model.hidden_size), # <--- changed
            # torch.nn.BatchNorm1d(num_features=conf.model.hidden_size),
            # activation,

            # torch.nn.Linear(conf.model.hidden_size, n_classes),
            torch.nn.Linear(channels, n_classes),
            torch.nn.LogSoftmax(-1)
        ])
        self.backbone = torch.nn.Sequential(*deepcopy(module_list))

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logprobs = self.backbone(inputs)
        preds = logprobs.argmax(1)
        loss = self.criterion(logprobs, labels)
        acc = (preds == labels).sum() / inputs.shape[0]

        logs = {'train_loss': loss.detach().cpu().numpy(), 'train_acccuracy': acc.detach().cpu().numpy()}
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logprobs = self.backbone(inputs)
        preds = logprobs.argmax(1)
        loss = self.criterion(logprobs, labels)
        acc = (preds == labels).sum() / inputs.shape[0]
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False) # 
        self.log('val_acc', acc, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return {'val_loss': loss, 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        print('\nVAL Loss: {}\nVAL Acc: {}\n'.format(
            round(float(avg_loss), 3),
            round(float(avg_acc), 3)
        ))
        return {'val_loss': avg_loss, 'log': logs}

    def on_fit_start(self):
        sample_shape = (
            1, 
            self.conf.features.n_mels, 
            self.conf.sample_rate // self.conf.features.hop_length + 1
        )
        sample_inputs = torch.randn(
            *sample_shape
        )
        print(summary(deepcopy(self), sample_shape[1:]))
        macs, params = thop.profile(
            deepcopy(self), 
            inputs=(sample_inputs.cuda(),)
        )
        report = '''
        #################################
        
                Total MACs: {}
                Total params: {}
        
        #################################
        '''.format(macs, params)
        print(report)
        assert macs <= 10**6, 'Must be less 10**6, got: {}'.format(macs)
        assert params <= 10**4, 'Must be less 10**4, got: {}'.format(params)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.conf.optim.lr 
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            threshold=1e-2, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=0, 
            eps=1e-08, 
            verbose=True
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        return [optimizer], [lr_dict]
