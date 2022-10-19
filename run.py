from model import Conv1dNet
from dataset import SpotterDataManager , SpotterDataset
from conf import Model, Optim, Features, Augmentations, ExpConfig, PretrainConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, GradientAccumulationScheduler
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

import omegaconf

RUN_NAME = 'Normalized-V4'


if __name__ == '__main__':
    conf = omegaconf.OmegaConf.structured(ExpConfig)
    dm = SpotterDataManager(
        conf=conf, 
        num_workers=8,
        manifest_path='/home/eugeny/Datasets/keyword-spotting/train/train/manifest.csv'
        # manifest_path='/home/eugeny/soundmipt/hw2/google-speech-commands/SpeechCommands/speech_commands_v0.02/manifest.csv'
    )
    model = Conv1dNet(conf=conf)

    logger = TensorBoardLogger('lightning_logs', name=RUN_NAME)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor='val_loss',
        mode='min',
        dirpath='runs/{}/'.format(RUN_NAME),
        filename=RUN_NAME + '-{epoch:02d}-{step:d}-{val_loss:.4f}',
        save_last=False,
        verbose=True,
        every_n_epochs=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=50,
        verbose=True,
        strict=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
        refresh_rate=1,
    )

    trainer = pl.Trainer(
        gpus=1, 
        precision=32,
        max_epochs=2000,
        min_epochs=1,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar, lr_monitor, swa_callback],
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1
    )

    trainer.fit(model, dm)
