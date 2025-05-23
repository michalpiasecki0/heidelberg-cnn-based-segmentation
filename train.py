import hydra
import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger


def train(cfg: DictConfig) -> None:
    model_lt = hydra.utils.instantiate(cfg.model)

    dataset_train = hydra.utils.instantiate(cfg.dataset.train)
    dataset_val = hydra.utils.instantiate(cfg.dataset.val)

    dataloader = DataLoader(dataset=dataset_train, batch_size=cfg.batch_size)
    val_dataloader = DataLoader(dataset=dataset_val, batch_size=cfg.batch_size)

    wand_logger = WandbLogger(
        project=cfg.tracker.project_name,
        name=cfg.tracker.task_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    trainer = pl.Trainer(max_epochs=cfg.max_epochs, logger=wand_logger)
    trainer.fit(model_lt, train_dataloaders=dataloader, val_dataloaders=val_dataloader)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # todo: uncomment this when using clearml
    #task = Task.init(project_name=cfg.tracker.project_name, task_name=cfg.tracker.task_name)

    OmegaConf.resolve(cfg)
    train(cfg)

    # todo: uncomment this when using clearml
    #task.mark_completed()


if __name__ == "__main__":
    main()
