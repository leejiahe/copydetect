import os

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__,
                              indicator = [".git", "pyproject.toml"],
                              pythonpath = True,
                              dotenv = True)

from typing import List, Tuple

import hydra
from omegaconf import DictConfig

from torch.utils.data import DataLoader

import faiss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.utils.pylogger import get_pylogger
from src.models.components.matching import (Embeddings,
                                            codec_train,
                                            retrieve_candidate_set,
                                            tabulate_result)
from src.datamodules.disc_datamodule import DISCMatching

log = get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {"cfg": cfg,
                   "datamodule": datamodule,
                   "model": model,
                   "logger": logger,
                   "trainer": trainer,
                   }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    model.set_test = True 
    trainer.validate(model = model, dataloaders = datamodule.test_dataloader(), ckpt_path = cfg.ckpt_path)
    
    h5py_path = os.path.join(model.logging_dir, 'embeddings.h5')
    embeddings = Embeddings()
    embeddings.load_from_h5py(h5py_path)
    embeddings = codec_train(embeddings, score_norm_arg = False)
    
    candidate_set = retrieve_candidate_set(embeddings = embeddings,
                                            k_candidates = model.k_candidates,
                                            global_candidates = False,
                                            metric = faiss.METRIC_L2,
                                            use_gpu = False,
                                            )
    
    matched_dataset = DISCMatching(query_embs = embeddings.query_emb,
                                    refer_embs = embeddings.refer_emb,
                                    candidate_set = candidate_set,
                                    )
    
    matched_loader = DataLoader(matched_dataset,
                                batch_size = datamodule.test_batch_size,
                                num_workers = datamodule.workers,
                                persistent_workers = True,
                                pin_memory = True,
                                shuffle = False,
                                drop_last = False, 
                                )
    
    trainer.test(model = model, dataloaders = matched_loader)
    
    scores_path = os.path.join(model.logging_dir, 'scores.npy')
    gt = datamodule.val_dataset.gt
    
    metrics = tabulate_result(embeddings = embeddings,
                                scores_path = scores_path,
                                gt = gt,
                                alpha = 1.0, 
                                )    

    return metrics


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
