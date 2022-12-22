import os
import time

import pyrootutils

root = pyrootutils.setup_root(search_from = __file__,
                              indicator = [".git", "pyproject.toml"],
                              pythonpath = True,
                              dotenv = True,
                              )

from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig



import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import faiss

from src.utils import utils, read_ground_truth
from src.utils.pylogger import get_pylogger
from src.models.components.matching import get_candidate_set, tabulate_result


log = get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers = True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer,
                                               callbacks = callbacks,
                                               logger = logger)

    object_dict = {"cfg": cfg,
                   "datamodule": datamodule,
                   "model": model,
                   "callbacks": callbacks,
                   "logger": logger,
                   "trainer": trainer,
                   }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model = model, datamodule = datamodule, ckpt_path = cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            
        """
        else:
            log.info("Converting Deepspeed checkpoint file...")
            lightning_checkpt = os.path.join(cfg.model.logging_dir, 'lightning_model.pt')
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, lightning_checkpt)
            model.load_from_checkpoint(lightning_checkpt)
            log.info("Finish converting Deepspeed checkpoint and model loaded with checkpoint")
        """
        
        log.info(f"Best ckpt path: {ckpt_path}")
        # Retrieve embeddings from patchify test set using the validation loop
        
        log.info("Retrieving embeddings from test dataset")
        model.set_test = True 
        trainer.validate(model = model, dataloaders = datamodule.test_dataloader(), ckpt_path = ckpt_path)
        print("-----------------Pause for 10s-----------------")
        time.sleep(10)
        #trainer.validate(model = model, dataloaders = datamodule.test_dataloader())
        log.info("Retrieved embeddings from test dataset")
        
        candidate_set, embeddings, matched_loader = get_candidate_set(logging_dir = cfg.model.logging_dir, 
                                                                      k_candidates = model.k_candidates,
                                                                      batch_size = cfg.datamodule.test_batch_size,
                                                                      workers = cfg.datamodule.workers,
                                                                      )
        log.info(f"Candidate set retrieved with {len(candidate_set)} candidates.")
    
        trainer.test(model = model, dataloaders = matched_loader, ckpt_path = ckpt_path)
        #trainer.test(model = model, dataloaders = matched_loader)
        
        scores_path = os.path.join(cfg.model.logging_dir, 'scores.npy')
        gt_path = cfg.datamodule.val_gt_path
        gt = read_ground_truth(gt_path)
        
        
        metrics = tabulate_result(embeddings = embeddings,
                                  scores_path = scores_path,
                                  gt = gt,
                                  alpha = 1.0, 
                                  )
        

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict = metric_dict,
                                          metric_name = cfg.get("optimized_metric")
                                          )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
