import os
from typing import Tuple, List, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.apply_func import move_data_to_device
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

from pytorch_metric_learning.losses import NTXentLoss, MultiSimilarityLoss, CrossBatchMemory
from pytorch_metric_learning.utils.distributed import DistributedLossWrapper
from pytorch_metric_learning.miners import MultiSimilarityMiner

from torchmetrics.classification import Accuracy

from src.utils import get_region_vector, get_local_vector
from src.datamodules.disc_datamodule import BatchRecords
from src.models.components.matching import retrieve_candidate_set, remove_duplicates, Embeddings
import src.models.components.vision_transformer as vits
from src.models.components.copyhead import CopyHead
from src.models.components.xbm import XBM
from src.losses.koleo import NTXentKoLeoLoss
from src.utils.metrics import PredictedMatch, evaluate
from src.utils.distributed_utils import gather_across_gpu



class CopyDetectorModule(LightningModule):
    def __init__(self,
                 dino_model:str = 'vit',
                 dino_size:'str' = 'base',
                 patch_size:int = 16,
                 temperature:float = 0.05,
                 entropy_weight:float = 10,
                 global_embedding_size:int = 512,
                 region_embedding_size:int = 512,
                 cross_batch_memory_size:int = 4096,
                 lr:float = 0.0005,
                 warmup_start_lr: float = 0.01,
                 weight_decay:float = 0.000001,
                 warmup_epochs: int = 5,
                 beta1:float = 0.9,
                 beta2:float = 0.999,
                 lambda1:float = 1,
                 lambda2:float = 1,
                 lambda3:float = 1,
                 logging_dir = os.getcwd(),
                 k_candidates:int = 10,
                ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained DINO model from torch hub
        self.model = vits.__dict__[f'{dino_model}_{dino_size}'](patch_size = patch_size, num_classes = 0)
        state_dict = torch.hub.load_state_dict_from_url(url = f"https://dl.fbaipublicfiles.com/dino/dino_{dino_model}{dino_size}{patch_size}_pretrain/dino_{dino_model}{dino_size}{patch_size}_pretrain.pth",
                                                        map_location = 'cpu')
        self.model.load_state_dict(state_dict, strict = True)
        
        self.dim = self.model.embed_dim
        self.patch_h = self.model.patch_embed.img_size // self.model.patch_embed.patch_size
        
        # Projection head
        self.projection = nn.Sequential(nn.Linear(self.model.embed_dim,
                                                  global_embedding_size),
                                        nn.LayerNorm(global_embedding_size))
        
        # Whitening Layer
        self.whitening = nn.Sequential(nn.Linear(self.model.embed_dim, region_embedding_size, bias = True),
                                       nn.LayerNorm(region_embedding_size))
        
        # Global distributed NTXent Loss
        self.global_loss = NTXentLoss(temperature = temperature)
        
        #global_loss = CrossBatchMemory(NTXentLoss(temperature = temperature),
        #                               embedding_size = global_embedding_size,
        #                               memory_size = cross_batch_memory_size)
        #self.global_loss = DistributedLossWrapper(global_loss)
        """                    
        self.global_loss = NTXentKoLeoLoss(temperature = temperature,
                                           entropy_weight = entropy_weight)
        # Cross-batch memory
        self.projected_xbm = XBM(embedding_size = global_embedding_size,
                                 memory_size = cross_batch_memory_size)
        """ 
        
        # Region distributed Multi-Similarity Loss
        """
        self.miner = MultiSimilarityMiner() # Multi-Similarity Miner
        region_loss = CrossBatchMemory(MultiSimilarityLoss(),
                                       embedding_size = region_embedding_size,
                                       memory_size = cross_batch_memory_size,
                                       miner = self.miner)
        self.region_loss = DistributedLossWrapper(region_loss)
        """
        
        # CopyDetector
        """
        #self.copyhead = CopyHead()
        #self.bce_loss = nn.BCELoss()
        """
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.beta = (beta1, beta2)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.k_candidates = k_candidates
        
        self.n_repeat_aug = 1
        
        self.accuracy = Accuracy()
        
        # For saving embedding during test phase
        self.set_test = False
        self.h5py_path = os.path.join(logging_dir, 'embeddings.h5')
        self.scores_path = os.path.join(logging_dir, 'scores.npy')
        
    """
    def configure_sharded_model(self) -> None:
        self.model = torch.hub.load('facebookresearch/dino:main', self.dino_pretrain)
        self.copydetector = CopyDetector()
    """


    def configure_optimizers(self):
        """
        optimizer = DeepSpeedCPUAdam(self.parameters(),
                                     lr = self.lr,
                                     betas = self.beta,
                                     weight_decay = self.weight_decay)
        
        scheduler = {'scheduler': LinearWarmupCosineAnnealingLR(optimizer = optimizer,
                                                                warmup_start_lr = self.warmup_start_lr,
                                                                warmup_epochs = self.warmup_epochs,
                                                                max_epochs = self.trainer.max_steps),
                     'interval': 'epoch',
                     'frequency': 1}
        
        return [optimizer], [scheduler]
        """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = self.lr,
                                     betas = self.beta,
                                     weight_decay = self.weight_decay)
        

        return optimizer
    
    
    def forward(self,
                img: List[torch.Tensor],
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        cls_token = deepspeed.checkpointing.checkpoint(self.model, img)
        attn = deepspeed.checkpointing.checkpoint(self.model.get_last_selfattention, img)
        embeddings = deepspeed.checkpointing.checkpoint(self.model.get_intermediate_layers, img)[0]
        """
        cls_token = self.model(img)
        attn = self.model.get_last_selfattention(img)
        embeddings = self.model.get_intermediate_layers(img)[0]
        
        return cls_token, attn, embeddings
    
    
    """
    def forward_copy(self,
                     emb_q: List[torch.Tensor],
                     emb_r: List[torch.Tensor],
                     ) -> torch.Tensor:
        
        copy_cls = deepspeed.checkpointing.checkpoint(self.copydetect, (emb_q, emb_r))
        return copy_cls
    """
    
    
    def feature_aggregate(self,
                          img: List[torch.Tensor],
                         ) -> List[torch.Tensor]:
        cls_token, att, embeddings = self(img)
        
        # Get projected CLS token
        projected = self.projection(cls_token)
        #projected = self.projection(get_local_vector(embeddings))
        
        """
        # Get local features
        feats = get_local_vector(embeddings)
        feats = self.whitening(feats)
        feat_agg = torch.cat([projected, feats], dim = 1)
        #return feat_agg
        """
        
        """
        # Get salient regions
        region = get_region_vector(embeddings, att)
        region = self.whitening(region)
        feat_agg = torch.cat([projected, region], dim = 1)
        return feat_agg
        """
        return projected
    
    
    def training_step(self,
                      batch: Any,
                      batch_idx: int,
                     ) -> torch.Tensor:
        
        img, label = batch['image0'], batch['instance_id']
        for i in range(self.n_repeat_aug):
            img = torch.cat((img, batch[f'image{i+1}']))
            label = torch.cat((label, batch['instance_id']))
        cls_token, att, embeddings = self(img)
        
        """
        img, label = batch['image0'], batch['instance_id']
        cls_token, att, embeddings = self(img)
        
        for i in range(self.n_repeat_aug):
            cls_token_, att_, embeddings_ = self(batch[f'image{i+1}'])
            
            cls_token = torch.cat((cls_token, cls_token_))
            att = torch.cat((att, att_))
            embeddings = torch.cat((embeddings, embeddings_))
            
            label = torch.cat((label, batch['instance_id']))
        """
        
        # Get CLS token
        # Global contrastive loss is computed at training_step_end 
        projected = self.projection(cls_token) # Projection of CLS token
        global_loss = self.global_loss(projected, label)
    
        #gathered_projected, gathered_label = gather_across_gpu(projected, label)
        #self.projected_xbm.enqueue(gathered_projected.detach(), gathered_label.detach())
        
        #xbm_projected, xbm_label = self.projected_xbm.get()

        #global_loss, loss_stats = self.global_loss(projected,
        #                                           label,
        #                                           gathered_projected,
        #                                           gathered_label,
        #                                           rank = self.global_rank)
        
        #global_loss, loss_stats = self.global_loss(projected,
        #                                           label,
        #                                           xbm_projected.to(projected),
        #                                           xbm_label.to(label),
        #                                           rank = self.global_rank)
        
        self.log_dict({'contrastive_loss':global_loss},
                      on_step = True,
                      on_epoch = True,
                      sync_dist = True,
                      rank_zero_only = True)
        """
        # Get salient regions
        region = get_region_vector(embeddings, att)
        region = self.whitening(region)
        
        # Get hard negative from miner
        indice_tuples = self.miner(region, label)
        
        # Compute multi-similarity loss
        region_loss = self.region_loss(region, label, indice_tuples)
        self.log_dict({'region_loss':region_loss},
                        on_step = True,
                        on_epoch = True,
                        sync_dist = True,
                        rank_zero_only = True)
        """
        
        
        region_loss = 0
        
        """
        # Copy Detector
        batch_size = batch['image0'].shape[0]
        indices = torch.arange(batch_size)
        refer_emb = embeddings[:batch_size]

        for i in range(self.n_repeat_aug):
            query_emb = embeddings[batch_size*(i+1):batch_size*(i+2)]
            
            if i == 0:
                pred = self.copyhead(query_emb, refer_emb)
            else:
                pred_ = self.copyhead(query_emb, refer_emb)
                pred = torch.cat((pred, pred_))
                
            label = torch.ones(batch_size).to(query_emb)

            shuffled_indices = torch.randperm(batch_size) # shuffled the indices    
            shuffled_query = query_emb[shuffled_indices]
            pred_ = self.copyhead(shuffled_query, refer_emb)
            pred = torch.cat((pred, pred_))

            shuffled_label = (indices == shuffled_indices).float().to(query_emb)

            label = torch.cat((label, shuffled_label))
        bce_loss = self.bce_loss(pred, label.unsqueeze(1))
        self.log_dict({'bce_loss':bce_loss},
                        on_step = True,
                        on_epoch = True,
                        sync_dist = True,
                        rank_zero_only = True)
                      
        acc = self.accuracy(pred, label.int())
        self.log({'CopyHead accuracy':acc},
                 on_step = True,
                 on_epoch = True,
                 rank_zero_only = True,
                 )
        """
        
        bce_loss = 0
        
        """
        for i in range(n_repeat_aug):
            #_, _, query_emb = self(batch[f'image{i+1}'])


            label = torch.ones(batch_size).to(query_emb)

            shuffled_indices = torch.randperm(batch_size) # shuffled the indices
            #shuffled_image = query_emb[shuffled_indices] 

            shuffled_query = query_emb[shuffled_indices]
            shuffled_label = (indices == shuffled_indices).float().to(query_emb)


            shuffled_indices = torch.randperm(batch_size * 2)

            refer_emb_ = refer_emb[shuffled_indices]

            query_emb = torch.cat((query_emb, shuffled_query))
            query_emb = query_emb[shuffled_indices]

            pred = copyhead(query_emb[:batch_size], refer_emb_[:batch_size])
            pred_ = copyhead(query_emb[batch_size:], refer_emb_[batch_size:])
            pred = torch.cat((pred, pred_))

            label = torch.cat((label, shuffled_label))
            label = label[shuffled_indices].unsqueeze(1)
            
        bce_loss = self.bce_loss(pred, label.unsqueeze(1))
        """
        
        loss = global_loss * self.lambda1 + \
               region_loss * self.lambda2 + \
               bce_loss * self.lambda3
        
        self.log_dict({'total_loss':loss},
                      on_step = True,
                      on_epoch = True,
                      sync_dist = True,
                      rank_zero_only = True)
        return loss
        
        
    def validation_step(self,
                        batch: Any,
                        batch_idx: int,
                       ) -> List[torch.Tensor]:
        
        batch.feat_agg = self.feature_aggregate(batch.image)
        batch.image = self.model.get_intermediate_layers(batch.image)[0]
        
        return batch
    
    
    def validation_epoch_end(self,
                             outputs: Any,
                            ) -> List[torch.Tensor]:

        # We gather all the values from all the steps
        batchrecords = self._gather(outputs)
        
        # Only perform on zero rank
        if torch.distributed.get_rank() == 0:
            embeddings = Embeddings()
            embeddings.load_from_batchrecords(batchrecords)
            
            if self.set_test:
                embeddings.save_to_h5py(self.h5py_path)
            else:
                print('-----Validation-----')
                candidate_set = retrieve_candidate_set(embeddings,
                                                       self.k_candidates,
                                                       use_gpu = False)
                
                preds = []
                for q_idx, r_idx, dis in candidate_set:
                    q_name = embeddings.query_name[q_idx]
                    r_name = embeddings.refer_name[r_idx]
                    score = dis
                    #score = cosine_similarity(embeddings.query_feat[q_idx], embeddings.refer_feat[r_idx])
                    #q_emb = embeddings.query_emb[q_idx]
                    #r_emb = embeddings.refer_emb[r_idx]

                    #score = self.copydetector(q_emb, r_emb)
                    
                    preds.append(PredictedMatch(q_name, r_name, score))
                
                #predictions = remove_duplicates(preds)
                predictions = remove_duplicates(preds, ascending = True)
                
                gt = self.trainer.datamodule.val_dataset.gt
                metrics = evaluate(gt, predictions)
                metrics = {k: 0.0 if v is None else v for (k, v) in metrics.items()}
                
                self.log_dict(metrics,
                              on_step = False,
                              on_epoch = True,
                              rank_zero_only = True)
        
        
    def test_step(self,
                  batch: Any,
                  batch_idx: int,
                 ):
        q_emb, r_emb, q_idx, r_idx = batch
        score = F.cosine_similarity(q_emb,r_emb)
        #score = self.copydetector(q_emb, r_emb)
        
        return score, q_idx, r_idx
    
    
    def test_epoch_end(self,
                      outputs: Any):
        
        scores, q_idxs, r_idxs = [], [], []
        
        for output in outputs:
            score, q_idx, r_idx = output
            scores.append(score)
            q_idxs.append(q_idx)
            r_idxs.append(r_idx)
            
        scores = self._reshape(scores, self.device)
        q_idxs = self._reshape(q_idxs, self.device)
        r_idxs = self._reshape(r_idxs, self.device)
        
        # Only perform on zero rank
        if torch.distributed.get_rank() == 0:
            with open(self.scores_path, 'wb') as f:
                np.save(f, scores)
                np.save(f, q_idxs)
                np.save(f, r_idxs)
    
    
    def _gather(self, batches):
        batchrecord = BatchRecords()
        
        for batch in batches:
            batchrecord.add_batch(batch)

        batches = self.all_gather(move_data_to_device(batchrecord, self.device))
        batches.numplify()

        return batches
    
    
    def _reshape(self, it, device):
        it = torch.hstack(it)
        it = self.all_gather(move_data_to_device(it, device))
        it = it.reshape(-1)
        return it.cpu().numpy()