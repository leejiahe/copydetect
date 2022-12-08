from typing import Dict
from dataclasses import dataclass, field, replace
import h5py

import numpy as np
import pandas as pd

import torch.nn as nn

import faiss

from src.utils.matching import search_with_capped_res
from src.utils.metrics import PredictedMatch, evaluate



SPLIT_REFER = 0
SPLIT_QUERY = 1
SPLIT_TRAIN = 2



@dataclass
class Embeddings:
    query_emb = None
    query_name = None
    query_feat = None
    query_patch = None
    refer_emb = None
    refer_name = None
    refer_feat = None    
    refer_patch = None
    train_feat = None
    
    def load_from_batchrecords(self, batchrecords):
        query_mask = batchrecords.split == SPLIT_QUERY

        if len(query_mask) > 0:
            query_ids = batchrecords.image_num[query_mask]
            self.query_name = ["Q%05d" % i for i in query_ids]

            self.query_patch = batchrecords.patch[query_mask]
            self.query_feat = batchrecords.feat_agg[query_mask, :]
            self.query_emb = batchrecords.image[query_mask, :]

        ref_mask = batchrecords.split == SPLIT_REFER
        if len(ref_mask) > 0:
            ref_ids = batchrecords.image_num[ref_mask]
            self.refer_name = ["R%06d" % i for i in ref_ids]
            
            self.refer_patch = batchrecords.patch[ref_mask]
            self.refer_feat = batchrecords.feat_agg[ref_mask, :]
            self.refer_emb = batchrecords.image[ref_mask, :]
                
        train_mask = batchrecords.split == SPLIT_TRAIN
        if len(train_mask) > 0:
            train_mask = batchrecords.split == SPLIT_TRAIN
            self.train_feat = batchrecords.feat_agg[train_mask, :]
    
    def sort_by_name(self, names):
        name_arr = np.array([bytes(name, "ascii") for name in names])
        index = name_arr.argsort()
        return name_arr, index 
    
    def save_to_h5py(self, h5py_path):
        # Sort the index by name. Evaluation script prefer sorted list.
        query_name, query_idx = self.sort_by_name(self.query_name)
        refer_name, refer_idx = self.sort_by_name(self.query_name)
        
        with h5py.File(h5py_path, 'w') as f:
            f.create_dataset('query_feat', data = self.query_feat[query_idx].astype('float32'))
            f.create_dataset('query_name', data = query_name)
            f.create_dataset('query_emb', data = self.query_emb[query_idx].astype('float32'))
            f.create_dataset('query_patch', data = self.query_patch[query_idx].astype('float32'))

            f.create_dataset('refer_feat', data = self.refer_feat[refer_idx].astype('float32'))
            f.create_dataset('refer_name', data = refer_name)
            f.create_dataset('refer_emb', data = self.refer_emb[refer_idx].astype('float32'))
            f.create_dataset('refer_patch', data = self.refer_patch[refer_idx].astype('float32'))

            f.create_dataset('train_feat', data = self.train_feat)

    def load_from_h5py(self, h5py_path):
        with h5py.File(h5py_path, 'r') as f:
            self.query_feat = f.get('query_feat')[:]
            self.query_name = f.get('query_name')[:]
            self.query_emb = f.get('query_emb')[:]
            self.query_patch = f.get('query_patch')[:]

            self.refer_feat = f.get('refer_feat')[:]
            self.refer_name = f.get('refer_name')[:]
            self.refer_emb = f.get('refer_emb')[:]
            self.refer_patch = f.get('refer_patch')[:]

            self.train_feat = f.get('train_feat')[:]
    
    @property
    def dims(self):
        return self.train_feat.shape[1]
    
    @property
    def query_len(self):
        return self.query_feat.shape[0]
    
    def project(self, codec_index, codec_str):
        for name, emb in zip(['train_feat', 'query_feat', 'refer_feat'],
                             [self.train_feat, self.query_feat, self.refer_feat]):
            projected = codec_index.sa_encode(emb)
            projected = np.frombuffer(projected, dtype = np.float32).reshape(emb.shape[0], -1)
        
            if not np.isfinite(projected).all():
                raise ValueError(f"Projection to {codec_str} resulted in non-finite values")
        
            setattr(self, name, projected)
            
            
            
def remove_duplicates(pred, ascending = False):
    df = pd.DataFrame(pred, columns = ['query', 'db', 'score'])
    df.sort_values(by = 'score', ascending = ascending, inplace = True)
    
    df.drop_duplicates(subset = ['query', 'db'], keep = 'first', inplace = True)

    predictions = []
    for _, row in df.iterrows():
        predictions.append(PredictedMatch(row['query'], row['db'], row['score']))
        
    return predictions



def retrieve_candidate_set(embeddings,
                           k_candidates:int = 10,
                           global_candidates:bool = False,
                           metric = faiss.METRIC_L2,
                           use_gpu:bool = False,
                          ) -> Dict:

    if global_candidates:
        lims, dis, ids = search_with_capped_res(xq = embeddings.query_feat,
                                                xb = embeddings.refer_feat,
                                                num_results = k_candidates * embeddings.query_len,
                                                metric = metric)
        if metric == faiss.METRIC_L2:
            dis = -dis # use negated distances as scores

        candidate_set = [(i, ids[j], dis[j])  # query idx, ref idx, distance
                         for i in range(embeddings.query_len) 
                         for j in range(lims[i], lims[i+1])
                        ]

    else:
        d = embeddings.query_feat.shape[1]
            
        index = faiss.IndexFlat(d, metric)
        index.add(embeddings.refer_feat)
        
        if use_gpu:
            co = faiss.GpuClonerOptions()
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index, co)
            
        D, I = index.search(embeddings.query_feat, k = k_candidates)

        if metric == faiss.METRIC_L2:
            D = -D # use negated distances as scores
        
        candidate_set = [(i, I[i,j], D[i,j])  # query idx, ref idx, distance
                         for i in range(embeddings.query_len) 
                         for j in range(k_candidates)
                        ]
        
    return candidate_set


def codec_train(embeddings,
                score_norm_arg:bool = False,
                codec_arg:str = 'PCAW64,L2norm,Flat',
                beta:float = 1.0,
                start_idx:int = 0,
                end_idx:int = 2,
                ):

    print("Indexing codec")
    codec = faiss.index_factory(embeddings.dims, codec_arg)

    # Train codec
    codec.train(embeddings.train_feat)
    print("Finish indexing codec")

    # Project
    print("Projecting codec")
    embeddings.project(codec, codec_arg)
    print("Finish codec projection")
    
    # Score norm
    if score_norm_arg:
        print("Normalizing codec")
        embeddings = score_norm(embeddings, beta, start_idx, end_idx)
        print("Finish normalizing codec")
    
    # Change metrics to faiss.METRIC_INNER_PRODUCT!!!!!
    return embeddings



def score_norm(embeddings,
              beta:float = 1.0,
              start_idx:int = 0,
              end_idx:int = 2,
              ):
    
    index = faiss.IndexFlatIP(embeddings.dims)
    index.add(embeddings.train_feat)
    index = faiss.index_cpu_to_all_gpus(index, ngpu = faiss.get_num_gpus())
    
    D, I = index.search(embeddings.query_feat, end_idx+1)
    
    adjustment = -beta * np.mean(D[:, start_idx:end_idx+1],
                                 axis = 1,
                                 keepdims = True)
    
    adjusted_queries = np.concatenate([embeddings.query_feat, adjustment], axis = 1)
    setattr(embeddings, 'query_feat', adjusted_queries)
    
    ones = np.ones_like(embeddings.refer_feat[:, :1])
    adjusted_refs = np.concatenate([embeddings.refer_feat, ones], axis = 1)
    setattr(embeddings, 'refer_feat', adjusted_refs)
    
    return embeddings


def negative_embedding_subtraction(faiss_index,
                                   embedding: np.ndarray,
                                   negative_embeddings: np.ndarray,
                                   k:int = 10,
                                   beta:float = 0.35,
                                  ) -> np.ndarray:
    
    # search for hard negatives
    _, topk_indexes = faiss_index.search(embedding, k = k)  
    topk_negative_embeddings = negative_embeddings[topk_indexes]
    
    # subtract by hard negative embeddings
    embedding -= (topk_negative_embeddings.mean(axis = 1) * beta)
    
    # L2-normalize
    embedding /= np.linalg.norm(embedding, axis = 1, keepdims=True)
    
    return embedding


def index_train(embeddings,
                score_norm_arg:bool = True,
                codec_arg:str = 'PCAW512,L2norm,Flat',
                alpha:float = 3.0,
                beta:float = 0.35,
                k:int = 10,
                ):

    index_train = faiss.IndexFlatIP(embeddings.dims)
    index_train = faiss.index_cpu_to_all_gpus(index_train, ngpu = faiss.get_num_gpus())
    
    # Train codec
    index_train.add(embeddings.train_feat)
    
    # DBA on training set
    sim, ind = index_train.search(embeddings.train_feat, k)
    _train = (embeddings.train_feat[ind[:, :k]] * (sim[:, :k, None] ** alpha)).sum(axis = 1)
    _train /= np.linalg.norm(_train, axis = 1, keepdims = True)

    index_train = faiss.index_factory(embeddings.dims, codec_arg)
    index_train = faiss.index_cpu_to_all_gpus(index_train, ngpu = faiss.get_num_gpus())
    index_train.train(_train)
    
    # Project
    embeddings.project(index_train, codec_arg)
    
    # Negative embedding subtraction for query
    sub_query_feat = negative_embedding_subtraction(faiss_index = index_train,
                                                    embedding = embeddings.query_feat,
                                                    negative_embeddings = _train,
                                                    k = k,
                                                    beta = beta)
    replace(embeddings, query_feat = sub_query_feat)
    
    # Negative embedding subtraction for reference
    sub_refer_feat = negative_embedding_subtraction(faiss_index = index_train,
                                                    embedding = embeddings.refer_feat,
                                                    negative_embeddings = _train,
                                                    k = k,
                                                    beta = beta)
    replace(embeddings, refer_feat = sub_refer_feat)
    
    return embeddings



def tabulate_result(embeddings, scores_path, gt, alpha = 1.0):
    
    with open(scores_path, 'rb') as f:
        scores = np.load(f)
        q_idxs = np.load(f)
        r_idxs = np.load(f)
    
    scores = np.where(embeddings.query_patch[q_idxs] > 0, scores * alpha, scores)
    scores = np.where(embeddings.refer_patch[r_idxs] > 0, scores * alpha, scores)
    
    scores  = scores.tolist()
    q_names = embeddings.query_name[q_idxs].tolist()
    r_names = embeddings.refer_name[r_idxs].tolist()
    
    pred = []
    for q_name, r_name, score in zip(q_names, r_names, scores):
        pred.append(PredictedMatch(q_name.decode('utf-8'), r_name.decode('utf-8'), score))
    
    predictions = remove_duplicates(pred)
         
    results = evaluate(gt, predictions)
    print(results.average_precision)
    return {"uAP": results.average_precision,
            "accuracy-at-1": results.recall_at_rank1,
            "recall-at-p90": results.recall_at_p90 or 0.0}