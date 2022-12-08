import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning import LightningDataModule

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.datamodules.components.augment import Augmentation
from src.utils import get_image_paths, read_ground_truth, readimage



SPLIT_REFER = 0
SPLIT_QUERY = 1
SPLIT_TRAIN = 2



class RepeatedAugment:        
    def __init__(self,
                 transform,
                 weak_augment, 
                 strong_augment,
                 n_repeat_aug:int = 1):
        
        self.transform = transform
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment
        self.n_repeat_aug = n_repeat_aug

    def __call__(self, img, idx):
        record = {'instance_id': idx}
        record['image0'] = self.transform(image=self.weak_augment(img))['image']
        
        for i in range(self.n_repeat_aug):
            record[f'image{i+1}'] = self.transform(image=self.strong_augment(img))['image']
        return record
    
    

class PatchifyTransform:    
    def __call__(self, image):
        h, w, c = image.shape
        patches = [image] # patch0 is uncropped
        
        crop_ords = [(0,     0,      0.6*w,  0.6*h),
                     (0.4*w, 0,      w,      0.6*h),
                     (0,     0.4*h,  0.6*w,  h),
                     (0.4*w, 0.4*h,  w,      h),
                     (0.2*w, 0.2*h,  0.8*w,  0.8*h)]
        
        for crop_ord in crop_ords:
            
            x_min, y_min, x_max, y_max = crop_ord
            img = A.crop(img = image,
                         x_min = int(x_min),
                         y_min = int(y_min),
                         x_max = int(x_max),
                         y_max = int(y_max))
            patches.append(img)
        return patches


  
class DISCTrainingDataset(Dataset):
    def __init__(self,
                 train_path:str,
                 repeated_augment:RepeatedAugment,
                 ndec_path:str = None):

        self.isc_path, self.files = get_image_paths(train_path)
        ############################SANITY CHECK##############################
        #self.files = self.files[:40] # Sanity Check
        self.files = self.files[:25000]
        self.metadata = ['I' for _ in range(len(self.files))]
        
        # Include additional hard negative training samples from NEDC dataset
        if ndec_path:
            self.ndec_path, ndec_files = get_image_paths(ndec_path)
            self.files.extend(ndec_files)
            
            ndec_meta = ['N' for _ in range(len(ndec_files))]
            self.metadata.extend(ndec_meta)

        self.repeated_augment = repeated_augment

    def __getitem__(self, idx: int) -> Dict:
        
        if self.metadata[idx] == 'I':
            path = self.isc_path
        elif self.metadata[idx] == 'N':
            path = self.ndec_path
        else:
            raise ValueError('Unknown metadata prefix in training set')
            
        img = readimage(os.path.join(path, self.files[idx]))
        record = self.repeated_augment(img, idx)
        
        return record
        
    def __len__(self) -> int:
        return len(self.files)
    
    
    
@dataclass
class Record:
    image:torch.Tensor = None
    image_num:int = None
    instance_id:int = None
    split:int = None
    patch:int = None
    feat_agg:torch.Tensor = None
        
    def __getnames__(self):
        return self.__dict__.keys()
    
    
    
@dataclass
class BatchRecords:
    image:List[torch.Tensor] = field(default_factory = list)
    image_num:List[int] = field(default_factory = list)
    instance_id:List[int] = field(default_factory = list)
    split:List[int] = field(default_factory = list)
    patch:List[int] = field(default_factory = list)
    feat_agg:List[torch.Tensor] = field(default_factory = list)
        
    def __getnames__(self):
        return self.__dict__.keys()
    
    def add_batch(self, batch):
        for name in batch.__getnames__():
            curr_attr = getattr(self, name)
            new_attr = getattr(batch, name)
            curr_attr.extend(new_attr)
            setattr(self, name, curr_attr)
    
    def add_record(self, record):
        for name in record.__getnames__():
            curr_attr = getattr(self, name)
            new_attr = getattr(record, name)
            curr_attr.append(new_attr)
            setattr(self, name, curr_attr)
            
    def numplify(self):
        for name in self.__getnames__():
            curr_attr = getattr(self, name)
            
            if type(curr_attr[0]) == int:
                curr_attr = np.array(curr_attr)
            elif type(curr_attr[0]) == torch.Tensor:
                curr_attr = torch.stack(curr_attr).cpu().numpy()
            else:
                raise ValueError(f'Cannot convert type{type(curr_attr[0])}')

            setattr(self, name, curr_attr)
            
            
            
def collate_fn(batch):
    batchrecord = BatchRecords()
    for record in batch:
        batchrecord.add_record(record)
    # Stack list of images into single tensor
    batchrecord.image = torch.cat(batchrecord.image)
    return batchrecord



def collate_batch_fn(batches):
    batchrecord = BatchRecords()
    for batch in batches:
        batchrecord.add_batch(batch)
    # Stack list of images into single tensor
    batchrecord.image = torch.cat(batchrecord.image)
    return batchrecord



class DISCEvalDataset(Dataset):
    def __init__(self, 
                ref_path: str,
                query_path: str,
                gt_path: str,
                transform:object,
                patchify: object = None,
                train_path: str = None,
                ref_subset_path: str = None,
                query_subset_path: str = None):
        
        super().__init__()
        
        ref_subset = query_subset = None
        if ref_subset_path:
            ref_subset = open(ref_subset_path, 'r').read().splitlines()
            ############################SANITY CHECK##############################
            #ref_subset = ref_subset[:50] # Sanity check
        if query_subset_path:
            query_subset = open(query_subset_path, 'r').read().splitlines()
            ############################SANITY CHECK##############################
            #query_subset = query_subset[:50] # Sanity check
            
        self.files, self.metadata = self.read_files(ref_path, SPLIT_REFER, ref_subset)
        query_files, query_metadata = self.read_files(query_path, SPLIT_QUERY, query_subset)
        self.files.extend(query_files)
        self.metadata.extend(query_metadata)
        
        if train_path:
            train_files, train_metadata = self.read_files(train_path, SPLIT_TRAIN)
            #self.files.extend(train_files)
            #self.metadata.extend(train_metadata)
            ############################SANITY CHECK##############################
            self.files.extend(train_files[:25000])
            self.metadata.extend(train_metadata[:25000])
        
        self.gt = read_ground_truth(gt_path)
        self.transform = transform
        self.patchify = patchify
        self.paths = {SPLIT_REFER: ref_path,
                      SPLIT_QUERY: query_path,
                      SPLIT_TRAIN: train_path}
    
    def read_files(self,
                   path:str,
                   split:str,
                   file_subset:List = None,
                  ) -> Tuple[List, List[Dict]]:
        
        if file_subset:
            files = [os.path.join(path, f'{file}.jpg') for file in file_subset]

        else:
            _, files = get_image_paths(path)
            
        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        metadata = [{'image_num': int(name[1:]), 'split': split} for name in names]
        return files, metadata
    
    def __getitem__(self, idx: int) -> Dict:
        file_dir = self.paths[self.metadata[idx]['split']]
        filename = os.path.join(file_dir, self.files[idx])
        img = readimage(filename)
        
        record = Record(instance_id = idx,
                        patch = 0,
                        split =  self.metadata[idx]['split'],
                        image_num = self.metadata[idx]['image_num'],
                        )
        
        # Patchify for ref and query during testing
        if self.patchify:
            batchrecord = BatchRecords()
            patches = self.patchify(img)
            for i, patch in enumerate(patches):
                record.image = self.transform(image = patch)['image'].unsqueeze(0)
                record.patch = i
                
                batchrecord.add_record(record)
                
                # We only extract the whole image for training images
                if self.metadata[idx]['split'] == SPLIT_TRAIN:
                    break

            return batchrecord
        else:
            record.image = self.transform(image = img)['image'].unsqueeze(0)

            return record
    
    def __len__(self) -> int:
        return len(self.files)
    
    
    
@dataclass       
class CopyDetectorDataModule(LightningDataModule):
    train_path:str
    ref_path:str
    val_query_path:str
    test_query_path:str
    val_gt_path:str
    test_gt_path:str
    ndec_path:str = None
    ref_subset_path:str = None
    query_subset_path:str = None
    train_batch_size:int = 2
    train_img_size:int = 224 
    val_batch_size:int = 2
    val_img_size:int = 224
    test_batch_size:int = 2
    test_img_size:int = 384
    workers:int = 10
    n_weak_aug:int = 1
    n_strong_aug:int = 3
    n_repeat_aug:int = 1
    
    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters(logger = False)
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        weak_augment = Augmentation(n_augments = self.n_weak_aug,
                                    bg_image_dir = self.train_path,
                                    strong_augment = False)
        
        strong_augment = Augmentation(n_augments = self.n_strong_aug,
                                     bg_image_dir = self.train_path,
                                     strong_augment = True)

        train_transform = A.Compose([A.Resize(self.train_img_size, self.train_img_size),
                                     A.Normalize(),
                                     ToTensorV2()])

        repeated_augment = RepeatedAugment(transform = train_transform,
                                           weak_augment = weak_augment,
                                           strong_augment = strong_augment,
                                           n_repeat_aug = self.n_repeat_aug)
                
        self.train_dataset = DISCTrainingDataset(train_path = self.train_path,
                                                 repeated_augment = repeated_augment,
                                                 ndec_path = self.ndec_path)
        
        print(f'Training dataset contains {len(self.train_dataset)} images.')
        
        val_transform = A.Compose([A.Resize(self.val_img_size, self.val_img_size),
                                   A.Normalize(),
                                   ToTensorV2()])
                    
        self.val_dataset = DISCEvalDataset(ref_path = self.ref_path,
                                           query_path = self.val_query_path,
                                           gt_path = self.val_gt_path,
                                           transform = val_transform,
                                           ref_subset_path = self.ref_subset_path,
                                           query_subset_path = self.query_subset_path)
        
        print(f'Validation dataset contains {len(self.val_dataset)} images.')
        
        patchify = PatchifyTransform()
        
        test_transform = A.Compose([A.Resize(self.test_img_size, self.test_img_size),
                                    A.Normalize(),
                                    ToTensorV2()])
        
        self.test_dataset = DISCEvalDataset(ref_path = self.ref_path,
                                            query_path = self.test_query_path,
                                            gt_path = self.test_gt_path,
                                            transform = test_transform,
                                            patchify = patchify,
                                            train_path = self.train_path,
                                            ref_subset_path = self.ref_subset_path,
                                            query_subset_path = self.query_subset_path)
        
        print(f'Test dataset contains {len(self.test_dataset)} images.')
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size = self.train_batch_size,
                          num_workers = self.workers,
                          persistent_workers = True,
                          pin_memory = True,
                          shuffle = True,
                          drop_last = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                         batch_size = self.val_batch_size,
                         num_workers = self.workers,
                         persistent_workers = True,
                         pin_memory = True,
                         collate_fn = collate_fn,
                         shuffle = False,
                         drop_last = False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                         batch_size = self.test_batch_size,
                         num_workers = self.workers,
                         persistent_workers = True,
                         pin_memory = True,
                         collate_fn = collate_batch_fn,
                         shuffle = False,
                         drop_last = False)
        
        
class DISCMatching(Dataset):
    def __init__(self,
                query_embs,
                refer_embs,
                candidate_set,
                ):
        self.query_embs = query_embs
        self.refer_embs = refer_embs
        self.candidate_set = candidate_set

    def __len__(self) -> int:
        return len(self.candidate_set)
    
    def __getitem__(self, idx:int):
        query_idx, refer_idx, _ = self.candidate_set[idx]
        
        refer_emb = self.refer_embs[refer_idx]
        query_emb = self.query_embs[query_idx]
        
        return query_emb, refer_emb, query_idx, refer_idx