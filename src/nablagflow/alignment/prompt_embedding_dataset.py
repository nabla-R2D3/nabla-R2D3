import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
from PIL import Image
from einops import rearrange, repeat, reduce
from glob import glob
from tqdm import tqdm

import logging
logger = logging.getLogger('root')


class Pixart_Cache_embeddingDataset(Dataset):

    def __init__(self, data_dir="output/Cached_embedding/pas_embedding_gobj83k_msked",read_first_n=None,is_apply_reward_threshold=False,reward_dir =None,reward_threshold=np.inf):
        
        
        
        D_DIM = 4096
        MAX_WORDS = 300
        self.D_DIM = D_DIM
        self.MAX_WORDS = MAX_WORDS
        self.data_dir = data_dir
        self.embedding_files = sorted(glob(os.path.join(data_dir, "*-*prompt_embeds.pt")))
        self.embedding_msk_files = sorted(glob(os.path.join(data_dir, "*-*prompt_attention_masks.pt")))
        # self.prompt_files=[]
        prompt_txt_file = glob(os.path.join(data_dir, "*.txt"))
        assert len(prompt_txt_file)==1
        logger.info(f"Loading prompts from {prompt_txt_file[0]}")
        if is_apply_reward_threshold:
            logger.info(f"loading reward file {reward_dir}")
            logger.info(f"Applying reward threshold {reward_threshold}")
            
            rewards =  np.load(reward_dir)
            rewards_mask = rewards>reward_threshold
            
        with open(prompt_txt_file[0], "r") as f:
            self.prompt_files = [line.strip() for line in f.readlines()]
            
        embeddings=[]
        masks=[]
        prompts=[]
        done=False
        global_idx=-1
        for emb_file, msk_file in tqdm(zip(self.embedding_files, self.embedding_msk_files)):
            if emb_file.split("/")[-1].split("-")[0] != msk_file.split("/")[-1].split("-")[0] or \
                emb_file.split("/")[-1].split("-")[1][:5] != msk_file.split("/")[-1].split("-")[1][:5]:
                    
                logger.info(f"Error: {emb_file} and {msk_file} do not match")
                break
            embedding= torch.load(emb_file)
            mask= torch.load(msk_file)
            Num= mask.shape[0]
            original_embedding=torch.zeros(Num,D_DIM)
            original_embedding[mask==1]=embedding
            original_embedding= rearrange(original_embedding,"(n w) d -> n w d",w = MAX_WORDS, d = D_DIM)   
            original_msk = rearrange(mask,"(n w) -> n w",w = MAX_WORDS) 
            
            for idx in range(original_msk.shape[0]):
                global_idx+=1
                if is_apply_reward_threshold and not rewards_mask[global_idx]:
                    continue
                
                embeddings.append(original_embedding[idx][original_msk[idx]==1])
                masks.append(original_msk[idx])
                prompts.append(self.prompt_files[global_idx])
                if read_first_n is not None and read_first_n >0 and (global_idx+1)>=read_first_n:
                    done=True
                    break
                
            if done:
                break
        logger.info(f"Total {len(embeddings)} embeddings loaded")
            
        self.embeddings = embeddings
        self.masks = masks
        self.prompts = prompts
        # self.prompts = self.prompt_files[:len(embeddings)]
        assert len(self.prompts)==len(self.embeddings),f"Prompt files {len(self.prompts)} and embeddings {len(self.embeddings)} do not match"
        assert len(self.prompts)==len(self.masks),f"Prompt files {len(self.prompts)} and masks {len(self.masks)} do not match"
        
        if is_apply_reward_threshold:
            logger.info(f"Total {len(embeddings)} embeddings loaded after applying reward threshold")
        
        
        self.current_idx=0
        
        

    def __len__(self):
        """
        Get the size of the dataset
        """
        return len(self.embeddings)
    


    def get_batch(self, batch_size):
        """

        """
        # for i in tqdm(range(1000)):
        idx = np.random.randint(0, len(self.embeddings), batch_size)
        idx_list = idx.tolist() 
        embeds = []
        masks = []
        prompts = [self.prompts[i] for i in idx_list]
        for  idx in idx_list:
            embed, mask = self.embeddings[idx], self.masks[idx]
        # for embed, mask in zip([self.embeddings[i] for i in idx_list], [self.masks[i] for i in idx_list]):
            embed_ori=torch.zeros(self.MAX_WORDS,self.D_DIM)
            embed_ori[mask==1]=embed
            # yield embed_ori, mask, self.prompts[idx
            embed_ori= rearrange(embed_ori,"w d -> 1 w d")
            mask= rearrange(mask,"w -> 1 w")
            embeds.append(embed_ori)
            masks.append(mask)
        meta = [{}]*batch_size
        return {"embeds":embeds, "masks":masks, "prompts":prompts,"metadata":meta}
    def get_by_idxs(self, idx_list:list):
        """

        """
        # for i in tqdm(range(1000)):
        batch_size = len(idx_list)
        embeds = []
        masks = []
        prompts = [self.prompts[i] for i in idx_list]
        for  idx in idx_list:
            embed, mask = self.embeddings[idx], self.masks[idx]
        # for embed, mask in zip([self.embeddings[i] for i in idx_list], [self.masks[i] for i in idx_list]):
            embed_ori=torch.zeros(self.MAX_WORDS,self.D_DIM)
            embed_ori[mask==1]=embed
            # yield embed_ori, mask, self.prompts[idx
            embed_ori= rearrange(embed_ori,"w d -> 1 w d")
            mask= rearrange(mask,"w -> 1 w")
            embeds.append(embed_ori)
            masks.append(mask)
        meta = [{}]*batch_size
        return {"embeds":embeds, "masks":masks, "prompts":prompts,"metadata":meta}
        # return {"embeds":torch.cat(embeds,dim=0), "masks":torch.cat(masks,dim=0), "prompts":prompts}
    def get_next_batch(self,batch_size):
        """generate data squentially

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.current_idx>=len(self.embeddings):
            return None
        
        
        if self.current_idx+batch_size>len(self.embeddings):
            idx = np.arange(self.current_idx, len(self.embeddings))
            self.current_idx=len(self.embeddings)
            
        else:
            idx = np.arange(self.current_idx, self.current_idx+batch_size)
            self.current_idx+=batch_size
        
        idx_list = idx.tolist() 
        embeds = []
        masks = []
        prompts = [self.prompts[i] for i in idx_list]
        for  idx in idx_list:
            embed, mask = self.embeddings[idx], self.masks[idx]
        # for embed, mask in zip([self.embeddings[i] for i in idx_list], [self.masks[i] for i in idx_list]):
            embed_ori=torch.zeros(self.MAX_WORDS,self.D_DIM)
            embed_ori[mask==1]=embed
            # yield embed_ori, mask, self.prompts[idx
            embed_ori= rearrange(embed_ori,"w d -> 1 w d")
            mask= rearrange(mask,"w -> 1 w")
            embeds.append(embed_ori)
            masks.append(mask)
        meta = [{}]*batch_size
        return {"embeds":embeds, "masks":masks, "prompts":prompts,"metadata":meta}
        
        
        
        

class Pixart_Cache_embeddingDataset_HPD(Pixart_Cache_embeddingDataset):

    def __init__(self, data_dirs=None,start_idx=0,end_idx=-1):
        import json
        
        
        D_DIM = 4096
        MAX_WORDS = 300 ### Hardcoded for T5Tokenizer
        self.D_DIM = D_DIM
        self.MAX_WORDS = MAX_WORDS
        self.data_dirs = data_dirs
        assert len(data_dirs)>=1,"at least one data_dir is required"
        
        self.embeddings = []
        self.masks = []
        self.prompts = []
        for data_dir in data_dirs:
            self.embedding_files = sorted(glob(os.path.join(data_dir, "*-*prompt_embeds.pt")))
            self.embedding_msk_files = sorted(glob(os.path.join(data_dir, "*-*prompt_attention_masks.pt")))
            # self.prompt_files=[]
            prompt_txt_file = glob(os.path.join(data_dir, "*.json"))
            assert len(prompt_txt_file)==1,"Only one prompt file is allowed"
            logger.info(f"Loading prompts from {prompt_txt_file[0]}")
            
            with open(prompt_txt_file[0], "r") as f:
                lines= json.load(f)
                prompt_files = [line.strip() for line in lines]
                
            embeddings=[]
            masks=[]
            done=False
            global_idx=0
            for emb_file, msk_file in tqdm(zip(self.embedding_files, self.embedding_msk_files)):
                if emb_file.split("/")[-1].split("-")[0] != msk_file.split("/")[-1].split("-")[0] or \
                    emb_file.split("/")[-1].split("-")[1][:5] != msk_file.split("/")[-1].split("-")[1][:5]:
                        
                    logger.info(f"Error: {emb_file} and {msk_file} do not match")
                    break
                embedding= torch.load(emb_file)
                mask= torch.load(msk_file)
                Num= mask.shape[0]
                original_embedding=torch.zeros(Num,D_DIM)
                original_embedding[mask==1]=embedding
                original_embedding= rearrange(original_embedding,"(n w) d -> n w d",w = MAX_WORDS, d = D_DIM)   
                original_msk = rearrange(mask,"(n w) -> n w",w = MAX_WORDS) 
                
                for idx in range(original_msk.shape[0]):
                    if start_idx>0 and global_idx<start_idx:
                        global_idx+=1
                        continue
                    if end_idx>0 and global_idx>end_idx:
                        done=True
                        break
                        
                    embeddings.append(original_embedding[idx][original_msk[idx]==1])
                    masks.append(original_msk[idx])
                    global_idx+=1
                    
                if done:
                    break
            logger.info(f"Total {len(embeddings)} embeddings loaded")
                
            self.embeddings.extend(embeddings)
            self.masks.extend(masks)
            if end_idx>0:
                prompt_files = prompt_files[:end_idx]
            if start_idx>0:
                prompt_files = prompt_files[start_idx:]
            assert len(prompt_files)==len(embeddings),f"Prompt files {len(prompt_files)} and embeddings {len(embeddings)} do not match"
            self.prompts.extend(prompt_files)
            
def pixart_cache_prompt_embedding_dataset(parent_dir,read_first_n=None,**kwargs):
    
    return Pixart_Cache_embeddingDataset(parent_dir,read_first_n=read_first_n,**kwargs)

def pixart_cache_prompt_embedding_dataset_hpd(parent_dir ):
    
    data_dirs =[ os.path.join(parent_dir,sub_dir) for sub_dir  in ["benchmark_anime", "benchmark_concept-art", "benchmark_photo", "benchmark_paintings"] ]
    
    
    return Pixart_Cache_embeddingDataset_HPD(data_dirs,start_idx=10,end_idx=-1)

def pixart_cache_prompt_embedding_dataset_hpd_photo(parent_dir ):
    
    data_dirs =[ os.path.join(parent_dir,sub_dir) for sub_dir  in ["benchmark_photo"] ]
    
    
    return Pixart_Cache_embeddingDataset_HPD(data_dirs,start_idx=10,end_idx=-1)

def pixart_cache_prompt_embedding_dataset_hpd_photo_painting(parent_dir ):
    
    data_dirs =[ os.path.join(parent_dir,sub_dir) for sub_dir  in ["benchmark_photo","benchmark_paintings"] ]
    
    
    return Pixart_Cache_embeddingDataset_HPD(data_dirs,start_idx=10,end_idx=-1)
def pixart_cache_prompt_embedding_dataset_hpd_photo_anime(parent_dir ):
    
    data_dirs =[ os.path.join(parent_dir,sub_dir) for sub_dir  in ["benchmark_photo","benchmark_anime"] ]
    
    
    return Pixart_Cache_embeddingDataset_HPD(data_dirs,start_idx=10,end_idx=-1)
def pixart_cache_prompt_embedding_dataset_hpd_photo_concept(parent_dir ):
    
    data_dirs =[ os.path.join(parent_dir,sub_dir) for sub_dir  in ["benchmark_photo","benchmark_concept-art"] ]
    
    
    return Pixart_Cache_embeddingDataset_HPD(data_dirs,start_idx=10,end_idx=-1)



if __name__ == "__main__":
    from torchvision import transforms
    from torch.nn.parallel import DistributedDataParallel as DDP

    A=Pixart_Cache_embeddingDataset("output/pas_embedding_gobj83k_msked")
    
    A = A.get_batch(10)
