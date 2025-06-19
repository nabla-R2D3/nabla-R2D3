# Nabla-R2D3: Effective and Efficient 3D Diffusion Alignment with 2D Rewards
This is the official implementation of [Nabla-R2D3](http://nabla-r2d3.github.io/), which is a highly effective and sample-efficient reinforcement learning alignment framework for 3D-native diffusion models using pure 2D rewards.

## TODO List:
- [x] Training codes.
- [x] Inference codes.
- [ ] Evaluation codes.
- [ ] Detailed instructions.
- [ ] Additional 3D-native generative models.


## 0. Environment Setup  
- Set up the environment according  

  ```bash
  bash  settings/setup.sh
  ```  
- [Optional] Install a custom Diffusers package (if using BF16 precision).
  ```bash  
  ## ready soon
  ``` 
  This branch mainly modifies the setting of transformers to use FP32 precision before feeding hidden feature into the softmax function. This change improves numerical stability while avoiding excessive GPU memory consumption.


## 1. Run Nabla-R2D3 
Here we provide the training scripts using DiffSplat-Pixart-Sigma as the base model. 

- **Step 0:** Prepare the DiffSplat weights using the script provided by DiffSplat to download them:  
  ```bash  
  python3 download_ckpt.py --model_type pas --local_dir output/diffSplatHFCKPT  
  ```  
  Organize the files into the following structure:  
  ```bash  
  ├── output  
  │   ├── diffSplatHFCKPT  
  │   │   ├── gsdiff_gobj83k_pas_fp16__render  
  │   │   └── gsvae_gobj265k_sdxl_fp16  
  │   │   └── gsrecon_gobj265k_cnp_even4  
  ```  

- **Step 1:** Precompute text embeddings: Follow the instructions in **Section 1: Caching Prompt Embedding** of `./cache_embedding.ipynb`, specifically subsections 1.1 and 1.2. (You may need to adjust the `dir` and `file_path` paths as needed.)

- **Step 2:** Set the `config.experiment.caching_dir` in `train_alignDiffSplat_PAS_{REWARD_NAME}.sh` to the embedding cache directory generated in Step 1. 



- **Step 3:** For Normal Estimator - StableNormal (Optional):  
  ```bash 
  cd ./extensions/normal_priors/StableNormal
  pip setup install 
  ```


- **Step 4:** Execute the training scripts:  
  ```bash  
  ## HPS reward  
  bash scripts/train_alignDiffSplat_PAS_HPS.sh  
  ## Aesthetic score  
  bash scripts/train_alignDiffSplat_PAS_Aesthetic.sh  
  ## Normal reward (includes Normal-estimator and DNC )
  bash scripts/train_alignDiffSplat_PAS_Normal.sh  
  ```  
  The main difference between the scripts lies in the configuration files:  
  For normal reward, use `--config=src/nablagflow/config/normal_reward.py`.  
  For aesthetic score, use `--config=src/nablagflow/config/aesthetic.py`. 
- **Step 5:** Inference with finetuned ckpts:  
  ```bash  
  ## Before execution, set the following variables in the bash file: `LORA_DIR`, `TAG`, `PROMPT_FILE`, and `OUTPUT_DIR`.
  ## The `LORA_DIR` variable specifies the directory containing the finetuned LoRA checkpoint generated in Step 4.
  bash scripts/test_alignDiffSplat_PAS.sh
  ```  


### Explanation of key configurations:  
  - `random_view`: Determines whether to add random camera pose perturbations.
  - `vis_normal`: Determines whether to visulize the normal information.  
  - `config.model.unet_reg_scale`: Determines the magnitude of updates with respect to the model at the previous training step.  
  - `config.training.reward_norm_clip`: If set to `True`, reward clipping will be enabled to filter outliers.
  - `config.training.reward_norm_value`: Specifies the norm threshold used for `reward_norm_clip`.
  - `config.model.timestep_fraction`: Determines the ratio of transitions used for fine-tuning in a trajectory.
  - `config.model.reward_scale`: The $\beta$ in our paper. higher values lead to faster reward convergence and higher reward at convergence, but worse prior preservation and worse text-object alignment.  
  - `config.training.lr`: Learning rate during finetuning.  
  - `config.training.memo_efficient_mode`: If set to `True`, evaluates reward by processing only two images at a time, which is used for saving gpu memory comsumption.  
  - `config.training.reward_context`: Determines whether to enable gradient computation or stop gradient propagation.
  - `config.experiment.apply_angle_mask`: Applies an angle mask to normals if enabled.  
  - `config.experiment.is_apply_reward_threshold`: When using normal rewards, determines whether to enable reward filtering.  
  - `config.experiment.reward_dir`: Specifies the path for pre-computed rewards file.  
    Example: `output/Cached_embedding/pas_embedding_G-obj-83k/normal_yoso_rewards_First10000.npy`.  
  - `config.experiment.reward_threshold`: Sets the filtering threshold for rewards when using normal reward mode.  
  - `config.diffsplat.random_view`: Determines whether to add random camera pose perturbations, fixed at 20°.  


