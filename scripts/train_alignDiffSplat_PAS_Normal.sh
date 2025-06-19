
######################################################################
########################################################################
#############   Depth-Normal Cons ######################################
########################################################################
##########################################################################

##################################

export Transformer_FP32_Softmax=True 
export MASTER_PORT=8882
torchrun --standalone --nproc_per_node=2 ./src/train_nabla_r2d3_diffsplat_pas.py \
     --config=src/nablagflow/config/normal_reward.py \
     --seed=0 \
     --config.model.reward_scale=5e5 \
     --config.model.reward_adaptive_scaling=True \
     --config.model.unet_reg_scale=1e4 \
     --config.sampling.low_var_subsampling=True \
     --config.model.timestep_fraction=0.4 \
     --config.training.gradient_accumulation_steps=8 \
     --config.sampling.batch_size=4 \
     --config.sampling.num_batches_per_epoch=4 \
     --config.training.num_epochs=100 \
     --config.training.lr=5e-4 \
     --config.logging.use_wandb=True \
     --config.embedding_cache.read_first_n=10000 \
     --config.experiment.prompt_fn=pixart_cache_prompt_embedding_dataset \
     --config.experiment.caching_dir=output/Cached_embedding/pas_embedding_G-obj-83k \
     --config.training.batch_size=1 \
     --config.experiment.reward_fn=normal_d2n_cons \
     --config.diffsplat.vis_normal=True \
     --config.experiment.apply_angle_mask=False \
     --config.training.mixed_precision=fp32 \
     --config.logging.save_freq=5 \
     --config.experiment.is_apply_reward_threshold=True \
     --config.experiment.reward_dir=output/Cached_embedding/pas_embedding_G-obj-83k/normal_yoso_rewards_First10000.npy \
     --config.experiment.reward_threshold=0.849 \
     --config.diffsplat.random_view=True \
     --config.training.reward_context=inference \
     --exp_name=exp_1.1_YOUE_EXP_NAME_3_FP32_Filter_rward  \

##############################################
##############################################
##############  Normal Yoso ###########################
##############################################
##############################################


# export CUDA_VISIBLE_DEVICES=6,7
# export Transformer_FP32_Softmax=True 
# torchrun --standalone --nproc_per_node=2 ./src/train_nabla_r2d3_diffsplat_pas.py \
#      --config=src/nablagflow/config/normal_reward.py \
#      --seed=0 \
#      --config.model.reward_scale=1e6 \
#      --config.model.reward_adaptive_scaling=True \
#      --config.model.unet_reg_scale=1e4 \
#      --config.sampling.low_var_subsampling=True \
#      --config.model.timestep_fraction=0.4 \
#      --config.training.gradient_accumulation_steps=8 \
#      --config.sampling.batch_size=4 \
#      --config.sampling.num_batches_per_epoch=4 \
#      --config.training.num_epochs=100 \
#      --config.training.lr=5e-4 \
#      --config.logging.use_wandb=True \
#      --config.embedding_cache.read_first_n=10000 \
#      --config.experiment.prompt_fn=pixart_cache_prompt_embedding_dataset \
#      --config.experiment.caching_dir=output/Cached_embedding/pas_embedding_G-obj-83k \
#      --config.training.batch_size=1 \
#      --config.experiment.reward_fn=normal_yoso \
#      --config.training.reward_norm_value=200 \
#      --config.diffsplat.vis_normal=True \
#      --config.experiment.apply_angle_mask=False \
#      --config.logging.save_freq=5 \
#      --config.experiment.is_apply_reward_threshold=False \
#      --config.experiment.reward_dir=output/Cached_embedding/pas_embedding_G-obj-83k/normal_yoso_rewards_First10000.npy \
#      --config.experiment.reward_threshold=0.849 \
#      --config.diffsplat.random_view=True \
#      --config.diffsplat.azi_delta=60 \
#      --config.training.reward_context=inference \
#      --exp_name=exp_1.1_YOUE_EXP_NAME_4 
