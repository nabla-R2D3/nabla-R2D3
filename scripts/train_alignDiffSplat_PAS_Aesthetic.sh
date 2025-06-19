

###############################
###############################
##### DiffSplat-PAS Aesthetic Reward #####
###############################
###############################

export PixArtTransformerMV2DModel_trick_mix_precesion=False
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nproc_per_node=2 ./src/train_nabla_r2d3_diffsplat_pas.py \
     --config=src/nablagflow/config/aesthetic.py \
     --seed=1 \
     --config.model.reward_scale=2e6 \
     --config.model.reward_adaptive_scaling=True \
     --config.model.unet_reg_scale=3e3 \
     --config.sampling.low_var_subsampling=True \
     --config.model.timestep_fraction=0.4 \
     --config.training.gradient_accumulation_steps=4 \
     --config.sampling.batch_size=4 \
     --config.sampling.num_batches_per_epoch=4 \
     --config.training.num_epochs=100 \
     --config.training.lr=3e-4 \
     --config.logging.use_wandb=True \
     --config.experiment.prompt_fn=pixart_cache_prompt_embedding_dataset \
     --config.experiment.reward_fn=aesthetic_score \
     --config.experiment.caching_dir=output/Cached_embedding/pas_embedding_G-obj-83k \
     --config.training.batch_size=1 \
     --config.training.reward_norm_clip=False \
     --config.training.reward_norm_value=1000000 \
     --config.diffsplat.vis_normal=False \
     --config.diffsplat.random_view=True \
     --exp_name=exp_1.1_YOUE_EXP_NAME_1 \
     # --caption=RewardScale1e6_RwNorm1e6_lastStep_Lr5e-4 \
