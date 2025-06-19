################################
################################
###### PAS-HPSv2          ########
################################
################################

export MASTER_PORT=8896

export CUDA_VISIBLE_DEVICES=6,7
export Transformer_FP32_Softmax=True 
torchrun --standalone --nproc_per_node=2 ./src/train_nabla_r2d3_diffsplat_pas.py \
     --config=src/nablagflow/config/hpsv2.py \
     --seed=1 \
     --config.model.reward_scale=5e7 \
     --config.model.reward_adaptive_scaling=True \
     --config.model.unet_reg_scale=5e3 \
     --config.sampling.low_var_subsampling=True \
     --config.model.timestep_fraction=0.4 \
     --config.training.gradient_accumulation_steps=8 \
     --config.sampling.batch_size=4 \
     --config.sampling.num_batches_per_epoch=4 \
     --config.training.num_epochs=100 \
     --config.training.lr=1e-4 \
     --config.logging.use_wandb=True \
     --config.diffsplat.random_view=True \
     --config.experiment.prompt_fn=pixart_cache_prompt_embedding_dataset \
     --config.experiment.caching_dir=output/Cached_embedding/pas_embedding_G-obj-83k \
     --config.training.batch_size=1 \
     --config.training.reward_norm_clip=False \
     --config.training.reward_norm_value=3000000 \
     --config.experiment.reward_fn=hpscore \
     --config.diffsplat.vis_normal=False \
     --exp_name=exp_1.1_YOUE_EXP_NAME_2





