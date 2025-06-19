from ml_collections import config_dict

def get_default_configs():
    config = config_dict.ConfigDict()

    diffsplat = config.diffsplat = config_dict.ConfigDict()
    diffsplat.init_std=0.0
    diffsplat.init_noise_strength=0.98
    diffsplat.init_bg=0.
    diffsplat.min_guidance_scale=1.0
    diffsplat.guidance_scale=7.5
    diffsplat.triangle_cfg_scaling=False
    diffsplat.num_views=4
    diffsplat.render_res=256
    diffsplat.fxfy=1422.222/1024
    diffsplat.elevation=10
    diffsplat.elevation_delta=0
    diffsplat.distance=1.4
    diffsplat.random_view=False
    diffsplat.vis_normal=False
    diffsplat.azi_delta=20
    
    
    
    embedding_cache = config.embedding_cache = config_dict.ConfigDict()
    embedding_cache.read_first_n =-1
    
    
    
    

    
    training = config.training = config_dict.ConfigDict()
    training.use_8bit_adam = False
    training.lr = 1e-3
    training.flow_lr = 1e-3
    training.flow_wd = 1e-2
    training.adam_beta1 = 0.9
    training.adam_beta2 = 0.999
    training.adam_weight_decay = 1e-4
    training.adam_epsilon = 1.e-8
    training.max_grad_norm = 1.0
    training.num_inner_epochs = 1
    training.cfg = True
    training.adv_clip_max = 5
    training.clip_range = 1e-4
    training.anneal = None
    training.batch_size = 1 ##  NOW only I is supported
    # training.batch_size = 4 ## TODO: changed this to 1
    # training.gradient_accumulation_steps = 8 ## TODO: changed this to 1
    training.gradient_accumulation_steps = 4 ## TODO: changed this to 1
    training.num_epochs = 100
    training.mixed_precision = "bf16"
    training.allow_tf32 = True
    training.gradscaler_growth_interval = 2000
    training.diffusion_reg = False


    training.adaptive_init_scale = False
    training.large_init_scale = False
    training.loss_type = 'l2'
    training.loss_clip_value = -1.0
    training.reward_norm_clip=False
    training.reward_norm_value=-1
    # training.unet_norm_clip=False
    training.unet_norm_clip=-1.0
    
    training.alpha_prod_clip=False
    
    training.reward_aggregate_func="mean"
    training.final_step_rw_only=False ## ONLY FOR DEBUGGING:Normal Rewards
    
    training.lora_dir=""
    training.load_ckpt=False
    training.memo_efficient_mode=False
    
    



    model = config.model = config_dict.ConfigDict()
    model.use_peft = True
    model.lora_rank = 8
    model.peft_type = 'lora'
    model.reward_scale = 1.0
    model.n_reward_avg = 1
    model.timestep_fraction = 0.4
    # model.timestep_fraction = 0.1
    ### GFN Specific
    model.unet_reg_scale = 1e3

    model.residual_loss = True
    model.pretrained_strength = 1.0
    model.reward_adaptive_scaling = False
    model.reward_adaptive_mode = 'squared'
    model.stop_forward_flow = False
    model.flow_peft = False

    experiment = config.experiment = config_dict.ConfigDict()


    sampling = config.sampling = config_dict.ConfigDict()
    sampling.num_steps = 20
    # sampling.num_steps = 50
    sampling.eta = 1
    sampling.guidance_scale = 5.0
    sampling.batch_size = 16
    sampling.num_batches_per_epoch = 4
    sampling.low_var_subsampling = False





    logging = config.logging = config_dict.ConfigDict()
    logging.use_wandb = False
    # logging.use_wandb = True
    logging.save_freq = 5
    logging.num_checkpoing_limit = 5
    logging.save_json = True
    
    logging.every_eval_epoch=20







    return config