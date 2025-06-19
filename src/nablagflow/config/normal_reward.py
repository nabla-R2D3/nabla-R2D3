from config.default_config import get_default_configs

def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "simple_animals" # for HPSv2
    config.experiment.reward_fn = "normal_dsine" # normal_yoso,  normal_d2n_cons
    config.experiment.caching_dir = "./output/Cached_embedding/pas_embedding_gobj83k_msked"
    config.experiment.apply_angle_mask = False
    config.experiment.prompt_fn_kwargs = {}
    config.experiment.is_apply_reward_threshold=False
    config.experiment.reward_dir ="NONE"
    config.experiment.reward_threshold=9999999.9999 ## inf
    
    
    config.training.reward_context = "train"
    # config.training.reward_context = "inference"
    # config.experiment.normal_reward_fn = "normal_dsine"

    return config