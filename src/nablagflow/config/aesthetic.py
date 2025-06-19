from config.default_config import get_default_configs

def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "xxx" 
    config.experiment.reward_fn = "aesthetic_score"
    config.experiment.prompt_fn_kwargs = {}
    # config.experiment.normal_reward_fn = "normal_dsine"         
    config.experiment.caching_dir = "./output/Cached_embedding/pas_embedding_gobj83k_msked"
    config.training.reward_context = "train"

    return config