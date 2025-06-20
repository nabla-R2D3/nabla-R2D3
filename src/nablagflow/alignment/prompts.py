# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from importlib import resources
import os
import functools
import random
import inflect
import torch
IE = inflect.engine()

import sys
from src.nablagflow.alignment.prompt_embedding_dataset import Pixart_Cache_embeddingDataset
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files
ASSETS_PATH = files("src.nablagflow.alignment.assets")


@functools.lru_cache() # will remember previous 128 calls
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `alignment/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or alignment.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]
def load_embedding_cache(path):
    assert os.path.exists(path), f"Could not find {path}"
    
    print(f"Loading embedding cache from {path}")
    embbeding_cache =  torch.load(path)
    assert "prompt" in embbeding_cache, f"prompt not found in {path}"
    assert "prompt_embedding" in embbeding_cache, f"prompt_embedding not found in {path}"
    assert "null_embedding" in embbeding_cache, f"null_embedding not found in {path}"
    return embbeding_cache

def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


EMBEDDING_CACHE_DATASET = None
def from_embedding_dir(path,batch_size=4, read_first_n=None):
    global EMBEDDING_CACHE_DATASET
    if EMBEDDING_CACHE_DATASET is None:
       EMBEDDING_CACHE_DATASET= Pixart_Cache_embeddingDataset(path,read_first_n=read_first_n)        
       print(f"Loaded {len(EMBEDDING_CACHE_DATASET)} prompts from {path}")
    
    # prompts = _load_lines(path)[low:high]
    # idx =  random.choice(len(EMBEDDING_CACHE_DATASET["prompts"])), {}
    # embedding_dict ={"prompts":  EMBEDDING_CACHE_DATASET["prompts"][idx],
    #                  "prompt_embedding":  EMBEDDING_CACHE_DATASET["prompt_embedding"][idx], 
    #                  "null_embedding":  EMBEDDING_CACHE_DATASET["null_embedding"]}
    
    return EMBEDDING_CACHE_DATASET.get_batch(batch_size)
# def from_embed

def from_file_eval(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    for prompt in prompts:
        yield prompt, {}


short_names = {
    "imagenet_all": "inall",
    "imagenet_animals": "inanm",
    "imagenet_dogs": "indog",
    "simple_animals": "simanm",
    "drawbench": "drawb",

    "hpd": "hpd",
    "hpd_photo": "hppho",
    "hpd_photo_painting": "hpphopa",
    "hpd_photo_anime": "hpphoan",
    "hpd_photo_concept": "hpphoct",

    "nouns_activities": "nounact",
    "counting": "count",
    "pap": "pap",
}

def pap():
    return from_file("pick_a_pic.txt")

def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")
def one_animals():
    return from_file("OneAnimal.txt")


def Gobj83k():
    return from_file("G-obj-83k.txt")
def Gobj83k_cached_PixArtS_embedding(batch_size,read_first_n):
    
    return from_embedding_dir("output/pas_embedding_gobj83k_msked",batch_size,read_first_n)

# def Gobj83k_cached_PixArtS_embedding(batch_size,read_first_n):
    
#     return from_embedding_dir("output/pas_embedding_gobj83k_msked",batch_size,read_first_n)


def simple_animals_eval():
    return from_file_eval("simple_animals.txt")


import csv
import collections
@functools.lru_cache()
def read_csv(path):
    # reader = csv.DictReader(open(path))
    with open (path, 'r') as f:
        reader = csv.DictReader(f)
        reader = [row for row in reader]

    info = collections.defaultdict(list)
    for row in reader:
        info[row["Category"]].append(row["Prompts"])
    """
    [(k, len(v)) for k, v in info.items()]
    [('Colors', 25), ('Conflicting', 10), ('Counting', 19), ('DALL-E', 20), ('Descriptions', 20), ('Gary Marcus et al. ', 10),
     ('Misspellings', 10), ('Positional', 20), ('Rare Words', 7), ('Reddit', 38), ('Text', 21)]
    """

    filtered_info = {}
    for k, v in info.items():
        if k in ["Misspellings", "Rare Words"]: # filter out, rest 183
            continue
        filtered_info[k] = v[2:] # saved for test
    drawbench_prompt_ls = sum(filtered_info.values(), [])
    return drawbench_prompt_ls  # len=165

def drawbench():
    drawbench_prompt_ls = read_csv(ASSETS_PATH.joinpath("DrawBench Prompts.csv"))
    return random.choice(drawbench_prompt_ls), {}


import json
@functools.lru_cache()
def read_hpd(style=None):
    if style is None:
        # 800 prompts for each of the 4 styles
        styles = ["anime", "concept-art", "paintings", "photo"]
    else:
        styles = [style,]
    # dic = {}
    prompts_ls = []
    for style in styles:
        with open(ASSETS_PATH.joinpath(f"HPDv2/benchmark_{style}.json"), "r") as f:
            # dic[style] = json.load(f)  # list of strings
            prompts_ls.extend(json.load(f)[10:]) # 790 for train, 10 for test

    return prompts_ls

def hpd():
    prompts_ls = read_hpd()
    return random.choice(prompts_ls), {}

def hpd_photo():
    prompts_ls = read_hpd("photo")
    return random.choice(prompts_ls), {}

def hpd_photo_painting():
    prompts_ls = read_hpd("photo")
    prompts_ls.extend(read_hpd("paintings")) # not "painting"
    return random.choice(prompts_ls), {}

def hpd_photo_anime():
    prompts_ls = read_hpd("photo")
    prompts_ls.extend(read_hpd("anime"))
    return random.choice(prompts_ls), {}

def hpd_photo_concept():
    prompts_ls = read_hpd("photo")
    prompts_ls.extend(read_hpd("concept-art"))
    return random.choice(prompts_ls), {}

def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata