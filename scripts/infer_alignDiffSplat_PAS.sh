LORA_DIR=output_align3D/exp_1.1_YOUE_EXP_NAME_1/250612_172339/checkpoint_epoch105
TAG=YOUR_TAG_NAME
PROMPT_FILE=src/nablagflow/alignment/assets/G-obj-83k_sample.txt
OUTPUT_DIR=output/out_inference
python src/infer_diffSplat_pas.py --config_file extensions/diffusers_diffsplat/configs/gsdiff_pas.yaml \
    --prompt_file $PROMPT_FILE \
    --output_dir $OUTPUT_DIR \
    --tag $TAG \
    --output_video_type fancy \
    --seed 0 \
    --infer_from_iter 013020 \
    --load_pretrained_gsvae_ckpt 039030 \
    --load_pretrained_gsrecon_ckpt 124320 \
    --half_precision \
    --allow_ft32 \
    --save_geometry \
    --lora_dir $LORA_DIR \
    --enable_lora
\