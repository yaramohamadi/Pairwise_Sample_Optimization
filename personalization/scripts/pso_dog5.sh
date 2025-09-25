unique_token="prt"
selected_subject="dog5"
class_token="dog"

echo "Training model for ${selected_subject} (${class_token})"

instance_prompt="a ${unique_token} ${class_token}"
class_prompt="a ${class_token}"

export MODEL_NAME="stabilityai/sdxl-turbo"
export INSTANCE_DIR="dreambooth/dataset/${selected_subject}"
export OUTPUT_DIR="./output"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CUDA_VISIBLE_DEVICES=$1 accelerate launch --main_process_port 22157 personalization/train_pso_sdxl_turbo_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="$instance_prompt" \
    --class_prompt="$class_prompt" \
    --class_name="$class_token" \
    --unique_token=$unique_token \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=4e-5 \
    --report_to="wandb" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1600 \
    --validation_steps=200 \
    --seed=0 \
    --do_edm_style_training \
    --rank=16 \
    --loss_type="pso_db" \
    --beta_pso=5 \
    --gamma_pso=0.0 \
    --num_negatives=20 \
    --prior_loss_weight=0.5 \
    --distill_train_timesteps=4 \
    --neg_generate_freq=1000