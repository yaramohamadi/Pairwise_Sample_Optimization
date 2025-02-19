export OUTPUT_DIR="./output"

accelerate launch train_online_pso_sdxl_turbo.py \
--config config/config_sdxl_turbo_dpo.py \
--config.output_dir=$OUTPUT_DIR \
--config.train.lora_rank=32 \
--config.sample.num_batches_per_epoch=4 \
--config.sample.batch_size=4 \
--config.sample.num_steps=4 \
--config.train.learning_rate=1e-5 \
--config.train.beta=50 \
--config.train.batch_size=4 \
--config.train.gradient_accumulation_steps=2 \
--config.train.distilled_train_steps=3 \
--config.train.num_inner_epochs=1