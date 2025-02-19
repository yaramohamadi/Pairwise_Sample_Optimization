export OUTPUT_DIR="./output"

accelerate launch train_d3po_sdxl_dmd2.py \
--config config/config_sdxl_dmd_dpo.py \
--config.output_dir=$OUTPUT_DIR \
--config.train.lora_rank=16 \
--config.sample.num_batches_per_epoch=8 \
--config.sample.batch_size=2 \
--config.sample.num_steps=8 \
--config.train.learning_rate=1e-5 \
--config.train.beta=50 \
--config.train.batch_size=2 \
--config.train.gradient_accumulation_steps=2 \
--config.train.distilled_train_steps=7 \
--config.train.num_inner_epochs=1