#!/bin/bash
# run_all_listed.sh
# Edit the list below to include only the scripts you want to run

SCRIPTS_DIR="personalization/scripts"
GPUS=(3)   # GPU IDs to use
NUM_GPUS=${#GPUS[@]}

# Explicit list of scripts (you can remove/comment out the ones you don’t want)
SCRIPTS=(
  "pso_backpack_dog.sh"
  "pso_backpack.sh"
  "pso_bear_plushie.sh"
  "pso_berry_bowl.sh"
  "pso_can.sh"
  "pso_candle.sh"
  "pso_cat.sh"
  "pso_cat2.sh"
  "pso_clock.sh"
  "pso_colorful_sneaker.sh"
  "pso_dog.sh"
  "pso_dog2.sh"
  "pso_dog3.sh"
  "pso_dog5.sh"
  "pso_dog6.sh"
)

# Loop through and run each script, round-robin across GPUs
for script in "${SCRIPTS[@]}"; do
    echo "Launching $script on GPU 3"
    bash "$SCRIPTS_DIR/$script" 3
done

wait
echo "All listed jobs finished."