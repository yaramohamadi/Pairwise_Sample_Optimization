#!/bin/bash
# run_all_listed.sh
# Edit the list below to include only the scripts you want to run

SCRIPTS_DIR="personalization/scripts"
GPUS=(1)   # GPU IDs to use
NUM_GPUS=${#GPUS[@]}

# Explicit list of scripts (you can remove/comment out the ones you don’t want)
SCRIPTS=(
  "pso_dog7.sh"
  "pso_dog8.sh"
  "pso_duck_toy.sh"
  "pso_fancy_boot.sh"
  "pso_grey_sloth_plushie.sh"
  "pso_monster_toy.sh"
  "pso_pink_sunglasses.sh"
  "pso_poop_emoji.sh"
  "pso_rc_car.sh"
  "pso_red_cartoon.sh"
  "pso_robot_toy.sh"
  "pso_shiny_sneaker.sh"
  "pso_teapot.sh"
  "pso_vase.sh"
  "pso_wolf_plushie.sh"
)

# Loop through and run each script, round-robin across GPUs
for script in "${SCRIPTS[@]}"; do
    echo "Launching $script on GPU 1"
    bash "$SCRIPTS_DIR/$script" 1
done

wait
echo "All listed jobs finished."
