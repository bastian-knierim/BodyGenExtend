N="range(1)"
MODEL="Flip"
# EXPTS="crawler terraincrosser cheetah swimmer glider-regular glider-medium glider-hard walker-regular walker-medium walker-hard"
EXPT="crawler-box-flip-small-small"
SEED=0
DES="" # continous_x90deg_flips moving_goal_fixed_box_inital fixed_goal_fixed_box

OMP_NUM_THREADS=1 python -m design_opt.test -m cfg=$EXPT n="$N" \
    seed=$SEED \
    hydra.sweep.dir="/home/knierim/results/$MODEL/$EXPT/$SEED/$(date "+%Y-%m-%d-%H-%M-%S-%N")" \
    group=$MODEL \
    name=$EXPT-$SEED
