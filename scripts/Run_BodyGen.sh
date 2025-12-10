N="range(1)"
MODEL="BodyGenExtend"
# EXPTS="crawler terraincrosser cheetah swimmer glider-regular glider-medium glider-hard walker-regular walker-medium walker-hard"
EXPT="crawler-box-tilted-small-light-flip"
DES="v2"
SEED=1

OMP_NUM_THREADS=1 python -m design_opt.train -m cfg=$EXPT n="$N" \
    seed=$SEED \
    hydra.sweep.dir="/home/knierim/results/$MODEL/$EXPT/$SEED/$(date "+%Y-%m-%d-%H-%M-%S-%N")" \
    group=$EXPT \
    name=$EXPT-$SEED
