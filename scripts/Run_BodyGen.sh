N="range(1)"
MODEL="BodyGenExtend"
# EXPTS="crawler terraincrosser cheetah swimmer glider-regular glider-medium glider-hard walker-regular walker-medium walker-hard"
EXPT="pusher"
SEED=0

OMP_NUM_THREADS=1 python -m design_opt.train -m cfg=$EXPT n="$N" \
    seed=$SEED \
    hydra.sweep.dir="multirun/$MODEL-$EXPT/$(date "+%Y-%m-%d-%H-%M-%S-%N")" \
    group=$MODEL-$EXPT \
    name=$EXPT
