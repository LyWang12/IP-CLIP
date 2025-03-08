SCcd ..

DATA=data # change your data path here
MODE=train

DATASET=officehome # dataset name; officehome, visda17, mini_domainnet
TRAINER=ADCLIPB16 # ADCLIPRN50, ADCLIPB16, ADCLIPL14
CFG=vitB16  # config file; rn50, vitB16, vitL14
#SEED=$4

for SEED in 1 2 3 4 5
do
    DIR=output/${DATASET}/${MODE}/${TRAINER}/${CFG}/${S2T}/${METHOD}/seed_${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python trainsh.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR}
    fi
done