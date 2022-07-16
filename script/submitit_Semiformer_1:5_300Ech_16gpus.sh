ROOT=/projects/Semiformer
DATA_ROOT=/imagenet_path

PARTITION=${1:-"learnaccel"}

cd $ROOT

python semi_run_with_submitit_concat_evalVersion.py --model Semiformer_small_patch16 \
                            --data-set SEMI-IMNET \
                            --batch-size 108 \
                            --lr 0.001 \
                            --num_workers 4 \
                            --data-path $DATA_ROOT \
                            --epochs 300 \
                            --partition ${PARTITION} \
                            --shared_folder /checkpoint/$USER/experiments/Semiformer \
                            --data-split-file $ROOT/data_splits/files2shards_train_size128116_split1.txt \
                            --mu 0.16667 \
                            --temperature 1.0 \
                            --threshold 0.7 \
                            --semi-lambda 4.0 \
                            --evaluate-freq 10 \
                            --use_volta32 \
                            --semi-start-epoch 30 \
                            --nodes 2 \
                            --mixup 0.0 \
                            --cutmix 0.0 \
                            --no-repeated-aug \
                            --pseudo-type cnn \
