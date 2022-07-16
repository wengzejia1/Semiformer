ROOT=/share/home/jia/workspace/semiformer-codeclean/Semiformer
DATA_ROOT=/share/common/ImageDatasets/imagenet_2012
CKPT=$ROOT/exp

cd $ROOT

# Train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT=$ROOT/semiformer

python -m torch.distributed.launch --master_port 50130 --nproc_per_node=8 --use_env semi_main_concat_evalVersion.py \
                                   --model Semiformer_small_patch16 \
                                   --data-set SEMI-IMNET \
                                   --batch-size 108 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path $DATA_ROOT \
                                   --epochs 300 \
                                   --data-split-file $ROOT/data_splits/files2shards_train_size128116_split1.txt \
                                   --mu 0.125 \
                                   --temperature 1.0 \
                                   --threshold 0.7 \
                                   --semi-lambda 4.0 \
                                   --evaluate-freq 4 \
                                   --semi-start-epoch 30 \
                                   --mixup 0.0 \
                                   --cutmix 0.0 \
                                   --no-repeated-aug \
                                   --pseudo-type cnn \
                                   --output_dir ${OUTPUT} \


