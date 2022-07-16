ROOT=/share/home/jia/workspace/semiformer-codeclean/Semiformer
DATA_ROOT=/share/common/ImageDatasets/imagenet_2012
RESUME_PATH=$ROOT/semiformer.pth

cd $ROOT

# Train
export CUDA_VISIBLE_DEVICES=0
OUTPUT=$ROOT/semiformer

python -m torch.distributed.launch --master_port 50131 --nproc_per_node=1 --use_env semi_main_concat_evalVersion.py \
                                   --model Semiformer_small_patch16 \
                                   --data-set SEMI-IMNET \
                                   --batch-size 256 \
                                   --lr 0.001 \
                                   --num_workers 4 \
                                   --data-path $DATA_ROOT \
                                   --data-split-file $ROOT/data_splits/files2shards_train_size128116_split1.txt \
                                   --mu 0.125 \
                                   --temperature 1.0 \
                                   --threshold 0.7 \
                                   --mixup 0.0 \
                                   --cutmix 0.0 \
                                   --no-repeated-aug \
                                   --output_dir ${OUTPUT} \
                                   --eval \
                                   --resume $RESUME_PATH \

