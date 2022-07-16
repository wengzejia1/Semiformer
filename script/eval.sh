ROOT=/share/home/jia/workspace/semiformer-codeclean/Semiformer
DATA_ROOT=/share/common/ImageDatasets/imagenet_2012
RESUME_PATH=$ROOT/semiformer.pth

cd $ROOT

export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --master_port 50131 --nproc_per_node=1 --use_env semi_main_concat_evalVersion.py \
                                   --model Semiformer_small_patch16 \
                                   --data-set SEMI-IMNET \
                                   --batch-size 256 \
                                   --num_workers 4 \
                                   --data-path $DATA_ROOT \
                                   --data-split-file $ROOT/data_splits/files2shards_train_size128116_split1.txt \
                                   --eval \
                                   --resume $RESUME_PATH \

