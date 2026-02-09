results_path="./test"  # replace to your results path
batch_size=256
#weight_path="ckpt/savedirrandom_zinc/checkpoint_best_seed-42.pt"
weight_path="checkpoint_best_seed-42.pt"
TASK="OOD"

CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1  CUDA_VISIBLE_DEVICES="0" python ./unimol/test_ood.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 0 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \
