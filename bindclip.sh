

data_path="data/train_no_test_af"

param_dir="hingeloss_lmdba1.5_seed42"
save_dir="ckpt/savedir${param_dir}"

tmp_save_dir="ckpt/tmp/tmp_save_dir${param_dir}"
tsb_dir="ckpt/tsb/tsb_dir${param_dir}"

n_gpu=1
MASTER_PORT=10055
finetune_mol_model="unimol/ckp/mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="unimol/ckp/pocket_pre_220816.pt" # unimol pretrained pocket model


batch_size=48
batch_size_valid=64
epoch=200
dropout=0.0
warmup=0.06
update_freq=4
dist_threshold=8.0
recycling=3
lr=1e-3

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0"  $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 0 --ddp-backend=c10d \
       --task drugclip --loss in_batch_softmax_hinge --arch drugclip  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 42 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-last-epochs 5 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
