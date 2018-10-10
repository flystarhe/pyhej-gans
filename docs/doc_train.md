# Train

## Train pix2pix
```commandline
python train_pix2pix.py \
--gpu_ids 0 \
--checkpoints_dir /data2/tmps/gans_pix2pix \
--json_file /data2/datasets/2017_1130_ct_dalian/dataset_base_180808_pair.txt \
--aligned true \
--batch_size 1 \
--input_nc 1 \
--workers 4 \
--conv_dim_d 64 \
--n_layers_d 6 \
--use_sigmoid \
--conv_dim_g 64 \
--n_blocks_g 6 \
--use_bias \
--init_type normal \
--resume_iters 0 \
--start_iters 1 \
--train_iters 2 \
--model_save 1 \
--lr 0.0002 \
--lr_update_step 1 \
--lr_update_gamma 0.5 \
--display_freq 1
```
