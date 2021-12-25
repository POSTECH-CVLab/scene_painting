CUDA_VISIBLE_DEVICES=0 python train.py \
 --name surface_feat_059 \
 --dataset_mode blender \
 --label_nc 15 \
 --semantic_nc 16 \
 --gpu_ids 0 \
 --batch_size 8 \
 --label_dir blender_set_008_test_image/labels_blender \
 --coordinate_image_dir blender_set_008_test_image/coordinate_images \
 --pseudo_image_dir blender_set_008_test_image/labels_ade20k \
 --pseudo_label_dir blender_set_008_test_image/labels_ade20k \
 --dataroot blender_set_008_test_image \
 --real_image_dir ade20k_indoor_size256/images \
 --real_label_dir ade20k_indoor_size256/labels_blender \
 --freq_print 1000 \
 --freq_save_ckpt 1000 \
 --freq_save_latest 1000 \
 --freq_fid 10000 \
 --freq_smooth_loss 100 \
 --freq_save_loss 100 \
 --num_epochs 5000 \
 --no_EMA \
 --D_steps_per_G 1 \
 --pretrained_oasis_checkpoints_dir ./checkpoints \
 --model surface_feat \
 --init_type none \
 --surface_feat_model_convblock_type None \
 --surface_feat_model_l2_use_norm \
 --surface_feat_model_defocal_weight \
 --surface_feat_model_defocal_lambda 0.1 \
 --mlp_hdim 740 \
 --channels_D 3 64 64 128 128 256 512 \
 --z_mapping_type mapping_net \
 --z_mapping_dim 256 \
 --pos_encoding_model nerf \
 --pos_encoding_num_freq 4 \
 --coordinate_embedding_model none \
 --coordinate_embedding_dim -1 \
 --add_vgg_loss \
 --lambda_vgg 1.0 \
 --add_pseudo_recon_l1_loss \
 --lambda_pseudo_recon_l1 1.0 \
 --add_pseudo_recon_l2_loss \
 --lambda_pseudo_recon_l2 10.0 \
 --use_netD_output1 \
 --add_pseudo_adv_loss_output1 \
 --lambda_G_real_output1 0.1 \
 --lambda_D_fake_output1 0.1 \
 --lambda_D_pseudo_output1 0.1 \
 --lr_g 0.0001 \
 --lr_d 0.0001 \
 --discriminator oasis \
 --num_workers 4

# --label_dir blender_set_008_trainset/labels_blender \
# --coordinate_image_dir blender_set_008_trainset/coordinate_images \
# --pseudo_image_dir blender_set_008_trainset/labels_ade20k \
# --pseudo_label_dir blender_set_008_trainset/labels_ade20k \
# --dataroot blender_set_008_trainset \

