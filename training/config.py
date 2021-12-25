import argparse
import pickle
import os
import utils.utils as utils


def read_arguments(train=True, args=None):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser, train)
    parser.add_argument('--phase', type=str, default='train')

    opt = parser.parse_args(args)

    if train:
        set_dataset_default_lm(opt, parser)
        if opt.continue_train:
            update_options_from_file(opt, parser)
    opt = parser.parse_args()
    opt.phase = 'train' if train else 'test'
    if train:
        opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)
    utils.fix_seed(opt.seed)
    print_options(opt, parser)
    if train:
        save_options(opt, parser)
    return opt


def add_all_arguments(parser, train):
    # --- general options ---
    parser.add_argument('--name', type=str, default='label2coco',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--no_spectral_norm', action='store_true',
                        help='this option deactivates spectral norm in all layers')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--val_batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/', help='path to dataset root')
    parser.add_argument('--dataset_mode', type=str, default='coco',
                        help='this option indicates which dataset should be loaded')

    parser.add_argument('--label_dir', type=str, required=True,
                        help='path to the directory that contains label images')
    parser.add_argument('--coordinate_image_dir', type=str, required=True,
                        help='path to the directory that contains photo images')
    parser.add_argument('--pseudo_image_dir', type=str, required=True,
                        help='path to the directory that contains photo images')
    parser.add_argument('--pseudo_label_dir', type=str, required=True,
                        help='path to the directory that contains photo images')
    parser.add_argument('--real_image_dir', type=str, required=True,
                        help='path to the directory that contains photo images')
    parser.add_argument('--real_label_dir', type=str, required=True,
                        help='path to the directory that contains photo images')
    parser.add_argument('--class_specific_real_image_dir', type=str, 
                        help='path to the directory that contains photo images')
    parser.add_argument('--class_specific_real_label_dir', type=str,
                        help='path to the directory that contains photo images')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data argumentation')

    # for generator
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=64, help='# of gen filters in first conv layer in generator')
    parser.add_argument('--param_free_norm', type=str, default='syncbatch',
                        help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--no_EMA', action='store_true',
                        help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
    parser.add_argument('--no_3dnoise', action='store_true', default=False,
                        help='if specified, do *not* concatenate noise to label maps')

    parser.add_argument('--z_dim', type=int, default=64, help="dimension of the latent z vector")
    parser.add_argument('--z_mapping_type', type=str, default='none')
    parser.add_argument('--z_mapping_dim', type=int, default=128, help="dimension of the latent z vector")

    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--pos_encoding_num_freq', type=int, default=10)
    parser.add_argument('--coordinate_embedding_model', type=str, default='none')
    parser.add_argument('--coordinate_embedding_dim', type=int, default=63)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--gan_mode', type=str)
    parser.add_argument('--model', type=str)

    parser.add_argument('--label_nc', type=int)
    parser.add_argument('--semantic_nc', type=int)

    parser.add_argument('--G_conv_model', type=str, default='not')

    parser.add_argument('--mlp_hdim', type=int, default=256)
    parser.add_argument('--G_num_layers', type=int, default=7)
    parser.add_argument('--res_block_type', type=str, default='spade')

    parser.add_argument('--channels_D', nargs='+',
                        type=int, default=[3, 128, 128, 256, 256, 512, 512])

    # ablation
    parser.add_argument('--pos_encoding_model', type=str, default='nerf', choices=['none', 'nerf', 'fourier'])
    parser.add_argument('--fourier_dim', type=int)
    parser.add_argument('--fourier_scale', type=float)

    parser.add_argument('--fusion_model', type=str)

    parser.add_argument('--pretrained_oasis_checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--normalize_z_vec', action='store_true')
    parser.add_argument('--un_normalize_coord', action='store_true')
    parser.add_argument('--use_fixed_z_vec', action='store_true')
    parser.add_argument('--num_z_vec', type=int, default=9)
    parser.add_argument('--z_list', type=int, nargs='+', default=[-1])
                        

    parser.add_argument('--use_ade20k_pretrained_D', action='store_true')


    parser.add_argument('--use_refine_net', action='store_true')

    # adain_model
    parser.add_argument('--norm_type', type=str, default='adain')
    parser.add_argument('--use_conv_in_output1', action='store_true')
    parser.add_argument('--use_output2', action='store_true')

    parser.add_argument('--init_type', type=str, default='none')

    # surface_feat_model
    parser.add_argument('--surface_feat_model_convblock_type', type=str)
    parser.add_argument('--surface_feat_model_3dnoise', type=str)
    parser.add_argument('--surface_feat_model_defocal_weight', action='store_true')
    parser.add_argument('--surface_feat_model_defocal_lambda', type=float, default=0.5)
    parser.add_argument('--surface_feat_model_l2_use_norm', action='store_true')
    parser.add_argument('--surface_feat_model_view_encoding', action='store_true')
    parser.add_argument('--view_encoding_input_type', type=str)
    parser.add_argument('--view_encoding_use_type', type=str)
    parser.add_argument('--surface_feat_model_quantize', type=int, default=-1)
    parser.add_argument('--use_point_embedding', action='store_true')
    parser.add_argument('--point_embedding_dir', type=str)
    parser.add_argument('--point_embedding_dim', type=int, default=32)
    parser.add_argument('--point_embedding_num', type=int)
    parser.add_argument('--use_label_embedding', action='store_true')
    parser.add_argument('--label_embedding_dim', type=int, default=32)
    parser.add_argument('--use_3dfeat', action='store_true')
    parser.add_argument('--feat_dir', type=str, default='blender_set_008_3dfaet_maps_scale2')
    parser.add_argument('--feat_dim', type=int, default=16)

    # usis
    parser.add_argument('--usis_conv_type', type=str)

    # class specific
    parser.add_argument('--base_netG_name', type=str)
    parser.add_argument('--base_netG_ckpt_iter', type=str, default='best')

    if train:
        parser.add_argument('--freq_print', type=int, default=1000, help='frequency of showing training results')
        parser.add_argument('--freq_save_ckpt', type=int, default=20000, help='frequency of saving the checkpoints')
        parser.add_argument('--freq_save_latest', type=int, default=10000, help='frequency of saving the latest model')
        parser.add_argument('--freq_smooth_loss', type=int, default=250, help='smoothing window for loss visualization')
        parser.add_argument('--freq_save_loss', type=int, default=2500, help='frequency of loss plot updates')
        parser.add_argument('--freq_fid', type=int, default=5000,
                            help='frequency of saving the fid score (in training iterations)')
        parser.add_argument('--continue_train', action='store_true', help='resume previously interrupted training')
        parser.add_argument('--which_iter', type=str, default='latest', help='which epoch to load when continue_train')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='G learning rate, default=0.0001')
        parser.add_argument('--lr_d', type=float, default=0.0004, help='D learning rate, default=0.0004')

        parser.add_argument('--optim', type=str, default='adam')

        ### --- Loss ---
        parser.add_argument('--no_balancing_inloss', action='store_true', default=False,
                            help='if specified, do *not* use class balancing in the loss function')
        parser.add_argument('--no_labelmix', action='store_true', default=False,
                            help='if specified, do *not* use LabelMix')
        parser.add_argument('--lambda_labelmix', type=float, default=10.0, 
                            help='weight for LabelMix regularization')
        parser.add_argument('--add_vgg_loss', action='store_true', 
                            help='if specified, add VGG feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight for VGG loss')

        parser.add_argument('--add_real_adv_loss_output2', action='store_true')
#        parser.add_argument('--lambda_G_real_output2', type=float, default=1.0)
#        parser.add_argument('--lambda_D_fake_output2', type=float, default=1.0)
#        parser.add_argument('--lambda_D_real_output2', type=float, default=1.0)

        parser.add_argument('--loss_binary', default='gan', help='type of binary GAN loss')
        parser.add_argument('--add_real_binary_gan_loss_output2', action='store_true') 
        parser.add_argument('--add_pseudo_binary_gan_loss_output1', action='store_true') 
        parser.add_argument('--add_real_binary_gan_loss_output1', action='store_true') 
#        parser.add_argument('--lambda_G_binary_output2', type=float, default=1.0, 
#                            help='weight of binary GAN loss')
#        parser.add_argument('--lambda_D_binary_output2', type=float, default=1.0, 
#                            help='weight of binary GAN loss')

        parser.add_argument('--add_pseudo_recon_l1_loss', action='store_true')
        parser.add_argument('--lambda_pseudo_recon_l1', type=float, default=1.0)
        parser.add_argument('--add_pseudo_recon_l2_loss', action='store_true')
        parser.add_argument('--lambda_pseudo_recon_l2', type=float, default=1.0)
        parser.add_argument('--add_output12_recon_l1_loss', action='store_true')
        parser.add_argument('--lambda_output12_recon_l1', type=float, default=1.0)
        
        parser.add_argument('--use_netD_output1', action='store_true')
        parser.add_argument('--add_pseudo_adv_loss_output1', action='store_true')
        parser.add_argument('--add_real_adv_loss_output1', action='store_true')
        parser.add_argument('--lambda_G_real_output1', type=float, default=1.0)
        parser.add_argument('--lambda_D_fake_output1', type=float, default=1.0)
        parser.add_argument('--lambda_D_real_output1', type=float, default=1.0)
        parser.add_argument('--lambda_D_pseudo_output1', type=float, default=1.0)

        parser.add_argument('--add_pseudo_adv_loss_output2', action='store_true')
#        parser.add_argument('--lambda_D_pseudo_output2', type=float, default=1.0)

        # usis
        parser.add_argument('--add_r1_regularize', action='store_true')
        parser.add_argument('--add_ortho_regularize', action='store_true')
        parser.add_argument('--D_init', type=str, default='ortho')

        ### --- End Loss ---

        parser.add_argument('--use_diff_aug', action='store_true', help='use differentiable augmentation')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='Discriminator update step per Generator.')


        parser.add_argument('--use_G_spectral_norm', action='store_true')
        parser.add_argument('--use_D_dontcare_zero_mask', action='store_true')
        parser.add_argument('--use_D_dontcare_fake_mask', action='store_true')
        parser.add_argument('--use_D_input_cat_label', action='store_true')

        # adain_model
        parser.add_argument('--use_output_log', action='store_true')

        # surface_feat_model
        parser.add_argument('--discriminator', type=str, default='oasis')

        # omni_model
        parser.add_argument('--g_weight_decay', type=float, default=0.0)
        parser.add_argument('--d_weight_decay', type=float, default=0.0)
        parser.add_argument('--margin', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=1.0)
        
    else:
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves testing results here.')
        parser.add_argument('--ckpt_iter', type=str, default='best', help='which epoch to load to evaluate a model')
        parser.add_argument('--output_type', type=str)
    return parser


def set_dataset_default_lm(opt, parser):
    if opt.dataset_mode == "ade20k":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
    if opt.dataset_mode == "cityscapes":
        parser.set_defaults(lr_g=0.0004)
        parser.set_defaults(lambda_labelmix=5.0)
        parser.set_defaults(freq_fid=2500)
        parser.set_defaults(EMA_decay=0.999)
    if opt.dataset_mode == "coco":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
        parser.set_defaults(num_epochs=100)


def save_options(opt, parser):
    path_name = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)


def update_options_from_file(opt, parser):
    new_opt = load_options(opt.checkpoints_dir, opt.name)
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_options(checkpoints_dir, name):
    file_name = os.path.join(checkpoints_dir, name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    return new_opt


def load_iter(opt):
    if opt.which_iter == "latest":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    elif opt.which_iter == "best":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "best_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    else:
        return int(opt.which_iter)


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
