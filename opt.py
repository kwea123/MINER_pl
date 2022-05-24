import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='image',
                        choices=['image', 'mesh'],
                        help='which reconstruction to perform')
    parser.add_argument('--path', type=str, default='images/pluto.png',
                        help='path to the object to reconstruct')
    parser.add_argument('--patch_size', nargs="+", type=int, default=[32, 32],
                        help='resolution of each patch')
    parser.add_argument('--n_scales', type=int, default=1,
                        help='number of laplacian pyramid levels')
    parser.add_argument('--input_size', nargs="+", type=int, default=[512, 512],
                        help='resolution of the input')
    parser.add_argument('--arch', type=str, default='mlp',
                        choices=['mlp', 'gabor'],
                        help='use which model architecture')
    parser.add_argument('--pyr', type=str, default='laplacian',
                        choices=['gaussian', 'laplacian'],
                        help='use which pyramid')

    parser.add_argument('--use_pe', action='store_true', default=False,
                        help='use positional encoding for uv')
    parser.add_argument('--n_freq', type=int, default=4,
                        help='number of frequencies of positional encoding')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of layers in each MLP')
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='number of hidden units in each MLP')
    parser.add_argument('--a', type=float, default=0.1,
                        help='initial a for gaussian activation')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per block, smaller than @patch_size product')
    parser.add_argument('--b_chunks', type=int, default=16384,
                        help='inputs are split into chunks of at most this number of blocks')
    parser.add_argument('--lr', type=float, default=3e-2,
                        help='learning rate')
    parser.add_argument('--num_epochs', nargs="+", type=int, default=[500],
                        help='list of number of epochs for each scale')
    parser.add_argument('--loss_thr', type=float, default=1e-4,
                        help='''stop training a block if loss is lower than this,
                        typically 1e-4 for image and 5e-3 or 1e-2 for mesh''')

    parser.add_argument('--val_freq', type=int, default=50,
                        help='validate (and prune blocks) every N epochs')
    # only for task=='image'
    parser.add_argument('--log_image', action='store_true', default=False,
                        help='whether to log image to tensorboard (might be slow for large images)')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()