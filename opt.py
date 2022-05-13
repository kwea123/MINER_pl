import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='images/pluto.png',
                        help='path to the image to reconstruct')
    parser.add_argument('--patch_wh', nargs="+", type=int, default=[32, 32],
                        help='resolution (w, h) of each patch')
    parser.add_argument('--n_scales', type=int, default=1,
                        help='number of laplacian pyramid levels')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[512, 512],
                        help='''resolution (img_w, img_h) of the image,
                        must be a multiple of patch_wh*2**(n_scales-1)''')
    parser.add_argument('--pyr', type=str, default='laplacian',
                        choices=['gaussian', 'laplacian'],
                        help='use which image pyramid')

    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of layers in each MLP')
    parser.add_argument('--n_hidden', type=int, default=20,
                        help='number of hidden units in each MLP')
    parser.add_argument('--a', type=float, default=0.1,
                        help='initial a for gaussian activation')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per block, smaller than @patch_wh product')
    parser.add_argument('--b_chunk', type=int, default=16384,
                        help='split the blocks into this number per forward step')
    parser.add_argument('--lr', type=float, default=3e-2,
                        help='learning rate')
    parser.add_argument('--num_epochs', nargs="+", type=int, default=[500],
                        help='list of number of epochs for each scale')
    parser.add_argument('--loss_thr', type=float, default=1e-4,
                        help='stop training a block if loss is lower than this')

    parser.add_argument('--val_freq', type=int, default=100,
                        help='validate every N epochs')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()