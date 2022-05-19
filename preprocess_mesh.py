import argparse
import trimesh
from einops import rearrange, reduce
import numpy as np
from pysdf import SDF
from tqdm import tqdm
import os


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='meshes/bunny.ply',
                        help='path to the object to reconstruct')

    parser.add_argument('--N', type=int, default=1024,
                        help='resolution (N^3) of the mesh, same for xyz')
    parser.add_argument('--M', type=int, default=1,
                        help='''number of samples inside each cell to predict
                        gt occupancy value. Larger value yields more precise result.
                        Must be a ODD CUBIC number (typically 1 or 27).
                        ''')
    return parser.parse_args()
    

if __name__ == '__main__':
    args = get_opts()
    N = args.N
    M = args.M # samples inside each cell
    assert M%2 == 1, 'M must be an odd cubic number!!'

    o = trimesh.load(args.path)
    bbox = np.amax(o.vertices, 0)-np.amin(o.vertices, 0)
    mesh_whl = bbox/2
    o.vertices -= np.amax(o.vertices, 0)-mesh_whl # center the mesh
    mesh_whl *= 1.02 # give some margin
    f = SDF(o.vertices, o.faces)

    cbrtM = int(M**(1/3))
    xs = np.linspace(-mesh_whl[0], mesh_whl[0], cbrtM*N)
    ys = np.linspace(-mesh_whl[1], mesh_whl[1], cbrtM*N)
    zs = np.linspace(-mesh_whl[2], mesh_whl[2], cbrtM*N)
    occ = np.zeros((N, N, N, 1), bool)
    print('computing occupancy values ...')
    for i, z in enumerate(tqdm(zs[::cbrtM])):
        xyz_ = np.stack(np.meshgrid(xs, ys, zs[i*cbrtM:(i+1)*cbrtM]), -1).reshape(-1, 3)
        occ_ = f.contains(xyz_).reshape(cbrtM*N, cbrtM*N, cbrtM)
        occ_ = rearrange(occ_, '(h a) (w b) c -> (a b c) h w',
                         a=cbrtM, b=cbrtM, c=cbrtM, h=N, w=N).mean(0)
        occ[:, :, i, 0] = occ_>=0.5

    os.makedirs('occupancies', exist_ok=True)
    base = os.path.basename(args.path)
    save_path = f'occupancies/{os.path.splitext(base)[0]}_{N}.npy'
    np.save(save_path, occ)
    print(f'occupancy saved to {save_path} !')