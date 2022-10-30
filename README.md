# MINER_pl
Unofficial implementation of [MINER: Multiscale Implicit Neural Representations](https://arxiv.org/pdf/2202.03532.pdf) in pytorch-lightning.

**Official implementation** : https://github.com/vishwa91/miner

![image](https://user-images.githubusercontent.com/11364490/168208863-656a0a7d-35d9-4b9b-86f9-d52da4182e35.png)

# :open_book: Ref readings

*  [Laplacian pyramid explanation](https://paperswithcode.com/method/laplacian-pyramid)

*  My explanatory videos

<p align="center">
  <a href="https://youtu.be/cXZtbfjnJtA">
    <img src="https://user-images.githubusercontent.com/11364490/168209075-330d879e-2bff-467f-bf31-4e0ad2809777.png", width="45%">
  </a>
  <a href="https://youtu.be/MSVEhq67Ca4">
    <img src="https://user-images.githubusercontent.com/11364490/168209233-4bde51ba-df6d-4fdb-87d6-9704986c1248.png", width="45%">
  </a>
</p>

# :warning: Main differences w.r.t. the original paper before continue:
*  In the pseudo code on page 8, where the author states **Weight sharing for images**, it means finer level networks are initialized with coarser level network weights. However, I did not find the correct way to implement this. Therefore, I initialize the network weights from scratch for all levels.
*  The paper says it uses sinusoidal activation (does he mean SIREN? I don't know), but I use gaussian activation (in hidden layers) with trainable parameters (per block) like my experiments in [the other repo](https://github.com/kwea123/Coordinate-MLPs). In finer levels where the model predicts laplacian pyramids, I use sinusoidal activation `x |-> sin(ax)` with trainable parameters `a` (per block) as output layer (btw, this performs *significantly* better than simple `tanh`). Moreover, I precompute the maximum amplitude for laplacian residuals, and use it to scale the output, and I find it to be better than without scaling.
*  I experimented with a common trick in coordinate mlp: *[positional encoding](https://github.com/tancik/fourier-feature-networks)* and find that using it can increase training time/accuracy with the same number of parameters (by reducing 1 layer). This can be turned on/off by specifying the argument `--use_pe`. The optimal number of frequencies depends on the patch size, the larger patch sizes, the more number of frequencies you need and vice versa.
*  Some difference in the hyperparameters: the default learning rate is `3e-2` instead of `5e-4`. Optimizer is `RAdam` instead of `Adam`. Block pruning happens when the loss is lower than `1e-4` (i.e. when PSNR>=40) for image and `5e-3` for occupancy rather than `2e-7`.

# :computer: Installation

*  Run `pip install -r requirements.txt`.
*  Download the images from [Acknowledgement](#gift_heart-acknowledgement) or prepare your own images into a folder called `images`.
*  Download the meshes from [Acknowledgement](#gift_heart-acknowledgement) or prepare your own meshes into a folder called `meshes`.

# :key: Training

<details>
  <summary><h2>image</summary>

Pluto example:
```python3
python train.py \
    --task image --path images/pluto.png \
    --input_size 4096 4096 --patch_size 32 32 --batch_size 256 --n_scales 4 \
    --use_pe --n_layers 3 \
    --num_epochs 50 50 50 200 \
    --exp_name pluto4k_4scale
```

Tokyo station example:
```python3
python train.py \
    --task image --path images/tokyo-station.jpg \
    --input_size 6000 4000 --patch_size 25 25 --batch_size 192 --n_scales 5 \
    --use_pe --n_layers 3 \
    --num_epochs 50 50 50 50 150 \
    --exp_name tokyo6k_5scale
```

| Image (size) | Train time (s) | GPU mem (MiB) | #Params (M) | PSNR |
|:---:|:---:|:---:|:---:|:---:|
| Pluto (4096x4096) | 53 | 3171 | 9.16 | 42.14 |
| Pluto (8192x8192) | 106 | 6099 | 28.05 | 45.09 |
| Tokyo station (6000x4000) | 68 | 6819 | 35.4 | 42.48 |
| Shibuya (7168x2560) | 101 | 8967 | 17.73 | 37.78 |
| Shibuya (14336x5120) | 372 | 8847 | 75.42 | 39.32 |
| Shibuya (28672x10240) | 890 | 10255 | 277.37 | 41.93 |
| Shibuya (28672x10240)* | 1244 | 6277 | 98.7 | 37.59 |

*paper settings (6 scales, each network has 4 layer with 9 hidden units)

The original image will be resized to `img_wh` for reconstruction. You need to make sure `img_wh` divided by `2^(n_scales-1)` (the resolution at the coarsest level) is still a multiple of `patch_wh`.

------------------------------------------

</details>
  
<details>
  <summary><h2>mesh</summary>

First, convert the mesh to N^3 occupancy grid by
```python3
python preprocess_mesh.py --N 512 --M 1 --T 1 --path <path/to/mesh> 
```
This will create N^3 occupancy to be regressed by the neural network.
For detailed options, please see [preprocess_mesh.py](preprocess_mesh.py). Typically, increase `M` or `T` if you find the resulting occupancy bad.

Next, start training (bunny example):
```python3
python train.py \
    --task mesh --path occupancy/bunny_512.npy \
    --input_size 512 --patch_size 16 --batch_size 512 --n_scales 4 \
    --use_pe --n_freq 5 --n_layers 2 --n_hidden 8 \
    --loss_thr 5e-3 --b_chunks 512 \
    --num_epochs 50 50 50 150 \
    --exp_name bunny512_4scale
```

------------------------------------------

</details>

For full options, please see [here](opt.py). Some important options:

*  If your GPU memory is not enough, try reducing `batch_size`.
*  By default it will not log intermediate images to tensorboard to save time. To visualize image reconstruction and active blocks, add `--log_image` argument.

You are recommended to monitor the training progress by
```
tensorboard --logdir logs
```

where you can see training curves and images.

# :red_square::green_square::blue_square: Block decomposition

To reconstruct the image using trained model and to visualize block decomposition per scale like Fig. 4 in the paper, see [image_test.ipynb](image_test.ipynb) or [mesh_test.ipynb](mesh_test.ipynb)

<!-- Pretrained models can be downloaded from [releases](https://github.com/kwea123/MINER_pl/releases). -->

Examples:
<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/168275200-e625d828-61df-4ff2-a658-7dd10e123847.jpg", width="45%">
  <img src="https://user-images.githubusercontent.com/11364490/168275208-a35e828d-0ca0-408f-90c3-89dd97d108ba.jpg", width="45%">
</p>
    
<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/169640414-9da542dc-2df5-4a46-b80a-9591e86f98b3.jpg", width="30%">
  <img src="https://user-images.githubusercontent.com/11364490/169640416-59e4391f-7377-4b7e-b103-715b25d7253c.jpg", width="30%">
  <img src="https://user-images.githubusercontent.com/11364490/169640742-49f4a43e-4705-4463-bbe4-822839220ddd.jpg", width="30%">
</p>

# :bulb: Implementation tricks

*  Setting `num_workers=0` in dataloader increased the speed a lot.
*  As suggested in *training details* on page 4, I implement parallel block inference by defining parameters of shape `(n_blocks, n_in, n_out)` and use `@` operator (same as `torch.bmm`) for faster inference.
*  To perform block pruning efficiently, I create two copies of the same network, and continually train and prune one of them while copying the trained parameters to the target network (somehow like in reinforcement learning, e.g. DDPG). This allows the network as well as the optimizer to shrink, therefore achieve higher memory and speed performance.
*  In validation, I perform inference in chunks like NeRF, and pass each chunk to cpu to reduce GPU memory usage.

# :gift_heart: Acknowledgement

*  Pluto image: [NASA](https://solarsystem.nasa.gov/resources/933/true-colors-of-pluto/?category=planets/dwarf-planets_pluto)

*  Shibuya image: [Trevor Dobson](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837)

*  Tokyo station image: [baroparo](https://pixabay.com/photos/tokyo-station-tokyo-station-japan-641769/?download)

*  [Stanford scanning](http://graphics.stanford.edu/data/3Dscanrep/)

*  [Turbosquid](https://www.turbosquid.com/)

# :question: Further readings

During a stream, my audience suggested me to test on this image with random pixels:

![random](https://user-images.githubusercontent.com/11364490/168567099-56c226a8-9f79-4710-9291-cc7ecef26f6d.png)

The default `32x32` patch size doesn't work well, since the texture varies too quickly inside a patch. Decreasing to `16x16` and increasing network hidden units make the network converge right away to `43.91 dB` under a minute. Surprisingly, with the other image reconstruction SOTA [instant-ngp](https://github.com/NVlabs/instant-ngp#image-of-einstein), the network is stuck at `17 dB` no matter how long I train.

![ngp-random](https://user-images.githubusercontent.com/11364490/168567783-40a9c123-01af-472c-9885-e52a778df18e.png)

Is this a possible weakness of instant-ngp? What effect could it bring to real application? You are welcome to test other methods to reconstruct this image!
