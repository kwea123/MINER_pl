# MINER_pl
Unofficial implementation of [MINER: Multiscale Implicit Neural Representations](https://arxiv.org/pdf/2202.03532.pdf) in pytorch-lightning.

![image](https://user-images.githubusercontent.com/11364490/168208863-656a0a7d-35d9-4b9b-86f9-d52da4182e35.png)

### Only image reconstruction task is implemented currently.

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
*  The paper says it uses sinusoidal activation (does he mean SIREN? I don't know), but I use gaussian activation (in hidden layers) with trainable parameters (per block) like my experiments in [the other repo](https://github.com/kwea123/Coordinate-MLPs). In finer levels where the model predicts laplacian pyramids, I use sinusoidal activation `x |-> sin(ax)` with trainable parameters `a` (per block) as output layer. Moreover, I precompute the maximum amplitude for laplacian residuals, and use that to scale the output, and I find it to be better than without scaling.
*  Some difference in the hyperparameters: the default learning rate is `3e-2` instead of `5e-4`. Optimizer is `RAdam` instead of `Adam`. Block pruning happens when the loss is lower than `1e-4` (i.e. when PSNR>=40) rather than `2e-7`. Feel free to adjust these hyperparameters, but I find them to be optimal for a `patch_wh` around `(32, 32)`. Larger or smaller patch sizes might need parameter tuning.

# :computer: Installation

*  Run `pip install -r requirements.txt`.
*  Download the images from [Acknowledgement](#gift_heart-acknowledgement) or prepare your own images into a folder called `images`.

# :key: Training

Pluto example (8GB mem required):
```python3
python train.py \
    --image_path images/pluto.png
    --img_wh 4096 4096 --patch_wh 32 32 --batch_size 256 --n_scales 4 \
    --num_epochs 500 \
    --exp_name pluto4k_4scale 
```

Tokyo station example (11GB mem required):
```python3
python train.py \
    --image_path images/tokyo-station.jpg
    --img_wh 6000 4000 --patch_wh 25 25 --batch_size 192 --n_scales 5 \
    --num_epochs 500 \
    --exp_name tokyo4k_5scale 
```

The original image will be resized to `img_wh` for reconstruction. You need to make sure `img_wh` divided by `2^(n_scales-1)` (the resolution at the coarsest level) is still a multiple of `patch_wh`.

For full options, please see [here](opt.py). If your GPU memory is not enough, try reducing `batch_size`.

You are recommended to monitor the training progress by
```
tensorboard --logdir logs
```

where you can see training curves and images.

# :red_square::green_square::blue_square: Block decomposition

To visualize block decomposition per scale like Fig. 4 in the paper, see [block_visualization.ipynb](block_visualization.ipynb).

Examples:
![pluto4k_5scale_lap](https://user-images.githubusercontent.com/11364490/168275200-e625d828-61df-4ff2-a658-7dd10e123847.jpg)
![tokyo6k_5scale_lap](https://user-images.githubusercontent.com/11364490/168275208-a35e828d-0ca0-408f-90c3-89dd97d108ba.jpg)

<!-- # Testing

To reconstruct the image using trained model, see [test.ipynb](test.ipynb). -->

# :bulb: Implementation tricks

*  As suggested in *training details* on page 4, I implement parallel block inference by defining parameters of shape `(n_blocks, n_in, n_out)` and use `@` operator (same as `torch.bmm`) for faster inference.
*  Since `nn.Parameter` doesn't support pruning (like `p.require_grad_(False)`) per layer, I manually save and reload the weights for pruned blocks during validation [start](https://github.com/kwea123/MINER_pl/blob/master/train.py#L115-L121) and [end](https://github.com/kwea123/MINER_pl/blob/master/train.py#L183-L187). Attention that **only zeroing the gradient does not work**, the reason is detailed [here](https://discuss.pytorch.org/t/freezing-part-of-the-layer-weights/9457/19?u=kwea123).

# :gift_heart: Acknowledgement

*  Pluto image: [NASA](https://solarsystem.nasa.gov/resources/933/true-colors-of-pluto/?category=planets/dwarf-planets_pluto)

*  Shibuya image: [Trevor Dobson](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837)

*  Tokyo station image: [baroparo](https://pixabay.com/photos/tokyo-station-tokyo-station-japan-641769/?download)
