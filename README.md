# MINER_pl
Unofficial implementation of [MINER: Multiscale Implicit Neural Representations](https://arxiv.org/pdf/2202.03532.pdf) in pytorch-lightning.

**Only image reconstruction task is implemented.**

# :warning: Differences w.r.t. the original paper:
*  Should be the biggest difference: I didn't find laplacian pyramid to be useful. Precisely, I find gaussian pyramid performs on-par with laplacian pyramid, so I totally discard the laplacian pyramid.
*  In the pseudo code on page 8, where the author states "Weight sharing for images", it means finer levels' networks are initialized with coarser level network weights. However, I find this to produce inferior quality, no matter laplacian or gaussian pyramid is used. Therefore, I initialize the network weights from scratch for all levels.
*  The paper uses sinusoidal activation (does he mean SIREN? I don't know), but I use gaussian activation (in hidden layers) with trainable parameters (per block) like my experiments in [the other repo](https://github.com/kwea123/Coordinate-MLPs).
*  Some difference in the hyperparameters: the default learning rate is `3e-2` instead of `5e-4`. Optimizer is `RAdam` instead of `Adam`. Block pruning happens when the loss is lower than `1e-4` rather than `2e-7`. Feel free to adjust these hyperparameters, but I find them to be optimal. 

# Installation

*  Run `pip install -r requirements.txt`.
*  Install [apex](https://github.com/NVIDIA/apex#linux)
*  Download the images from [Acknowledgement](#acknowledgement) or prepare your own images into a folder called `images`.

# Train image reconstruction


# Reference reading

This should be the core of the paper... However I don't find it better than simple gaussian pyramid.
[Laplacian pyramid explanation](https://paperswithcode.com/method/laplacian-pyramid)

# Acknowledgement

*  Pluto image: [NASA](https://solarsystem.nasa.gov/resources/933/true-colors-of-pluto/?category=planets/dwarf-planets_pluto)

*  Shibuya image: [Trevor Dobson](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837)

*  Tokyo station image: [baroparo](https://pixabay.com/photos/tokyo-station-tokyo-station-japan-641769/?download)