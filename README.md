# Implicit Feature Decoupling with Depthwise Quantization

code for paper: https://arxiv.org/pdf/2203.08080.pdf

This is minimal code demonstrating the performance advantage of DQ vs VQ.

![Model](/images/model.png)

## Depedencies

* python3.8

```
pip install -r requirements.txt
```
## Configuration

Modify `dq.config` to your experiment settings.
### Train Configuration
```python
dataset = cifar10
save_dir = path
device = cuda
epochs = 200
eval_itr = -1 # set to -1 to disable evaluation or the number of steps within an epoch i.e. 2 would be evaluating every 2 epochs and .5 would be evaluating 2 times per epoch
use_amp = True # automatic mixed precision
image_size = 32
batch_size = 128
lr = 2e-4
```

### Model Configuration
```python
in_channel = 3 # number of dataset image channels as input
channel = 256 # channels used in the hidden dimmension
n_res_block = 3
n_res_channel = 256 # res block channels
n_coder_blocks = 2 # number of res blocks
embed_dim = 64 # dimensionality of each codebook
n_codebooks = 5
stride = 2 # first block stride. Larger stride leads to smaller bottom-level codes
decay = 0.99 # vq decay
loss_name = mse # mse, ce and mix
vq_type = dq # dq or vq
beta = 0.25 # vq hyper-param
n_hier = [32,32,32] # number of codes per hierarchy. To define more hierarchies simply append to the list
n_logistic_mix = 10 # used only when loss_name = mix
```
### Test Configuration
```python
model_path = ""
dataset = "cifar10"
use_amp = False
image_size = 32
batch_size = 128
```
## Train DQ

```
python train_dq.py --config dq.config --device cuda
```


## Test DQ

```
python test_dq.py --config dq.config --device cuda
```

## Cite

```
@misc{https://doi.org/10.48550/arxiv.2203.08080,
  doi = {10.48550/ARXIV.2203.08080},

  url = {https://arxiv.org/abs/2203.08080},

  author = {Fostiropoulos, Iordanis and Boehm, Barry},

  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.10; I.2.10; H.1.1},

  title = {Implicit Feature Decoupling with Depthwise Quantization},

  publisher = {arXiv},

  year = {2022},

  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
