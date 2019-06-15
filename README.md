# Active Shift Layer

pytorch [Constructing Fast Network through Deconstruction of Convolution](https://papers.nips.cc/paper/7835-constructing-fast-network-through-deconstruction-of-convolution.pdf) implementation

> **It does not support CUDA now.**

## Prerequisite

* Windows10
* Pytorch 1.1.0
* cuda 9.0

## Build
```console
???@???:~$ python ./DepthwiseAffineGrid/setup.py install
???@???:~$ python ./DepthwiseGridSample/setup.py install
```

## To DO
- [x] use C++ extension
- [x] Write Backward
- [ ] Support CUDA tensor
- [ ] Comparison with paper
- [ ] Improve performance using CUDA
