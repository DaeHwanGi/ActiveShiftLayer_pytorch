# Active Shift Layer

pytorch [Constructing Fast Network through Deconstruction of Convolution](https://papers.nips.cc/paper/7835-constructing-fast-network-through-deconstruction-of-convolution.pdf) implementation

## Prerequisite

* Pytorch 1.1.0
* cuda 9.0

## Build
```console
???@???:~$ python ./DepthwiseAffineGrid/setup.py install
???@???:~$ python ./DepthwiseGridSample/setup.py install
```

## To DO
- [v] use C++ extension
- [ ] Write Backward
- [ ] improve performance using CUDA
- [ ] Comparison with paper
