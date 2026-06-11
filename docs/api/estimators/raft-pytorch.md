# RAFT32 PyTorch estimator

PyTorch-backed **RAFT32-PIV** (learning-based flow). This is the default RAFT
estimator (`raft_torch`, also registered explicitly as `raft_torch_32`); it
runs on 32×32 interrogation windows. The larger-window sibling is
[RAFT256 PyTorch](raft-pytorch-256.md).

RAFT extracts per-pixel features from both frames and builds a 4D
correlation volume that scores every pair of locations,
$C_{ijkl} = \langle f_1(i,j),\, f_2(k,l) \rangle$. A recurrent (GRU) update
operator then repeatedly looks up correlation values around the current
estimate and refines a dense flow field; the output is the field after a
fixed number of recurrent updates. The weights are learned from data.

**Reference:** Z. Teed, J. Deng, *RAFT: Recurrent All-Pairs Field Transforms
for Optical Flow*, ECCV 2020
([arXiv:2003.12039](https://arxiv.org/abs/2003.12039)).

```{eval-rst}
.. automodule:: flowgym.flow.raft.raft_piv_pytorch
   :members:
```
