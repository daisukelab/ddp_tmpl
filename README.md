# DDP template

This is for learning DDP.

- https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py

```
wget https://raw.githubusercontent.com/mlfoundations/open_clip/main/src/training/distributed.py
```

```
$ OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py
2023-11-20,18:22:52 | INFO | Added key: store_based_barrier_key:1 to store for rank: 0
2023-11-20,18:22:52 | INFO | Added key: store_based_barrier_key:1 to store for rank: 1
2023-11-20,18:22:52 | INFO | Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
2023-11-20,18:22:52 | INFO | Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
2023-11-20,18:22:52 | INFO | Running in distributed mode with multiple processes. Device: cuda:1.Process (global: 1, local 1), total 2.
2023-11-20,18:22:52 | INFO | Running in distributed mode with multiple processes. Device: cuda:0.Process (global: 0, local 0), total 2.
2023-11-20,18:22:57 | INFO | Model:
2023-11-20,18:22:57 | INFO | TestNetwork(
  (net): Linear(in_features=768, out_features=768, bias=True)
)
2023-11-20,18:22:57 | INFO | TestNetwork(
  (net): Linear(in_features=512, out_features=512, bias=True)
)
2023-11-20,18:22:57 | INFO | rank=0: x.shape=torch.Size([64, 768]), x[:2, 0]=tensor([0.2307, 0.9086], device='cuda:0'), y.shape=torch.Size([64, 512]), y[:2, 0]=tensor([0.8680, 0.3645], device='cuda:0')
2023-11-20,18:22:57 | INFO | rank=1: x.shape=torch.Size([64, 768]), x[:2, 0]=tensor([0.4374, 0.6311], device='cuda:1'), y.shape=torch.Size([64, 512]), y[:2, 0]=tensor([0.5973, 0.8499], device='cuda:1')
2023-11-20,18:22:57 | INFO | rank=0: all_x.shape=torch.Size([128, 768]), [:2, 0]=tensor([382.6943, 390.0041], device='cuda:0', grad_fn=<SelectBackward0>), all_y.shape=torch.Size([128, 512]), [:2, 0]=tensor([255.4693, 256.3419], device='cuda:0', grad_fn=<SelectBackward0>)
2023-11-20,18:22:57 | INFO | rank=1: all_x.shape=torch.Size([128, 768]), [:2, 0]=tensor([389.6259, 385.8355], device='cuda:1', grad_fn=<SelectBackward0>), all_y.shape=torch.Size([128, 512]), [:2, 0]=tensor([252.2272, 253.5870], device='cuda:1', grad_fn=<SelectBackward0>)
```