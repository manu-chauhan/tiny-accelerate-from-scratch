# tiny-huggingface-accelerate
A minimal working 🤗 accelerate


## `DataParallel.py` has:
1. A simple bare minimum ManualNaiveDataParallel implementation, for distributed all_reduce for all parameters' gradients with `async_op=True`
2. A simple DataParallelBucket implementation which buckets gradients and uses distributed all-reduce operation.