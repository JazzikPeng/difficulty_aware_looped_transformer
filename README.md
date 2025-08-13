# difficulty_aware_looped_transformer
Last Update: 2025-08-12

Find a training / Inference framework to use the optimal number of loops for problem with various level of difficulty.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3


# Dataset Generation
## p-Hop Induction Task
In this task we define a p-hop induction task with three difficulty levels:

```
.
├── LICENSE
├── README.md
└── data
    └── phop
```


| Level | p  | vocab_size | seq_len | num_loops (l) |
|------:|---:|-----------:|--------:|--------------:|
| 1     | 16 | 4          | 256     | 3             |
| 2     | 32 | 8          | 512     | 6             |
| 3     | 64 | 16         | 1024    | 12            |


First randomly sample and generate the dataset in txt format for all three difficulty levels. 
Each contains train/test and eval data. Eval data contains 100 p-hop steps.

For each level, I will create 4M training samples. I then train a looped transformer on those data using different number of loops.
I will make sure the model BLOCK_SIZE is greater than 1024 for each level to handle any seqeunce up to 1024 sequence length. 
I will then evaluate the model on the eval data with different number of loops for each level of difficulty.

# Experiment and Training

# TODO





