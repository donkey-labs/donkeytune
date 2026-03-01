# QuickStart Guide

Copy the 2 `jsonl` files from `samples/hawaiian-pizza-dataset` in `data/` directory, then run the following commands to train your model to be an Hawaiian Pizza expert.

```bash
make setup
# wait for the setup to complete, it may take a while

make train MODEL=unsloth/gemma-3-270m-it
# the model is downloaded from HuggingFace,
# it's better to create a .env file with the following content:
#
# HF_TOKEN=<your-huggingface-token>
#
# this will speed up the download.
# In case of problem with the training (like "Python quit unexpectedly"), you can try to reduce the batch size by setting `BATCH_SIZE` variable, for example:
# make train MODEL=unsloth/Qwen2.5-1.5B-Instruct BATCH_SIZE=1

make convert MODEL=unsloth/gemma-3-270m-it
# this will convert the trained model to GGUF format in `output/gguf/` directory.

make test-prompt MODEL=unsloth/gemma-3-270m-it PROMPT="Was Hawaiian pizza immediately popular when it was invented?"
make test-prompt MODEL=unsloth/gemma-3-270m-it PROMPT="Why is it called Hawaiian pizza if it's not from Hawaii?"
```
