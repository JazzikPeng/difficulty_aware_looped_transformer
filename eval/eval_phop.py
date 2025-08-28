"""
Evaluate Phop Tasks.
1. Evaluate on 16 hops
2. Evaluate on only the last answers
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT, GPTLooped

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-phop-16-looped' # ignored if init_from is not 'resume'
model_output_name = 'ckpt_2_0_6_random.pt'
batch_size = 256
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 80 # number of tokens generated in each sample
temperature = 0.1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# Loop config
num_loops = 6
loop_start = 0
loop_func = 'z=f(x+z)'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, model_output_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    print("checkpoint model args:", checkpoint['model_args'])
    print("looped model:", 'looped' in out_dir)
    model = GPTLooped(gptconf, num_loops=num_loops, loop_start=loop_start, loop_func=loop_func) if 'looped' in out_dir else GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Read example input sequence
with open('data/phop/p_hop_sequences_test.txt', 'r') as f:
    p_hop_sequences = f.readlines()

# Convert p_hop_sequences to tensor for model input
data = torch.tensor([list(map(int, seq.split())) for seq in p_hop_sequences], device=device)
# Split into x and y
# Find index of 3 
start_idx = torch.where(data[0]==3)[0].item()
x = data[:, :start_idx+1]  # all but last token
y = data[:, start_idx+1:start_idx+1+17]    # last token as target

# Convert evaluation to batches for model inference
pred_batches = []

with torch.no_grad():
    with ctx:
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            x_batch = x[start:end]  # take a batch of inputs
            
            pred = model.generate(
                x_batch,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            pred_batches.append(pred)  
            print("Running batch", x_batch.size(0), "/", x.size(0))

pred = torch.cat(pred_batches, dim=0)
print(pred.shape)

# # run generation
# with torch.no_grad():
#     with ctx:
#         pred1 = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#         print('---------------')
    
    
# with torch.no_grad():
#     with ctx:
#         pred = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#         print('---------------')

            
# print("pred_batch and pred are the same:", torch.equal(pred1, pred))

# Evaluation methods 1: Evaluate only on the last results at p
# Find elements after len(x)
last_results = pred[:, x.size(1):][:, y.size(1) - 1]
# Compare with y last result
last_token_acc = (last_results == y[:, -1]).sum().float() / last_results.size(0)
print(f'Accuracy on last results: {last_token_acc.item() * 100:.2f}%')

# Evaluation methods 2: Evaluate on all tokens
all_results = pred[:, x.size(1):][:, :y.size(1)]
# Compare with y all results
all_token_acc = (all_results == y).sum().float() / all_results.numel()
print(f'Accuracy on all results: {all_token_acc.item() * 100:.2f}%')

# Evaluate methods 3: Both last and intermediate results needs to be correct
# Check if all tokens in the last 17 are correct
correct_last_tokens = (all_results == y).all(dim=1)
correct_last_tokens_acc = correct_last_tokens.sum().float() / correct_last_tokens.size(0)
print(f'Accuracy on both last tokens and intermediate tokens: {correct_last_tokens_acc.item() * 100:.2f}%')

# Evaluate methods 4: Plot the acc as y axis and the number of hops as x axis
import matplotlib.pyplot as plt
accuracies = []
y = data[:, start_idx+1:]    # last token as target
all_results = pred[:, x.size(1):][:, :y.size(1)]
hops = list(range(51))  # 0 to 49 hops
for hop in hops:
    correct_hop_tokens = (all_results[:, hop] == y[:, hop]).sum().float() / pred.size(0)
    accuracies.append(correct_hop_tokens.item())

assert y.size(1) == len(accuracies), "Hops and accuracies must have the same length"

print("accuracies:", len(accuracies))

# Create a better plot with improved x-axis
plt.figure(figsize=(12, 6))
plt.plot(hops, accuracies, marker='o', linewidth=2, markersize=4)
plt.title('Accuracy vs Number of Hops', fontsize=14, fontweight='bold')
plt.xlabel('Number of Hops', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
# Add a vertical line at 16 hops
plt.axvline(x=16, color='red', linestyle='--', label='16 Hops')

# Better x-axis tick handling
if len(hops) <= 20:
    # Show all ticks if reasonable number
    plt.xticks(hops)
else:
    # Show every 5th tick for better readability
    tick_positions = [i for i in hops if i % 5 == 0]
    plt.xticks(tick_positions)

# Optional: Rotate x-axis labels if needed
plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()

# Save plot with higher DPI
plot_path = f'./eval_plots/accuracy_vs_hops_{model_output_name}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{plot_path}'")

# Orgnize results into a dictionary and write out to a file
results = {f'{out_dir}_{model_output_name}': {
    'experiment_description': f'{out_dir}_{model_output_name}',
    'last_token_accuracy': round(last_token_acc.item(), 6),
    'all_tokens_accuracy': round(all_token_acc.item(), 6),
    'both_tokens_accuracy': round(correct_last_tokens_acc.item(), 6),
    'hops': hops,
    'accuracies': accuracies,
    'eval_samples': data.size(0),
    'plot_path': plot_path,
}}
# Write results to a file as json
import json
results_file = './eval_plots/eval_results.json'
# If the key already exists in the file, overwrite it. Else append to the file
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        try:
            existing_results = json.load(f)
        except (json.JSONDecodeError, ValueError):
            # File is empty or contains invalid JSON
            existing_results = {}
    existing_results.update(results)
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=4)
else:
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

print(f"Results saved to '{results_file}'")
# -----------------------------------------------------------------------------