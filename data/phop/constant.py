"""
This is a constant map from difficulty levels to L.
"""

# Difficulty
p_values = [16, 32, 64]
vocab_size = [4, 8, 16]
seq_len = [256, 512, 1024]
num_loops = [4, 8, 16]

difficulty_to_l = {
    (p_values[i], vocab_size[i], seq_len[i]) : num_loops[i] for i in range(len(p_values))
}


# Constants
BLOCK_SIZE = 1152
SPECIAL_MASK_TOKEN = -1