# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # all positives are 17 nt long, so setting seq length to 17
    seq_length = 17

    # convert to numpy arrays for easier indexing
    seqs = np.array(seqs)
    labels = np.array(labels)

    # separate positive and negative sequences
    pos_seqs = seqs[labels == 1]  # 1 = positive sequences
    neg_seqs = seqs[labels == 0]  # 0 = negative sequences

    # trim negative sequences to 17 nt
    trimmed_neg_seqs = []
    for seq in neg_seqs:
        if len(seq) >= seq_length:
            """
            looping through each negative sequence and if it's longer than 17 (and it will be lol),
            then choose a random starting point in the sequence and go 16 bases out to have a random 17nt negative seq
            after, append to list
            """
            start_idx = np.random.randint(0, len(seq) - seq_length + 1)
            trimmed_neg_seqs.append(seq[start_idx: start_idx + seq_length])

    # ensure equal sampling from both classes
    num_samples = min(len(pos_seqs), len(trimmed_neg_seqs))

    # sample from positives and trimmed negatives
    pos_indices = np.random.choice(len(pos_seqs), num_samples, replace=True)
    neg_indices = np.random.choice(len(trimmed_neg_seqs), num_samples, replace=True)

    # select sequences
    """
    sampled_seqs = list of positive seqs + list of negative sequences
    sampled labels = list of positive labels + list of negative labels (not multiplying by 1/0 -- these are the labels)
    """
    sampled_seqs = list(pos_seqs[pos_indices]) + [trimmed_neg_seqs[i] for i in neg_indices]
    sampled_labels = [1] * num_samples + [0] * num_samples 

    # randomly shuffing dataset (removes order bias between positive and negative samples)
    indices = np.arange(len(sampled_seqs)) # making an array of indices using # of sampled seqs
    np.random.shuffle(indices) # randomly rearranges the order of indices 

    # reorder sequences and labels using shuffled indices 
    """
    sampled_ = []
    for i in indices:
        sampled_.append(i)
    """
    sampled_seqs = [sampled_seqs[i] for i in indices]
    sampled_labels = [sampled_labels[i] for i in indices]

    return sampled_seqs, sampled_labels

    
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # dictionary for mapping nucleotides
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    """neural networks expect inputs to be the same size"""
    num_sequences = len(seq_arr) # number of seqs in seqs_arr
    sequence_length = len(seq_arr[0]) # should be 17

    # initializing empty array with shape(num_sequences (rows), sequence_length (cols are each nt), 4 (one hot encode))
    encoded_seqs = np.zeros((num_sequences, sequence_length, 4), dtype=int)

    for i, seq in enumerate(seq_arr): # loop over seqs
        for j, nuc in enumerate(seq): # looping over nt in the current seq
            encoded_seqs[i, j] = mapping[nuc] # assign one hot encoding

    return encoded_seqs