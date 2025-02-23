import torch


def matrix_orientation_criterion_padding(transfer_matrix, attn_matrix):
    # Determine the maximum size along each dimension
    max_shape = (
        max(transfer_matrix.shape[-4], attn_matrix.shape[-4]),
        max(transfer_matrix.shape[-3], attn_matrix.shape[-3]),
        max(transfer_matrix.shape[-2], attn_matrix.shape[-2]),
        max(transfer_matrix.shape[-1], attn_matrix.shape[-1])
    )

    # Pad both matrices to the maximum size
    transfer_padded = torch.nn.functional.pad(
        transfer_matrix,
        (0, max_shape[3] - transfer_matrix.shape[-1],
         0, max_shape[2] - transfer_matrix.shape[-2],
         0, max_shape[1] - transfer_matrix.shape[-3],
         0, max_shape[0] - transfer_matrix.shape[-4]),
        mode='constant',
        value=0
    )
    attn_padded = torch.nn.functional.pad(
        attn_matrix,
        (0, max_shape[3] - attn_matrix.shape[-1],
         0, max_shape[2] - attn_matrix.shape[-2],
         0, max_shape[1] - attn_matrix.shape[-3],
         0, max_shape[0] - attn_matrix.shape[-4]),
        mode='constant',
        value=0
    )

    # Create a mask to ignore the padded positions
    mask = (
            (transfer_padded != 0).float() *
            (attn_padded != 0).float()
    )

    # Compute the difference and apply the mask
    difference = (transfer_padded - attn_padded) * mask

    # Compute the Frobenius norm of the difference
    norm = torch.linalg.matrix_norm(difference, ord='fro')

    # Return the mean of the norm over the unmasked elements
    return norm.mean() #/ mask.sum()

def matrix_orientation_criterion_light(transfer_matrix, attn_matrix):
    # Determine the minimum and maximum sizes
    min_shape = (
        min(transfer_matrix.shape[-4], attn_matrix.shape[-4]),
        min(transfer_matrix.shape[-3], attn_matrix.shape[-3]),
        min(transfer_matrix.shape[-2], attn_matrix.shape[-2]),
        min(transfer_matrix.shape[-1], attn_matrix.shape[-1])
    )

    # Slice overlapping regions
    transfer_overlap = transfer_matrix[
        ..., :min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]
    ]
    attn_overlap = attn_matrix[
        ..., :min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]
    ]

    # Compute the squared difference over the overlapping regions
    difference = transfer_overlap - attn_overlap
    # Compute the Frobenius norm
    norm = torch.linalg.matrix_norm(difference, ord='fro')

    # Return the mean of the norm
    return norm.mean()

def matrix_orientation_criterion_light_with_extras(transfer_matrix, attn_matrix):
    # Determine the minimum and maximum sizes
    min_shape = (
        min(transfer_matrix.shape[-4], attn_matrix.shape[-4]),
        min(transfer_matrix.shape[-3], attn_matrix.shape[-3]),
        min(transfer_matrix.shape[-2], attn_matrix.shape[-2]),
        min(transfer_matrix.shape[-1], attn_matrix.shape[-1])
    )

    # Slice overlapping regions
    transfer_overlap = transfer_matrix[
        ..., :min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]
    ]
    attn_overlap = attn_matrix[
        ..., :min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]
    ]

    # Compute the squared difference over the overlapping regions
    overlap_loss = ((transfer_overlap - attn_overlap) ** 2).sum()

    # Compute loss for non-overlapping regions in transfer_matrix
    transfer_extra_loss = ((transfer_matrix[..., min_shape[0]:, :, :, :] ** 2).sum() +
                           (transfer_matrix[..., :, min_shape[1]:, :, :] ** 2).sum() +
                           (transfer_matrix[..., :, :, min_shape[2]:, :] ** 2).sum() +
                           (transfer_matrix[..., :, :, :, min_shape[3]:] ** 2).sum())

    # Compute loss for non-overlapping regions in attn_matrix
    attn_extra_loss = ((attn_matrix[..., min_shape[0]:, :, :, :] ** 2).sum() +
                       (attn_matrix[..., :, min_shape[1]:, :, :] ** 2).sum() +
                       (attn_matrix[..., :, :, min_shape[2]:, :] ** 2).sum() +
                       (attn_matrix[..., :, :, :, min_shape[3]:] ** 2).sum())

    # Total loss
    total_loss = overlap_loss + transfer_extra_loss + attn_extra_loss

    # Compute the Frobenius norm
    norm = torch.sqrt(total_loss)

    # Return the mean of the norm
    return norm.mean()

def matrix_orientation_criterion_MHA_confidence(transfer_matrix, attn_matrix):
    """
    Compute the difference between two attention matrices after pruning the MHA heads of the teacher's attention matrix
    to match the student's attention matrix.
    transfer_matrix: (batch_size, num_heads1, seq_len, seq_len)
    attn_matrix: (batch_size, num_heads2, seq_len, seq_len)
    """

    # prune attn_matrix to match the number of heads in transfer_matrix by selecting the heads with the highest attention scores and averageing the lowest exluded heads with the lowest included heads
    num_heads = transfer_matrix.shape[1]
    # calculate mean max weight for each head
    mean_max_weight = attn_matrix.max(dim=-1).values.mean(dim=-1)
    # sort the heads by mean max weight
    sorted_heads = torch.argsort(mean_max_weight, descending=True)
    # select the top num_heads heads
    attn_matrix_pruned = attn_matrix[:, sorted_heads[:num_heads], :, :]
    # average the lowest exluded heads with the lowest included heads

    # Compute the difference between the pruned attention matrix and the transfer matrix
    difference = transfer_matrix - attn_matrix_pruned

    # Compute the Frobenius norm of the difference
    norm = torch.linalg.matrix_norm(difference, ord='fro')

    # create random tensor like transfer_matrix

    return norm.mean()


def matrix_orientation_criterion_MHA_importance(transfer_matrix, attn_matrix, mask_head_index):
    # Compute the squared difference over the overlapping regions
    assert transfer_matrix.shape[1] == (attn_matrix.shape[1] -  mask_head_index.shape[0]), \
        "Number of heads in transfer_matrix should be equal to the number of heads in attn_matrix minus the number of heads to be pruned"
    difference = transfer_matrix - attn_matrix[:, mask_head_index, :, :]
    # Compute the Frobenius norm
    norm = torch.linalg.matrix_norm(difference, ord='fro')

    # Return the mean of the norm
    return norm.mean()


def hidden_states_alignment_criterion(hidden_states, teacher_hidden_states):
    return torch.norm(hidden_states - teacher_hidden_states, p=2, dim=(-1,)).mean()


def hidden_states_alignment_criterion_padding(hidden_states, teacher_hidden_states):
    # Determine the maximum size along the last dimension
    max_size = max(hidden_states.shape[-1], teacher_hidden_states.shape[-1])

    # Calculate padding sizes for each tensor
    pad_size_student = max_size - hidden_states.shape[-1]
    pad_size_teacher = max_size - teacher_hidden_states.shape[-1]

    # Pad both tensors to the maximum size
    hidden_states_padded = torch.nn.functional.pad(
        hidden_states,
        (0, pad_size_student),  # Padding on the last dimension
        mode='constant',
        value=0
    )
    teacher_hidden_states_padded = torch.nn.functional.pad(
        teacher_hidden_states,
        (0, pad_size_teacher),
        mode='constant',
        value=0
    )

    # Create masks to identify valid (non-padded) positions
    mask_student = torch.ones_like(hidden_states_padded, dtype=torch.bool)
    if pad_size_student > 0:
        mask_student[..., -pad_size_student:] = False
    mask_teacher = torch.ones_like(teacher_hidden_states_padded, dtype=torch.bool)
    if pad_size_teacher > 0:
        mask_teacher[..., -pad_size_teacher:] = False
    # Combined mask where both tensors have valid data
    mask = mask_student & mask_teacher

    # Compute the difference and apply the mask
    difference = hidden_states_padded - teacher_hidden_states_padded
    difference_masked = difference.masked_fill(~mask, 0)

    # Compute the L2 norm along the last dimension
    norm = torch.norm(difference_masked, p=2, dim=-1)

    norm_mean = norm.mean()

    return norm_mean

def hidden_states_alignment_criterion_light_with_extras(hidden_states, teacher_hidden_states):
    # Determine the minimum size along the last dimension
    min_size = min(hidden_states.shape[-1], teacher_hidden_states.shape[-1])

    # Slice both tensors to the minimum size
    hidden_states_sliced = hidden_states[..., :min_size]
    teacher_hidden_states_sliced = teacher_hidden_states[..., :min_size]

    # Compute the difference over the overlapping region
    difference = hidden_states_sliced - teacher_hidden_states_sliced

    # Compute the L2 norm along the last dimension for the overlapping region
    norm_overlap = torch.norm(difference, p=2, dim=-1)

    # Initialize total norm with the overlapping norm
    total_norm = norm_overlap

    # Handle non-overlapping regions in hidden_states
    if hidden_states.shape[-1] > min_size:
        extra_hidden_states = hidden_states[..., min_size:]
        norm_hidden_extra = torch.norm(extra_hidden_states, p=2, dim=-1)
        total_norm += norm_hidden_extra

    # Handle non-overlapping regions in teacher_hidden_states
    if teacher_hidden_states.shape[-1] > min_size:
        extra_teacher_hidden_states = teacher_hidden_states[..., min_size:]
        norm_teacher_extra = torch.norm(extra_teacher_hidden_states, p=2, dim=-1)
        total_norm += norm_teacher_extra

    # Compute the mean of the total norm
    norm_mean = total_norm.mean()

    return norm_mean


def hidden_states_alignment_criterion_light(hidden_states, teacher_hidden_states):
    # Determine the minimum size along the last dimension
    min_size = min(hidden_states.shape[-1], teacher_hidden_states.shape[-1])

    # Slice both tensors to the minimum size
    hidden_states_sliced = hidden_states[..., :min_size]
    teacher_hidden_states_sliced = teacher_hidden_states[..., :min_size]

    # Compute the difference over the overlapping region
    difference = hidden_states_sliced - teacher_hidden_states_sliced

    # Compute the L2 norm along the last dimension for the overlapping region
    norm_overlap = torch.norm(difference, p=2, dim=-1)

    return norm_overlap.mean()



def pad_weight(student_proj_weights, teacher_proj_weight, axis):
    # Pad the teacher's weights to match the student's shape
    padding_dim = student_proj_weights.shape[axis] - teacher_proj_weight.shape[axis]
    # If padding is needed
    if padding_dim > 0:
        padded_teacher_weight = torch.nn.functional.pad(teacher_proj_weight,
                                                        (0, padding_dim)
                                                        if axis == 1 else
                                                        (0, 0, 0, padding_dim),
                                                        mode='constant', value=0)
    else:
        #trim the teacher's weight
        padded_teacher_weight = teacher_proj_weight.narrow(axis, 0, student_proj_weights.shape[axis])
    return padded_teacher_weight
