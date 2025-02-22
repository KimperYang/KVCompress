'''
For segment1, the number refers to the chunk id, and the 0 refers to the token has global attention.
For segment2, the number 1 refers to the memory token, the number 2 refers to the compress token, the number 3 refers to the remaining token which can only attend to number 2.
'''
import torch

def _bool_mask_first_rule(
    source_segments: torch.Tensor,
    target_segments: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a boolean tensor indicating where the first rule is satisfied:
      1. target_seg == 0 => True for all source positions.
      2. otherwise => (target_seg == source_seg).
    Shape: [B, ..., T, S].
    """
    # Broadcast shapes to [B, ..., T, S].
    # bool_mask[i, j] = (target_segments[i] == source_segments[j]) or target_segments[i] == 0
    t = target_segments.unsqueeze(-1)  # [B, ..., T, 1]
    s = source_segments.unsqueeze(-2)  # [B, ..., 1, S]

    same_segment_mask = (t == s)
    target_is_zero = (t == 0)

    # The final mask is True if "target is 0" or "segments match."
    bool_mask = target_is_zero | same_segment_mask
    return bool_mask

def _bool_mask_second_rule(
    source_segments: torch.Tensor,
    target_segments: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a boolean tensor for the second rule:
      - If target_seg == 1 => can attend to source_seg == 1
      - If target_seg == 2 => can attend to source_seg in {1, 2}
      - If target_seg == 3 => can attend to source_seg in {2, 3}
    Shape: [B, ..., T, S].
    """
    t = target_segments.unsqueeze(-1)  # shape [B, ..., T, 1]
    s = source_segments.unsqueeze(-2)  # shape [B, ..., 1, S]

    # Build masks for each target-segment condition
    target_is_1 = (t == 1)
    target_is_2 = (t == 2)
    target_is_3 = (t == 3)

    # For source segments:
    source_is_1 = (s == 1)
    source_is_2 = (s == 2)
    source_is_3 = (s == 3)

    # Now build boolean conditions:
    # - If target is 1 => source must be 1
    # - If target is 2 => source can be 1 or 2
    # - If target is 3 => source can be 2 or 3
    can_attend_1 = target_is_1 & source_is_1
    can_attend_2 = target_is_2 & (source_is_1 | source_is_2)
    can_attend_3 = target_is_3 & (source_is_2 | source_is_3)

    return can_attend_1 | can_attend_2 | can_attend_3

def make_segment_mask_with_two_rules(
    *,
    # First set of segment IDs
    source_segments_1: torch.Tensor,
    target_segments_1: torch.Tensor,
    # Second set of segment IDs
    source_segments_2: torch.Tensor,
    target_segments_2: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    add_causal_lm_mask: bool = True,
) -> torch.Tensor:
    """
    Build an attention logit bias that satisfies BOTH segment rules:
      1. The "first segment" rule (same as make_segment_mask).
      2. The "second segment" rule (segment 1->1, 2->(1 or 2), 3->(2 or 3)).

    A position is only allowed if both rules allow it.
    Returns: [batch, seq_len, seq_len] float tensor with 0 (allowed) or -∞ (masked).
    """

    # 1) Prepare an initial matrix with the causal mask if desired
    NEG_INF = -float('inf')
    batch_size = source_segments_1.size(0)
    seq_len = source_segments_1.size(-1)  # assuming shape: [B, seq_len]

    if add_causal_lm_mask:
        segment_logit_bias = torch.triu(
            torch.full((batch_size, seq_len, seq_len), NEG_INF,
                       dtype=dtype, device=source_segments_1.device),
            diagonal=1
        )
        # Triu(diagonal=1) => upper triangle is -∞, lower triangle + diag is 0.
    else:
        segment_logit_bias = torch.zeros((batch_size, seq_len, seq_len),
                                         dtype=dtype,
                                         device=source_segments_1.device)

    # 2) Create boolean masks for rule #1 and rule #2.
    bool_mask_1 = _bool_mask_first_rule(source_segments_1, target_segments_1)
    bool_mask_2 = _bool_mask_second_rule(source_segments_2, target_segments_2)

    # Both need to be True to allow attention
    combined_bool_mask = bool_mask_1 & bool_mask_2  # shape [B, seq_len, seq_len]

    # 3) Handle invalid/padded tokens => segment == -1
    #    If source or target is -1, we want to mask out entirely.
    source_is_invalid = (source_segments_1 == -1).unsqueeze(-2) | (source_segments_2 == -1).unsqueeze(-2)
    target_is_invalid = (target_segments_1 == -1).unsqueeze(-1) | (target_segments_2 == -1).unsqueeze(-1)
    invalid_mask = source_is_invalid | target_is_invalid

    # 4) Combine invalid_mask with our combined_bool_mask
    #    If invalid, we want to mask out => set True in all_masks.
    all_masks = invalid_mask | (~combined_bool_mask)

    # 5) Fill with -∞
    segment_logit_bias.masked_fill_(all_masks, NEG_INF)

    return segment_logit_bias

# seg_1 = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]])  
# seg_2 = torch.tensor([[1, 1, 2, 3, 3, 1, 1, 2, 3, 3]])   

# seg_1 = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]])  
# seg_2 = torch.tensor([[3, 3, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 3, 3]])

# seg_1 = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1, -1], [1, 1, 1, 2, 2, 2, -1, -1, -1, -1, -1, -1]])  
# seg_2 = torch.tensor([[1, 1, 2, 3, 3, 1, 1, 2, 3, 3, -1, -1], [1, 2, 3, 1, 2, 3, -1, -1, -1, -1, -1, -1]]) 

# # Build the combined mask
# mask = make_segment_mask_with_two_rules(
#     source_segments_1=seg_1,
#     target_segments_1=seg_1,
#     source_segments_2=seg_2,
#     target_segments_2=seg_2,
#     dtype=torch.float,           # For printing, let's keep float
#     add_causal_lm_mask=True     # No causal mask for clarity
# )
# print("Mask shape:", mask.shape)  # Should be [1, 5, 5]
# print("Mask matrix (0 or -inf):\n", mask[0])
# print("Mask matrix (0 or -inf):\n", mask[1])