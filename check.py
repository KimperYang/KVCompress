import torch

def make_chunked_summary_mask(
    *,
    first_segments:  torch.Tensor,   # [B, L]   1-based ids, −1 = PAD
    second_segments: torch.Tensor,   # [B, L]   {1,2},  2 = “summary”, −1 = PAD
    chunk_ids:       torch.Tensor,   # [B, L]   0,1,2,… real chunks; −1 = “global”
    dtype: torch.dtype = torch.bfloat16,
    allow_self: bool = True,
) -> torch.Tensor:
    """
    Returns an attention-logit bias of shape [B, L, L] (target row, source col).

    Rules (all positions j must also satisfy causal j ≤ i):

      •  Inside the same chunk (chunk_id >= 0):
            allowed ⇔  same first-segment
                        OR source second-segment == 2   (a summary token)

      •  Across chunks (different chunk_id, both ≥0):
            never allowed.

      •  Target tokens with chunk_id == −1  (“global” tokens):
            may attend to any *preceding* source token that is
            –  a summary token   (second == 2)                 ╮
            –  OR another global token (source chunk_id == −1) ├─ ★ new
                                                              ┘

      •  Any token whose first or second segment is −1 (padding)
        cannot attend or be attended.
    """
    NEG_INF = -float("inf")
    device   = first_segments.device
    B, L     = first_segments.shape

    # ------------------------------------------------- causal: j ≤ i
    idx      = torch.arange(L, device=device)
    causal   = (idx.unsqueeze(0) <= idx.unsqueeze(1)).expand(B, L, L)
    if not allow_self:
        causal &= ~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)

    # ------------------------------------------------- helpers (broadcast)
    tgt_first   = first_segments.unsqueeze(-1)        # [B,L,1]
    src_first   = first_segments.unsqueeze(-2)        # [B,1,L]
    same_first  = (tgt_first == src_first)

    src_sum     = (second_segments == 2).unsqueeze(-2)  # [B,1,L]

    tgt_chunk   = chunk_ids.unsqueeze(-1)             # [B,L,1]
    src_chunk   = chunk_ids.unsqueeze(-2)             # [B,1,L]
    same_chunk  = (tgt_chunk == src_chunk) & (tgt_chunk >= 0)

    is_tgt_glb  = (tgt_chunk == -1)
    is_src_glb  = (src_chunk == -1)

    # ------------------------------------------------- allowed matrices
    tgt_sum = (second_segments == 2).unsqueeze(-1)      # [B,L,1]   is the target a summary?
    src_sum = (second_segments == 2).unsqueeze(-2)      # [B,1,L]   is the source a summary?

    inside_chunk = same_chunk & (
        (~tgt_sum & (same_first | src_sum))             # non-summary target
        | (tgt_sum & same_first)                        # summary     target
    )

    global_read  = is_tgt_glb & (src_sum | is_src_glb)   # ★ changed line
    allowed      = inside_chunk | global_read

    # ------------------------------------------------- padding
    pad_src = ((first_segments == -1) | (second_segments == -1)).unsqueeze(-2)
    pad_tgt = ((first_segments == -1) | (second_segments == -1)).unsqueeze(-1)
    allowed &= ~(pad_src | pad_tgt)

    # ------------------------------------------------- final bias
    allowed &= causal
    bias = torch.full((B, L, L), NEG_INF, dtype=dtype, device=device)
    bias.masked_fill_(allowed, 0.0)
    return bias


B, L = 1, 10                        # one batch, 3 chunks (0,1,2) + 2 globals
# first  = torch.tensor([[1,1,2,2,  1,1,2,2, 1, 1]])
# second = torch.tensor([[1,2,1,2,  1,2,1,2, 1, 1]])
# chunks = torch.tensor([[0,0,0,0,  1,1,1,1, -1, -1]])  # last index (9) is global

first  = torch.tensor([[1,1,1,1,2,2,2,3],[1,1,1,1,2,2,2,3]])
second = torch.tensor([[1,1,1,2,1,1,2,1],[1,1,1,2,1,1,2,1]])
chunks = torch.tensor([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])  # last index (9) is global

mask = make_chunked_summary_mask(
    first_segments=first,
    second_segments=second,
    chunk_ids=chunks,
    dtype=torch.float,
)
print(mask)       # row = target, col = source
