import torch

# def make_chunked_summary_mask(
#     *,
#     first_segments:  torch.Tensor,   # [B, L]   1-based ids, −1 = PAD
#     second_segments: torch.Tensor,   # [B, L]   {1,2},  2 = “summary”, −1 = PAD
#     chunk_ids:       torch.Tensor,   # [B, L]   0,1,2,… real chunks; −1 = “global”
#     dtype: torch.dtype = torch.bfloat16,
#     allow_self: bool = True,
# ) -> torch.Tensor:
#     """
#     Returns an attention-logit bias of shape [B, L, L] (target row, source col).

#     Rules (all positions j must also satisfy causal j ≤ i):

#       •  Inside the same chunk (chunk_id >= 0):
#             allowed ⇔  same first-segment
#                         OR source second-segment == 2   (a summary token)

#       •  Across chunks (different chunk_id, both ≥0):
#             never allowed.

#       •  Target tokens with chunk_id == −1  (“global” tokens):
#             may attend to any *preceding* source token that is
#             –  a summary token   (second == 2)                 ╮
#             –  OR another global token (source chunk_id == −1) ├─ ★ new
#                                                               ┘

#       •  Any token whose first or second segment is −1 (padding)
#         cannot attend or be attended.
#     """
#     NEG_INF = -float("inf")
#     device   = first_segments.device
#     B, L     = first_segments.shape

#     # ------------------------------------------------- causal: j ≤ i
#     idx      = torch.arange(L, device=device)
#     causal   = (idx.unsqueeze(0) <= idx.unsqueeze(1)).expand(B, L, L)
#     if not allow_self:
#         causal &= ~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)

#     # ------------------------------------------------- helpers (broadcast)
#     tgt_first   = first_segments.unsqueeze(-1)        # [B,L,1]
#     src_first   = first_segments.unsqueeze(-2)        # [B,1,L]
#     same_first  = (tgt_first == src_first)

#     src_sum     = (second_segments == 2).unsqueeze(-2)  # [B,1,L]

#     tgt_chunk   = chunk_ids.unsqueeze(-1)             # [B,L,1]
#     src_chunk   = chunk_ids.unsqueeze(-2)             # [B,1,L]
#     same_chunk  = (tgt_chunk == src_chunk) & (tgt_chunk >= 0)

#     is_tgt_glb  = (tgt_chunk == -1)
#     is_src_glb  = (src_chunk == -1)

#     # ------------------------------------------------- allowed matrices
#     tgt_sum = (second_segments == 2).unsqueeze(-1)      # [B,L,1]   is the target a summary?
#     src_sum = (second_segments == 2).unsqueeze(-2)      # [B,1,L]   is the source a summary?

#     inside_chunk = same_chunk & (
#         (~tgt_sum & (same_first | src_sum))             # non-summary target
#         | (tgt_sum & same_first)                        # summary     target
#     )

#     global_read  = is_tgt_glb & (src_sum | is_src_glb)   # ★ changed line
#     allowed      = inside_chunk | global_read

#     # ------------------------------------------------- padding
#     pad_src = ((first_segments == -1) | (second_segments == -1)).unsqueeze(-2)
#     pad_tgt = ((first_segments == -1) | (second_segments == -1)).unsqueeze(-1)
#     allowed &= ~(pad_src | pad_tgt)

#     # ------------------------------------------------- final bias
#     allowed &= causal
#     bias = torch.full((B, L, L), NEG_INF, dtype=dtype, device=device)
#     bias.masked_fill_(allowed, 0.0)
#     return bias

NEG_INF = -float("inf")


def make_chunk_aug_mask(
    *,
    # ------------- first (“classic”) segment -----------------
    source_seg1: torch.Tensor,   # [B, S]
    target_seg1: torch.Tensor,   # [B, T]
    # ------------- second segment (values ∈ {1,2,3}) ---------
    source_seg2: torch.Tensor,   # [B, S]
    target_seg2: torch.Tensor,   # [B, T]
    # ------------- chunk ids ---------------------------------
    source_chunk: torch.Tensor,  # [B, S]
    target_chunk: torch.Tensor,  # [B, T]
    dtype: torch.dtype = torch.bfloat16,
    add_causal_lm_mask: bool = True,
) -> torch.Tensor:
    """
    Returns an attention-bias matrix of shape [B, T, S] whose entries are
    0.0 (allowed) or −∞ (masked).  A pair (target i, source j) is allowed
    **iff** all of the following hold:

    ───────────────────────────── Rule 1 (seg1) ───────────────────────────
      • target_seg1[i] == 0        ⇒ ignore seg1 and pass (subject to Rule 2)
      • otherwise                  ⇒ target_seg1[i] == source_seg1[j]

    ───────────────────────────── Rule 2 (seg2) ───────────────────────────
      Let t2 ≜ target_seg2[i] and s2 ≜ source_seg2[j].

        • t2 == 1  →  allowed only if (s2 == 1)           and Rule 1 passes
        • t2 == 2  →  allowed if (Rule 1 passes)  OR
                            (s2 == 2  ∧  same-chunk  ∧  j ≤ i)
        • t2 == 3  →  allowed if (s2 == 3  ∧  Rule 1 passes)  OR
                            (s2 == 2  ∧  j ≤ i)          # chunk-agnostic

      (Tokens whose seg2 == 3 are expected to carry chunk_id == −1, which
       is harmless because “same-chunk” is never checked for them.)

    Padding tokens are indicated by seg1 == −1 or seg2 == −1 and are always
    masked out.

    `add_causal_lm_mask=True` additionally forbids attending to *future*
    positions (upper-triangle set to −∞).
    """
    B, S = source_seg1.shape          # source length
    T = target_seg1.shape[-1]         # target length
    dev = source_seg1.device

    # ---------------------------------------------------- 1· causal mask
    if add_causal_lm_mask:
        bias = torch.triu(
            torch.full((B, T, S), NEG_INF, dtype=dtype, device=dev),
            diagonal=1
        )
    else:
        bias = torch.zeros((B, T, S), dtype=dtype, device=dev)

    # ---------------------------------------------------- 2· Rule-1 mask
    t1 = target_seg1.unsqueeze(-1)                 # [B,T,1]
    s1 = source_seg1.unsqueeze(-2)                 # [B,1,S]
    rule1_mask = (t1 == 0) | (t1 == s1)            # [B,T,S]

    # ------------------------------------------------------------------ #
    # 3. rule-2  +  chunk logic   ( FINAL spec – 28 Apr 2025 )            #
    # ------------------------------------------------------------------ #
    t2 = target_seg2.unsqueeze(-1)          # [B, T, 1]
    s2 = source_seg2.unsqueeze(-2)          # [B, 1, S]

    tgt_is1, tgt_is2, tgt_is3 = (t2 == 1), (t2 == 2), (t2 == 3)
    src_is1, src_is2, src_is3 = (s2 == 1), (s2 == 2), (s2 == 3)

    # helper: “source position precedes target position”
    t_idx = torch.arange(T, device=dev).unsqueeze(-1)   # [T,1]
    s_idx = torch.arange(S, device=dev).unsqueeze(0)    # [1,S]
    preceding = (t_idx >= s_idx).unsqueeze(0)           # [1,T,S]  → broadcast

    # helper: same chunk id
    same_chunk = (target_chunk.unsqueeze(-1) == source_chunk.unsqueeze(-2))

    # ─── target sec-rule == 1 ────────────────────────────────────────── #
    # • must satisfy rule-1 , source sec-rule ==1 ,  **and same chunk**
    allow1 = tgt_is1 & src_is1 & rule1_mask & same_chunk

    # ─── target sec-rule == 2 ────────────────────────────────────────── #
    # allowed if
    #   (a) rule-1 is satisfied **and same chunk**,   OR
    #   (b) source sec-rule ==2  **and same chunk**  and preceding
    allow2_a = tgt_is2 & rule1_mask & same_chunk
    allow2_b = tgt_is2 & src_is2 & same_chunk & preceding
    allow2 = allow2_a | allow2_b

    # ─── target sec-rule == 3 ────────────────────────────────────────── #
    # • may attend ONLY  *preceding* tokens whose sec-rule ==2
    #   (chunk / first-rule are irrelevant; tokens with sec-rule==3 get chunk_id = −1)
    allow3 = tgt_is3 & (src_is2 | src_is3) & preceding

    # combined rule-2 mask
    rule2_mask = allow1 | allow2 | allow3          # [B, T, S]


    # ---------------------------------------------------- 4· pad / invalid
    src_bad = (source_seg1 == -1).unsqueeze(-2) | (source_seg2 == -1).unsqueeze(-2)
    tgt_bad = (target_seg1 == -1).unsqueeze(-1) | (target_seg2 == -1).unsqueeze(-1)
    invalid = src_bad | tgt_bad

    # ---------------------------------------------------- 5· final fill
    disallowed = invalid | (~rule2_mask)
    bias.masked_fill_(disallowed, NEG_INF)
    return bias



B, L = 1, 10                        # one batch, 3 chunks (0,1,2) + 2 globals
# first  = torch.tensor([[1,1,2,2,  1,1,2,2, 1, 1]])
# second = torch.tensor([[1,2,1,2,  1,2,1,2, 1, 1]])
# chunks = torch.tensor([[0,0,0,0,  1,1,1,1, -1, -1]])  # last index (9) is global

first  = torch.tensor([[0,0,1,1,1,1,2,2,2,3,3,0,0, -1]])
second = torch.tensor([[3,3,1,1,1,2,1,1,2,1,2,3,3, -1]])
chunks = torch.tensor([[-1,-1,0,0,0,0,0,0,0,0,0,-1,-1, -1]])  # last index (9) is global

# mask = make_chunked_summary_mask(
#     first_segments=first,
#     second_segments=second,
#     chunk_ids=chunks,
#     dtype=torch.float,
# )

mask = make_chunk_aug_mask(
    source_seg1=first,
    target_seg1=first,
    source_seg2=second,
    target_seg2=second,
    source_chunk=chunks,
    target_chunk=chunks,
    dtype=torch.float,
    add_causal_lm_mask=True
)

print(mask)       # row = target, col = source
