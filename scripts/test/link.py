

compress_tokens = list(range(128011, 128061))
max_chunk_num = 10
link_token_num = 2

link_token_start = compress_tokens[-1] + 1
link_tokens = [
    [
        link_token_start + idx * link_token_num + offset
        for offset in range(link_token_num)
    ]
    for idx in range(max_chunk_num)
]

print(link_tokens)