import torch 
import random
from src.data.attention import make_segment_mask_with_two_rules

def process_text(system_id, document_ids, chunk_compress_tokens, link_tokens, user_id):
    segment_ids_1 = []
    segment_ids_2 = []
    labels = []
    output_sequence = []
    position_ids = []

    chunk_compress_token_len = len(chunk_compress_tokens)

    sys_len = len(system_id)

    output_sequence.extend(system_id)
    segment_ids_1.extend([0] * sys_len)
    segment_ids_2.extend([3] * sys_len)
    labels.extend([-100] * sys_len)
    position_ids.extend(list(range(sys_len)))

    link_token_num = len(link_tokens)

    current_index = sys_len

    for j in range(len(document_ids)):
        tem_id = document_ids[j]

        segment_ids_1.extend([j+1] * (len(tem_id) + chunk_compress_token_len) + [0] * link_token_num)
        segment_ids_2.extend([1] * len(tem_id) + [2] * chunk_compress_token_len + [3] * link_token_num)
        labels.extend([-100] * (len(tem_id) + chunk_compress_token_len + link_token_num))
        position_ids.extend(list(range(current_index - len(tem_id), current_index + chunk_compress_token_len + link_token_num)))
        output_sequence.extend(tem_id + chunk_compress_tokens + link_tokens)

        current_index += chunk_compress_token_len + link_token_num

    # user_id = ["user"]
    user_len = len(user_id)
    segment_ids_1.extend([0] * user_len)
    segment_ids_2.extend([3] * user_len)
    labels.extend([-100] * user_len)
    position_ids.extend(list(range(current_index, current_index + user_len)))
    output_sequence.extend(user_id)
    current_index += user_len

    return segment_ids_1, segment_ids_2, position_ids, labels, output_sequence


# system_id = ["sys", "sys"]
# document_ids = [
#         ["chunk1_1", "chunk1_2", "chunk1_3"],
#         ["chunk2_1", "chunk2_2", "chunk2_3"]
#     ]
# chunk_compress_tokens = ["comp"]
# link_tokens = ["link"]
# user_id = ["user"]

system_id = ["sys"] * random.randint(1, 10)
document_ids = [[f"chunk{i+1}"] * random.randint(50, 100) for i in range(random.randint(6, 10))]
chunk_compress_tokens = ["comp"] * random.randint(10,20)
link_tokens = ["link"] * random.randint(1,5)
user_id = ["user"] * random.randint(20,30)

segment_ids_1, segment_ids_2, position_ids, labels, output_sequence = process_text(system_id=system_id, 
        document_ids=document_ids, chunk_compress_tokens=chunk_compress_tokens, link_tokens=link_tokens, user_id=user_id)


# print(segment_ids_1)
# print(segment_ids_2)

print("Input ids:\n", output_sequence)
print("Labels:\n", labels)
print("Position ids:\n", position_ids)

# assert position_ids == [0, 1, -1, 0, 1, 2, 3, 1, 2, 3, 4, 5, 6]

mask = make_segment_mask_with_two_rules(
    source_segments_1=torch.tensor([segment_ids_1]),
    target_segments_1=torch.tensor([segment_ids_1]),
    source_segments_2=torch.tensor([segment_ids_2]),
    target_segments_2=torch.tensor([segment_ids_2]),
    dtype=torch.bfloat16,
    add_causal_lm_mask=True
)

print("Attention mask:\n",mask)

# Test
for i in range(len(output_sequence)):
    for j in range(i+1):
        # If the token is a normal global token, it attends to all the compression token or preceding normal token.
        if segment_ids_2[i] == 3 and segment_ids_1[i] == 0 and (segment_ids_2[j] == 2 or segment_ids_2[j] == 3):
            assert mask[0][i][j] == float(0)
        # If the token is a chunk token, it attends to all the chunk tokens in the same chunk.
        elif segment_ids_2[i] == 1 and segment_ids_2[j] == 1 and segment_ids_1[i] == segment_ids_1[j]:
            assert mask[0][i][j] == float(0)
        # If the token is a compression token, it attends to the preceding compression tokens and chunk tokens in the same chunk.
        elif segment_ids_2[i] == 2 and (segment_ids_2[j] == 1 or segment_ids_2[j] == 2) and (segment_ids_1[i] == segment_ids_1[j]):
            assert mask[0][i][j] == float(0)
        else:
            assert mask[0][i][j] == float('-inf')