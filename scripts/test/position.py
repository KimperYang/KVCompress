segment_ids_1 = []
segment_ids_2 = []
labels = []
output_sequence = []
position_ids = []

system_id = [1, 2]
document_ids = [
    [3, 4, 5],
    [6, 7, 8]
]

chunk_compress_tokens = [-1]
chunk_compress_token_len = len(chunk_compress_tokens)

sys_len = len(system_id)

output_sequence.extend(system_id)
segment_ids_1.extend([0] * sys_len)
segment_ids_2.extend([3] * sys_len)
labels.extend([-100] * sys_len)
position_ids.extend(list(range(sys_len)))

link_tokens = [999]
link_token_num = 1

current_index = sys_len

for j in range(len(document_ids)):
    tem_id = document_ids[j]

    segment_ids_1.extend([j+1] * (len(tem_id) + chunk_compress_token_len) + [0] * link_token_num)
    segment_ids_2.extend([1] * len(tem_id) + [2] * chunk_compress_token_len + [3] * link_token_num)
    labels.extend([-100] * (len(tem_id) + chunk_compress_token_len + link_token_num))
    position_ids.extend(list(range(current_index - len(tem_id), current_index + chunk_compress_token_len + link_token_num)))
    output_sequence.extend(tem_id + chunk_compress_tokens + link_tokens)

    current_index += chunk_compress_token_len + link_token_num

user_id = [9]
user_len = len(user_id)
segment_ids_1.extend([0] * user_len)
segment_ids_2.extend([3] * user_len)
labels.extend([-100] * user_len)
position_ids.extend(list(range(current_index, current_index + user_len)))
output_sequence.extend(user_id)
current_index += user_len

ans_id = [10]
ans_len = len(ans_id)
segment_ids_1.extend([0] * ans_len)
segment_ids_2.extend([3] * ans_len)
labels.extend(ans_id)
position_ids.extend(list(range(current_index, current_index + ans_len)))
output_sequence.extend(ans_id)
print(segment_ids_1)
print(segment_ids_2)
print(position_ids)
print(labels)
print(output_sequence)