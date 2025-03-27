import random
from typing import Dict
import math
import torch
from transformers import PreTrainedTokenizerBase

general_prompts = [
    "You are an AI assistant. Provide helpful, accurate, and clear answers. When uncertain, explain your reasoning or request clarification.",
    "You are an AI assistant. Focus on achieving the user's goal in each interaction. Use concise yet informative explanations.",
    "You are an AI assistant. Speak clearly and stay consistent with prior statements. If you need more information, politely ask for it.",
    "You are an AI assistant. Provide truthful, well-sourced information whenever possible. Acknowledge any limitations and avoid speculation if unsure."
]
qa_prompts = [
    "You are an AI assistant. Use the provided documents to answer the user’s question. If the information is insufficient, acknowledge the gap or request clarification.",
    "You are an AI assistant. Always ground your answers in the retrieved documents and do not add unsupported details. If the documents lack sufficient information, indicate that.",
    "You are an AI assistant. Rely solely on the given documents for evidence when answering questions. When necessary, cite or paraphrase the document content accurately.",
    "You are an AI assistant. Base your replies on the retrieved documents, ensuring completeness and correctness. Ask for more details if the documents do not cover the question fully."
]
summary_prompts = [
    "You are an AI assistant. Read the provided text and produce a concise summary. Capture the main points without unnecessary detail.",
    "You are an AI assistant. Summarize the essential ideas from the given text. Avoid minor details and focus on critical insights.",
    "You are an AI assistant. Provide a brief, high-level overview of the text. Ensure clarity and coherence, prioritizing key themes.",
    "You are an AI assistant. Summarize the text clearly and logically. Organize the main ideas in a coherent sequence."
]


class compress_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        compress_tokens: list[int],
        chunk_size:int,
        chunk_end_token: int,
        do_shuffle: bool,
        global_start_token: int = 128254,
        global_end_token: int = 128255,
        pad_token: int = 128004,
        link_token_num: int = 1,
        max_memory_num: int = 50
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.compress_tokens = compress_tokens
        self.chunk_size = chunk_size
        self.chunk_end_token = chunk_end_token
        self.do_shuffle = do_shuffle
        self.global_start_token = global_start_token
        self.global_end_token = global_end_token
        self.pad_token = pad_token
        self.link_token_num = link_token_num

        link_token_start = self.compress_tokens[-1] + 1
        self.link_tokens = [
            [
                link_token_start + idx * self.link_token_num + offset
                for offset in range(self.link_token_num)
            ]
            for idx in range(max_memory_num)
        ]

    # def process_pretraining(
    #     self,
    #     example
    # ):
    #     text_tokens = self.tokenizer(example['text']).input_ids[:self.max_len]
    #     output_sequence = text_tokens
    #     segment_ids_1 = [0] * len(text_tokens)
    #     segment_ids_2 = [3] * len(text_tokens)
    #     labels = text_tokens
    #     position_ids = list(range(len(text_tokens)))

    #     return {
    #         "input_ids": output_sequence,
    #         "segment_ids_1": segment_ids_1,
    #         "segment_ids_2": segment_ids_2,
    #         "labels": labels,
    #         "position_ids": position_ids,
    #     }

    def process_pretraining_repeat_compress(
        self,
        example
    ):
        text_tokens = self.tokenizer(example['text']).input_ids
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []
        chunk_counter = 0
        # 1. Split text_tokens into slices of size `self.chunk_size`.
        for i in range(0, len(text_tokens), self.chunk_size):

            chunk_counter += 1
            chunk = text_tokens[i : i + self.chunk_size]

            chunk_len = len(chunk)  # could be < self.chunk_size for the last chunk

            if chunk_len < self.chunk_size:
                break  # no more tokens

            # 2. Build the processed chunk
            #    chunk + [chunk_end_token] + self.compress_tokens + chunk
            processed_chunk = chunk + [self.chunk_end_token] + self.compress_tokens + chunk

            # 3. Check if adding this processed chunk would exceed max_length
            if len(output_sequence) + len(processed_chunk) > self.max_len:
                # If we can't add this chunk without exceeding,
                # we stop processing further.
                break

            segment_ids_1.extend([chunk_counter] * len(processed_chunk))

            segment_ids_2.extend([1] * (chunk_len + 1) + [2] * len(self.compress_tokens) + [3] * chunk_len)

            labels.extend([-100] * (chunk_len + 1 + len(self.compress_tokens)) + chunk)

            position_ids.extend(list(range(-chunk_len-1, len(self.compress_tokens) + chunk_len)))

            output_sequence.extend(processed_chunk)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_pretraining_instruct_compress(
        self,
        example
    ):
        instruction_tokens = self.tokenizer("Repeat the preceding sentence.").input_ids
        instruction_len = len(instruction_tokens)
        text_tokens = self.tokenizer(example['text']).input_ids
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []
        chunk_counter = 0
        # 1. Split text_tokens into slices of size `self.chunk_size`.
        for i in range(0, len(text_tokens), self.chunk_size):

            chunk_counter += 1
            chunk = text_tokens[i : i + self.chunk_size]

            chunk_len = len(chunk)  # could be < self.chunk_size for the last chunk

            if chunk_len < self.chunk_size:
                break  # no more tokens

            # 2. Build the processed chunk
            #    chunk + [chunk_end_token] + self.compress_tokens + chunk
            processed_chunk = chunk + [self.chunk_end_token] + self.compress_tokens + instruction_tokens + chunk

            # 3. Check if adding this processed chunk would exceed max_length
            if len(output_sequence) + len(processed_chunk) > self.max_len:
                # If we can't add this chunk without exceeding,
                # we stop processing further.
                break

            segment_ids_1.extend([chunk_counter] * len(processed_chunk))

            segment_ids_2.extend([1] * (chunk_len + 1) + [2] * len(self.compress_tokens) + [3] * (instruction_len + chunk_len))

            labels.extend([-100] * (chunk_len + 1 + len(self.compress_tokens) + instruction_len) + chunk)

            position_ids.extend(list(range(-chunk_len-1, len(self.compress_tokens) + instruction_len + chunk_len)))

            output_sequence.extend(processed_chunk)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_pretraining_singlechunk_completion_compress(
        self,
        example
    ):
        text_tokens = self.tokenizer(example['text']).input_ids
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []
        chunk_counter = 0
        # 1. Split text_tokens into slices of size `self.chunk_size`.
        for i in range(0, len(text_tokens), 2 * self.chunk_size):

            chunk_counter += 1
            chunk1 = text_tokens[i : i + self.chunk_size]
            chunk2 = text_tokens[i + self.chunk_size : i + 2 * self.chunk_size]
            chunk1_len = len(chunk1)
            chunk2_len = len(chunk2)  # could be < self.chunk_size for the last chunk

            if chunk2_len < self.chunk_size:
                break  # no more tokens

            # 2. Build the processed chunk
            #    chunk + [chunk_end_token] + self.compress_tokens + chunk
            processed_chunk = chunk1 + [self.chunk_end_token] + self.compress_tokens + chunk2

            # 3. Check if adding this processed chunk would exceed max_length
            if len(output_sequence) + len(processed_chunk) > self.max_len:
                # If we can't add this chunk without exceeding,
                # we stop processing further.
                break

            segment_ids_1.extend([chunk_counter] * len(processed_chunk))

            segment_ids_2.extend([1] * (chunk1_len + 1) + [2] * len(self.compress_tokens) + [3] * chunk2_len)

            labels.extend([-100] * (chunk1_len + 1 + len(self.compress_tokens)) + chunk2)

            position_ids.extend(list(range(-chunk1_len - 1, len(self.compress_tokens) + chunk2_len)))

            output_sequence.extend(processed_chunk)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_pretraining_multichunk_completion_compress(
        self,
        example
    ):
        text_tokens = self.tokenizer(example['text'], add_special_tokens=False).input_ids[:self.max_len]
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []
        chunk_num = random.randint(5,20)
        remaining_ids = text_tokens[chunk_num * self.chunk_size:]
        remaining_len = len(remaining_ids)

        bos_tokens = self.tokenizer("").input_ids
        output_sequence.extend(bos_tokens + [self.global_start_token])
        segment_ids_1.extend([0]*2)
        segment_ids_2.extend([3]*2)
        labels.extend([-100]*2)
        position_ids.extend([0,1])

        current_position = 2
        for i in range(chunk_num):
            chunk_counter = i+1
            chunk_ids = text_tokens[i * self.chunk_size : i * self.chunk_size + self.chunk_size] + [self.chunk_end_token]
            chunk_len = len(chunk_ids)

            segment_ids_1.extend([chunk_counter] * (chunk_len + len(self.compress_tokens)))

            segment_ids_2.extend([1] * chunk_len + [2] * len(self.compress_tokens))

            labels.extend([-100] * (chunk_len + len(self.compress_tokens)))

            position_ids.extend(list(range(current_position - chunk_len, current_position + len(self.compress_tokens))))
            current_position += len(self.compress_tokens)

            output_sequence.extend(chunk_ids + self.compress_tokens)

        output_sequence.extend([self.global_end_token] + remaining_ids)
        segment_ids_1.extend([0] * (1 + remaining_len))
        segment_ids_2.extend([3] * (1 + remaining_len))
        labels.extend([-100] + remaining_ids)
        position_ids.extend(list(range(current_position, current_position + 1 + remaining_len)))

        return {
            "input_ids": output_sequence[:self.max_len],
            "segment_ids_1": segment_ids_1[:self.max_len],
            "segment_ids_2": segment_ids_2[:self.max_len],
            "labels": labels[:self.max_len],
            "position_ids": position_ids[:self.max_len],
        }

    def process_pretraining_multichunk_batch(
        self,
        example
    ):
        all_input_ids = []
        all_segment_ids_1 = []
        all_segment_ids_2 = []
        all_labels = []
        all_position_ids = []

        for text in example['text']:
            text_tokens = self.tokenizer(text, add_special_tokens=False).input_ids[:self.max_len]
            output_sequence = []
            segment_ids_1 = []
            segment_ids_2 = []
            labels = []
            position_ids = []
            chunk_num = random.randint(5,20)
            remaining_ids = text_tokens[chunk_num * self.chunk_size:]
            remaining_len = len(remaining_ids)

            bos_tokens = self.tokenizer("").input_ids
            output_sequence.extend(bos_tokens + [self.global_start_token])
            segment_ids_1.extend([0]*2)
            segment_ids_2.extend([3]*2)
            labels.extend([-100]*2)
            position_ids.extend([0,1])

            current_position = 2
            for i in range(chunk_num):
                chunk_counter = i+1
                chunk_ids = text_tokens[i * self.chunk_size : i * self.chunk_size + self.chunk_size] + [self.chunk_end_token]
                chunk_len = len(chunk_ids)

                segment_ids_1.extend([chunk_counter] * (chunk_len + len(self.compress_tokens)))

                segment_ids_2.extend([1] * chunk_len + [2] * len(self.compress_tokens))

                labels.extend([-100] * (chunk_len + len(self.compress_tokens)))

                position_ids.extend(list(range(current_position - chunk_len, current_position + len(self.compress_tokens))))
                current_position += len(self.compress_tokens)

                output_sequence.extend(chunk_ids + self.compress_tokens)

            output_sequence.extend([self.global_end_token] + remaining_ids)
            segment_ids_1.extend([0] * (1 + remaining_len))
            segment_ids_2.extend([3] * (1 + remaining_len))
            labels.extend([-100] + remaining_ids)
            position_ids.extend(list(range(current_position, current_position + 1 + remaining_len)))

            all_input_ids.append(output_sequence)
            all_segment_ids_1.append(segment_ids_1)
            all_segment_ids_2.append(segment_ids_2)
            all_labels.append(labels)
            all_position_ids.append(position_ids)

        return {
            "input_ids": all_input_ids,
            "segment_ids_1": all_segment_ids_1,
            "segment_ids_2": all_segment_ids_2,
            "labels": all_labels,
            "position_ids": all_position_ids,
        }

    def process_pretraining_multichunk2_completion_compress(
        self,
        example
    ):
        text_tokens = self.tokenizer(example['text'], add_special_tokens=False).input_ids[:self.max_len]
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []
        total_chunks = math.floor(len(text_tokens) / self.chunk_size)

        chunk_num = random.randint(1, min(total_chunks - 1, 30))
        remaining_ids = text_tokens[chunk_num * self.chunk_size:]
        remaining_len = len(remaining_ids)

        bos_tokens = self.tokenizer("").input_ids
        output_sequence.extend(bos_tokens + [self.global_start_token])
        segment_ids_1.extend([0]*2)
        segment_ids_2.extend([3]*2)
        labels.extend([-100]*2)
        position_ids.extend([0,1])

        current_position = 2
        for i in range(chunk_num):
            chunk_counter = i+1
            chunk_ids = text_tokens[i * self.chunk_size : i * self.chunk_size + self.chunk_size] + [self.chunk_end_token]
            chunk_len = len(chunk_ids)

            segment_ids_1.extend([chunk_counter] * (chunk_len + len(self.compress_tokens)))

            segment_ids_2.extend([1] * chunk_len + [2] * len(self.compress_tokens))

            labels.extend([-100] * (chunk_len + len(self.compress_tokens)))

            position_ids.extend(list(range(current_position - chunk_len, current_position + len(self.compress_tokens))))
            current_position += len(self.compress_tokens)

            output_sequence.extend(chunk_ids + self.compress_tokens)

        output_sequence.extend([self.global_end_token] + remaining_ids)
        segment_ids_1.extend([0] * (1 + remaining_len))
        segment_ids_2.extend([3] * (1 + remaining_len))
        labels.extend([-100] + remaining_ids)
        position_ids.extend(list(range(current_position, current_position + 1 + remaining_len)))

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_pretraining_multichunk2_batch(
        self,
        example
    ):
        all_input_ids = []
        all_segment_ids_1 = []
        all_segment_ids_2 = []
        all_labels = []
        all_position_ids = []

        for text in example['text']:
            text_tokens = self.tokenizer(text, add_special_tokens=False).input_ids[:self.max_len]
            output_sequence = []
            segment_ids_1 = []
            segment_ids_2 = []
            labels = []
            position_ids = []
            total_chunks = math.floor(len(text_tokens) / self.chunk_size)

            chunk_num = random.randint(1, min(total_chunks - 1, 30))
            remaining_ids = text_tokens[chunk_num * self.chunk_size:]
            remaining_len = len(remaining_ids)

            bos_tokens = self.tokenizer("").input_ids
            output_sequence.extend(bos_tokens + [self.global_start_token])
            segment_ids_1.extend([0]*2)
            segment_ids_2.extend([3]*2)
            labels.extend([-100]*2)
            position_ids.extend([0,1])

            current_position = 2
            for i in range(chunk_num):
                chunk_counter = i+1
                chunk_ids = text_tokens[i * self.chunk_size : i * self.chunk_size + self.chunk_size] + [self.chunk_end_token]
                chunk_len = len(chunk_ids)

                segment_ids_1.extend([chunk_counter] * (chunk_len + len(self.compress_tokens)))

                segment_ids_2.extend([1] * chunk_len + [2] * len(self.compress_tokens))

                labels.extend([-100] * (chunk_len + len(self.compress_tokens)))

                position_ids.extend(list(range(current_position - chunk_len, current_position + len(self.compress_tokens))))
                current_position += len(self.compress_tokens)

                output_sequence.extend(chunk_ids + self.compress_tokens)

            output_sequence.extend([self.global_end_token] + remaining_ids)
            segment_ids_1.extend([0] * (1 + remaining_len))
            segment_ids_2.extend([3] * (1 + remaining_len))
            labels.extend([-100] + remaining_ids)
            position_ids.extend(list(range(current_position, current_position + 1 + remaining_len)))

            all_input_ids.append(output_sequence)
            all_segment_ids_1.append(segment_ids_1)
            all_segment_ids_2.append(segment_ids_2)
            all_labels.append(labels)
            all_position_ids.append(position_ids)

        return {
            "input_ids": all_input_ids,
            "segment_ids_1": all_segment_ids_1,
            "segment_ids_2": all_segment_ids_2,
            "labels": all_labels,
            "position_ids": all_position_ids,
        }

    def process_qa_compress(
        self,
        example
    ):
        
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids + [self.global_start_token]
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([3] * sys_len)
        labels.extend([-100] * sys_len)
        position_ids.extend(list(range(sys_len)))

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        current_index = sys_len
        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids + [self.chunk_end_token]

            segment_ids_1.extend([j+1] * (len(tem_id) + len(self.compress_tokens)))
            segment_ids_2.extend([1] * len(tem_id) + [2] * len(self.compress_tokens))
            labels.extend([-100] * (len(tem_id) + len(self.compress_tokens)))
            position_ids.extend(list(range(current_index - len(tem_id), current_index + len(self.compress_tokens))))
            output_sequence.extend(tem_id + self.compress_tokens)

            current_index += len(self.compress_tokens)

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = [self.global_end_token] + self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([3] * user_len)
        labels.extend([-100] * user_len)
        position_ids.extend(list(range(current_index, current_index + user_len)))
        output_sequence.extend(user_id)
        current_index += user_len

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        ans_len = len(ans_id)
        segment_ids_1.extend([0] * ans_len)
        segment_ids_2.extend([3] * ans_len)
        labels.extend(ans_id)
        position_ids.extend(list(range(current_index, current_index + ans_len)))
        output_sequence.extend(ans_id)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_qa_chunk_compress(
        self,
        example
    ):
        
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids + [self.global_start_token]
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([3] * sys_len)
        labels.extend([-100] * sys_len)
        position_ids.extend(list(range(sys_len)))

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        current_index = sys_len

        chunk_idx = 1
        for j in range(0,10):

            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids + [self.chunk_end_token]

            for idx in range(0, len(tem_id), self.chunk_size):
                chunk_id = tem_id[idx : idx + self.chunk_size]
                if len(chunk_id) < self.chunk_size:
                    chunk_id.extend([self.pad_token] * (self.chunk_size - len(chunk_id)))
                
                segment_ids_1.extend([chunk_idx] * (self.chunk_size + 1 + len(self.compress_tokens)))
                segment_ids_2.extend([1] * (self.chunk_size + 1) + [2] * len(self.compress_tokens))
                labels.extend([-100] * (self.chunk_size + 1 + len(self.compress_tokens)))
                position_ids.extend(list(range(current_index - self.chunk_size - 1, current_index + len(self.compress_tokens))))
                output_sequence.extend(chunk_id + [self.chunk_end_token] + self.compress_tokens)

                current_index += len(self.compress_tokens)
                chunk_idx += 1

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = [self.global_end_token] + self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([3] * user_len)
        labels.extend([-100] * user_len)
        position_ids.extend(list(range(current_index, current_index + user_len)))
        output_sequence.extend(user_id)
        current_index += user_len

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        ans_len = len(ans_id)
        segment_ids_1.extend([0] * ans_len)
        segment_ids_2.extend([3] * ans_len)
        labels.extend(ans_id)
        position_ids.extend(list(range(current_index, current_index + ans_len)))
        output_sequence.extend(ans_id)

        # import ipdb
        # ipdb.set_trace()

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_qa_chunk_nopadding_compress(
        self,
        example
    ):
        
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids + [self.global_start_token]
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([3] * sys_len)
        labels.extend([-100] * sys_len)
        position_ids.extend(list(range(sys_len)))

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        current_index = sys_len

        chunk_idx = 1
        for j in range(0,10):

            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids + [self.chunk_end_token]

            for idx in range(0, len(tem_id), self.chunk_size):
                chunk_id = tem_id[idx : idx + self.chunk_size]
                chunk_len = len(chunk_id)
                
                segment_ids_1.extend([chunk_idx] * (chunk_len + 1 + len(self.compress_tokens)))
                segment_ids_2.extend([1] * (chunk_len + 1) + [2] * len(self.compress_tokens))
                labels.extend([-100] * (chunk_len + 1 + len(self.compress_tokens)))
                position_ids.extend(list(range(current_index - chunk_len - 1, current_index + len(self.compress_tokens))))
                output_sequence.extend(chunk_id + [self.chunk_end_token] + self.compress_tokens)

                current_index += len(self.compress_tokens)
                chunk_idx += 1

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = [self.global_end_token] + self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([3] * user_len)
        labels.extend([-100] * user_len)
        position_ids.extend(list(range(current_index, current_index + user_len)))
        output_sequence.extend(user_id)
        current_index += user_len

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        ans_len = len(ans_id)
        segment_ids_1.extend([0] * ans_len)
        segment_ids_2.extend([3] * ans_len)
        labels.extend(ans_id)
        position_ids.extend(list(range(current_index, current_index + ans_len)))
        output_sequence.extend(ans_id)

        # import ipdb
        # ipdb.set_trace()

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_qa_chunk_nopadding_kvlink(
        self,
        example
    ):

        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids + [self.global_start_token]
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([3] * sys_len)
        labels.extend([-100] * sys_len)
        position_ids.extend(list(range(sys_len)))

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        current_index = sys_len

        chunk_idx = 1
        for j in range(0,10):

            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            for idx in range(0, len(tem_id), self.chunk_size):
                chunk_id = tem_id[idx : idx + self.chunk_size]
                chunk_len = len(chunk_id)
                
                segment_ids_1.extend([chunk_idx] * (chunk_len + 1 + len(self.compress_tokens)) + [0] * self.link_token_num)
                segment_ids_2.extend([1] * (chunk_len + 1) + [2] * len(self.compress_tokens) + [3] * self.link_token_num)
                labels.extend([-100] * (chunk_len + 1 + len(self.compress_tokens) + self.link_token_num))
                position_ids.extend(list(range(current_index - chunk_len - 1, current_index + len(self.compress_tokens) + self.link_token_num)))
                output_sequence.extend(chunk_id + [self.chunk_end_token] + self.compress_tokens + self.link_tokens[chunk_idx-1])

                current_index += len(self.compress_tokens) + self.link_token_num
                chunk_idx += 1

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = [self.global_end_token] + self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([3] * user_len)
        labels.extend([-100] * user_len)
        position_ids.extend(list(range(current_index, current_index + user_len)))
        output_sequence.extend(user_id)
        current_index += user_len

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        ans_len = len(ans_id)
        segment_ids_1.extend([0] * ans_len)
        segment_ids_2.extend([3] * ans_len)
        labels.extend(ans_id)
        position_ids.extend(list(range(current_index, current_index + ans_len)))
        output_sequence.extend(ans_id)

        # import ipdb
        # ipdb.set_trace()

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

def custom_collate_compress(batch):

        input_ids = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []
        length_list = [len(x['input_ids']) for x in batch]

        max_length = max(length_list)
        for item in batch:

            seq_length = len(item['input_ids'])
            residual = max_length - seq_length

            padded_input_ids = item['input_ids'] + [0] * residual
            input_ids.append(padded_input_ids)

            padded_segment_ids_1 = item['segment_ids_1'] + [-1] * residual
            segment_ids_1.append(padded_segment_ids_1)

            padded_segment_ids_2 = item['segment_ids_2'] + [-1] * residual
            segment_ids_2.append(padded_segment_ids_2)

            padded_labels = item['labels'] + [-100] * residual
            labels.append(padded_labels)

            padded_position_ids = item['position_ids'] + [0] * residual
            position_ids.append(padded_position_ids)
        print(max_length)
        return {
            "input_ids": torch.LongTensor(input_ids),
            "segment_ids_1": torch.LongTensor(segment_ids_1),
            "segment_ids_2": torch.LongTensor(segment_ids_2),
            "labels": torch.LongTensor(labels),
            "position_ids": torch.LongTensor(position_ids),
        }

class upper_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        do_shuffle: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.do_shuffle = do_shuffle

    def process_pretraining(
        self,
        example
    ):
        text_tokens = self.tokenizer(example['text']).input_ids[:self.max_len]
        output_sequence = text_tokens
        labels = text_tokens

        return {
            "input_ids": output_sequence,
            "labels": labels,
        }

    def process_qa(
        self,
        example
    ):
        
        output_sequence = []
        labels = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        labels.extend([-100] * sys_len)

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
            output_sequence.extend(tem_id)
            labels.extend([-100] * (len(tem_id)))

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        labels.extend([-100] * user_len)
        output_sequence.extend(user_id)

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        labels.extend(ans_id)
        output_sequence.extend(ans_id)

        return {
            "input_ids": output_sequence,
            "labels": labels,
        }

def custom_collate_upper(batch):

        input_ids = []
        labels = []
        length_list = [len(x['input_ids']) for x in batch]

        max_length = max(length_list)

        for item in batch:

            seq_length = len(item['input_ids'])
            residual = max_length - seq_length

            padded_input_ids = item['input_ids'] + [0] * residual
            input_ids.append(padded_input_ids)

            padded_labels = item['labels'] + [-100] * residual
            labels.append(padded_labels)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels)
        }

class compress_ratio_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        compress_tokens: list[int],
        compress_ratio: float,
        chunk_end_token: int,
        do_shuffle: bool,
        global_start_token: int = 128254,
        global_end_token: int = 128255,
        link_token_num: int = 1,
        max_chunk_num: int = 20,
        max_total_chunk_length: int = 2000,
        # single_chunk_size: tuple[int] = (20, 400)
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.compress_tokens = compress_tokens
        self.compress_ratio = compress_ratio
        self.chunk_end_token = chunk_end_token
        self.do_shuffle = do_shuffle
        self.global_start_token = global_start_token
        self.global_end_token = global_end_token
        self.link_token_num = link_token_num

        link_token_start = self.compress_tokens[-1] + 1
        self.link_tokens = [
            [
                link_token_start + idx * self.link_token_num + offset
                for offset in range(self.link_token_num)
            ]
            for idx in range(max_chunk_num)
        ]

        self.max_chunk_num = max_chunk_num
        self.max_total_chunk_length = max_total_chunk_length
        self.single_chunk_size = (20, int(len(compress_tokens) / self.compress_ratio))

    def round_up_to_10(self, num):
        return math.ceil(num / 10.0) * 10

    def process_pretraining_singlechunk_completion_compress(
        self,
        example
    ):
        text_tokens = self.tokenizer(example['text']).input_ids
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []
        chunk_counter = 0

        chunk_len = int(random.randint(self.single_chunk_size[0], self.single_chunk_size[1]) / 2)
        # 1. Split text_tokens into slices of size `chunk_len`.
        for i in range(0, len(text_tokens), 2 * chunk_len):

            chunk_counter += 1
            chunk1 = text_tokens[i : i + chunk_len]
            chunk2 = text_tokens[i + chunk_len : i + 2 * chunk_len]
            chunk1_len = len(chunk1)
            chunk2_len = len(chunk2)  # could be < self.chunk_size for the last chunk

            if chunk2_len < chunk_len:
                break  # no more tokens

            chunk_compress_token_len = self.round_up_to_10((chunk_len) * self.compress_ratio)
            chunk_compress_tokens = self.compress_tokens[:chunk_compress_token_len]

            # 2. Build the processed chunk
            processed_chunk = chunk1 + [self.chunk_end_token] + chunk_compress_tokens + chunk2

            # 3. Check if adding this processed chunk would exceed max_length
            if len(output_sequence) + len(processed_chunk) > self.max_len:
                # If we can't add this chunk without exceeding,
                # we stop processing further.
                break

            segment_ids_1.extend([chunk_counter] * len(processed_chunk))

            segment_ids_2.extend([1] * (chunk1_len + 1) + [2] * chunk_compress_token_len + [3] * chunk2_len)

            labels.extend([-100] * (chunk1_len + 1 + chunk_compress_token_len) + chunk2)

            position_ids.extend(list(range(-chunk1_len - 1, chunk_compress_token_len + chunk2_len)))

            output_sequence.extend(processed_chunk)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_pretraining_multichunk_completion_compress(
        self,
        example
    ):
        # data used for this processing should contain at least self.max_total_chunk_length tokens
        text_tokens = self.tokenizer(example['text'], add_special_tokens=False).input_ids[:self.max_len]

        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        bos_tokens = self.tokenizer("").input_ids
        output_sequence.extend(bos_tokens + [self.global_start_token])
        segment_ids_1.extend([0]*2)
        segment_ids_2.extend([3]*2)
        labels.extend([-100]*2)
        position_ids.extend([0,1])

        current_position = 2
        chunk_counter = 0
        idx = 0
        while True:
            chunk_counter += 1
            # Condition1: If the number of chunks reach the maximum, jump out of chunking.
            if chunk_counter > self.max_chunk_num:
                break

            chunk_len = random.randint(self.single_chunk_size[0], self.single_chunk_size[1])

            # Condition2: If total chunk length reach the maximum, jump out of chunking.
            if idx + chunk_len > self.max_total_chunk_length:
                break

            chunk_ids = text_tokens[idx : idx+chunk_len]

            chunk_compress_token_len = self.round_up_to_10((chunk_len) * self.compress_ratio)
            chunk_compress_tokens = self.compress_tokens[:chunk_compress_token_len]

            # if len(chunk_compress_tokens) != chunk_compress_token_len:
            #     import ipdb
            #     ipdb.set_trace()

            segment_ids_1.extend([chunk_counter] * (chunk_len + 1 + chunk_compress_token_len))

            segment_ids_2.extend([1] * (chunk_len + 1) + [2] * chunk_compress_token_len)

            labels.extend([-100] * (chunk_len + 1 + chunk_compress_token_len))

            position_ids.extend(list(range(current_position - chunk_len - 1, current_position + chunk_compress_token_len)))

            output_sequence.extend(chunk_ids + [self.chunk_end_token] + chunk_compress_tokens)

            current_position += chunk_compress_token_len
            idx += chunk_len

        remaining_ids = text_tokens[idx:]
        remaining_len = len(remaining_ids)

        output_sequence.extend([self.global_end_token] + remaining_ids)
        segment_ids_1.extend([0] * (1 + remaining_len))
        segment_ids_2.extend([3] * (1 + remaining_len))
        labels.extend([-100] + remaining_ids)
        position_ids.extend(list(range(current_position, current_position + 1 + remaining_len)))

        # if len(output_sequence) != len(segment_ids_1):
        #     import ipdb
        #     ipdb.set_trace()

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_qa_compress(
        self,
        example
    ):
        
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids + [self.global_start_token]
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([3] * sys_len)
        labels.extend([-100] * sys_len)
        position_ids.extend(list(range(sys_len)))

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        current_index = sys_len
        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids + [self.chunk_end_token]

            chunk_compress_token_len = self.round_up_to_10(len(tem_id) * self.compress_ratio)
            if chunk_compress_token_len > len(self.compress_tokens):
                chunk_compress_token_len = len(self.compress_tokens)
            chunk_compress_tokens = self.compress_tokens[:chunk_compress_token_len]

            segment_ids_1.extend([j+1] * (len(tem_id) + chunk_compress_token_len))
            segment_ids_2.extend([1] * len(tem_id) + [2] * chunk_compress_token_len)
            labels.extend([-100] * (len(tem_id) + chunk_compress_token_len))
            position_ids.extend(list(range(current_index - len(tem_id), current_index + chunk_compress_token_len)))
            output_sequence.extend(tem_id + chunk_compress_tokens)

            current_index += chunk_compress_token_len

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = [self.global_end_token] + self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([3] * user_len)
        labels.extend([-100] * user_len)
        position_ids.extend(list(range(current_index, current_index + user_len)))
        output_sequence.extend(user_id)
        current_index += user_len

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        ans_len = len(ans_id)
        segment_ids_1.extend([0] * ans_len)
        segment_ids_2.extend([3] * ans_len)
        labels.extend(ans_id)
        position_ids.extend(list(range(current_index, current_index + ans_len)))
        output_sequence.extend(ans_id)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }

    def process_qa_compress_kvlink(
        self,
        example
    ):
        
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids + [self.global_start_token]
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([3] * sys_len)
        labels.extend([-100] * sys_len)
        position_ids.extend(list(range(sys_len)))

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        current_index = sys_len
        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids + [self.chunk_end_token]

            chunk_compress_token_len = self.round_up_to_10(len(tem_id) * self.compress_ratio)
            if chunk_compress_token_len > len(self.compress_tokens):
                chunk_compress_token_len = len(self.compress_tokens)
            chunk_compress_tokens = self.compress_tokens[:chunk_compress_token_len]

            segment_ids_1.extend([j+1] * (len(tem_id) + chunk_compress_token_len) + [0] * self.link_token_num)
            segment_ids_2.extend([1] * len(tem_id) + [2] * chunk_compress_token_len + [3] * self.link_token_num)
            labels.extend([-100] * (len(tem_id) + chunk_compress_token_len + self.link_token_num))
            position_ids.extend(list(range(current_index - len(tem_id), current_index + chunk_compress_token_len + self.link_token_num)))
            output_sequence.extend(tem_id + chunk_compress_tokens + self.link_tokens[j])

            current_index += chunk_compress_token_len + self.link_token_num

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = [self.global_end_token] + self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([3] * user_len)
        labels.extend([-100] * user_len)
        position_ids.extend(list(range(current_index, current_index + user_len)))
        output_sequence.extend(user_id)
        current_index += user_len

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        ans_len = len(ans_id)
        segment_ids_1.extend([0] * ans_len)
        segment_ids_2.extend([3] * ans_len)
        labels.extend(ans_id)
        position_ids.extend(list(range(current_index, current_index + ans_len)))
        output_sequence.extend(ans_id)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }
    
class kvlink_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        do_shuffle: bool,
        global_start_token: int = 128254,
        global_end_token: int = 128255,
        link_token_num: int = 1,
        max_chunk_num: int = 20,
        max_total_chunk_length: int = 2000,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.do_shuffle = do_shuffle
        self.global_start_token = global_start_token
        self.global_end_token = global_end_token
        self.link_token_num = link_token_num
        self.max_chunk_num = max_chunk_num
        self.max_total_chunk_length = max_total_chunk_length
        self.single_chunk_size = (20,400)
        link_token_start = 128011
        self.link_tokens = [
            [
                link_token_start + idx * self.link_token_num + offset
                for offset in range(self.link_token_num)
            ]
            for idx in range(max_chunk_num)
        ]

        
    def process_qa_kvlink(
        self,
        example
    ):
        
        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids + [self.global_start_token]
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([3] * sys_len)
        labels.extend([-100] * sys_len)
        position_ids.extend(list(range(sys_len)))

        doc_list = []

        for k in range(0,10):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text':text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        current_index = sys_len
        for j in range(0,10):
            title = doc_list[j]['title']
            text = doc_list[j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            segment_ids_1.extend([j+1] * len(tem_id) + [0] * self.link_token_num)
            segment_ids_2.extend([3] * (len(tem_id) + self.link_token_num))
            labels.extend([-100] * (len(tem_id) + self.link_token_num))
            position_ids.extend(list(range(current_index, current_index + len(tem_id) + self.link_token_num)))
            output_sequence.extend(tem_id + self.link_tokens[j])

            current_index += len(tem_id) + self.link_token_num

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = [self.global_end_token] + self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([3] * user_len)
        labels.extend([-100] * user_len)
        position_ids.extend(list(range(current_index, current_index + user_len)))
        output_sequence.extend(user_id)
        current_index += user_len

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
        ans_len = len(ans_id)
        segment_ids_1.extend([0] * ans_len)
        segment_ids_2.extend([3] * ans_len)
        labels.extend(ans_id)
        position_ids.extend(list(range(current_index, current_index + ans_len)))
        output_sequence.extend(ans_id)

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }
    
    def process_pretraining_kvlink(
        self,
        example
    ):
        # data used for this processing should contain at least self.max_total_chunk_length tokens
        text_tokens = self.tokenizer(example['text'], add_special_tokens=False).input_ids[:self.max_len]

        output_sequence = []
        segment_ids_1 = []
        segment_ids_2 = []
        labels = []
        position_ids = []

        bos_tokens = self.tokenizer("").input_ids
        output_sequence.extend(bos_tokens + [self.global_start_token])
        segment_ids_1.extend([0]*2)
        segment_ids_2.extend([3]*2)
        labels.extend([-100]*2)
        position_ids.extend([0,1])

        current_position = 2
        chunk_counter = 0
        idx = 0
        while True:
            chunk_counter += 1
            # Condition1: If the number of chunks reach the maximum, jump out of chunking.
            if chunk_counter > self.max_chunk_num:
                break

            chunk_len = random.randint(self.single_chunk_size[0], self.single_chunk_size[1])

            # Condition2: If total chunk length reach the maximum, jump out of chunking.
            if idx + chunk_len > self.max_total_chunk_length:
                break

            chunk_ids = text_tokens[idx : idx+chunk_len]

            # if len(chunk_compress_tokens) != chunk_compress_token_len:
            #     import ipdb
            #     ipdb.set_trace()

            segment_ids_1.extend([chunk_counter] * (chunk_len) + [0] * self.link_token_num)

            segment_ids_2.extend([3] * (chunk_len + self.link_token_num))

            labels.extend([-100] * (chunk_len + self.link_token_num))

            position_ids.extend(list(range(current_position, current_position + chunk_len + self.link_token_num)))

            output_sequence.extend(chunk_ids + self.link_tokens[chunk_counter-1])

            current_position += chunk_len + self.link_token_num
            idx += chunk_len

        remaining_ids = text_tokens[idx:]
        remaining_len = len(remaining_ids)

        output_sequence.extend([self.global_end_token] + remaining_ids)
        segment_ids_1.extend([0] * (1 + remaining_len))
        segment_ids_2.extend([3] * (1 + remaining_len))
        labels.extend([-100] + remaining_ids)
        position_ids.extend(list(range(current_position, current_position + 1 + remaining_len)))

        # if len(output_sequence) != len(segment_ids_1):
        #     import ipdb
        #     ipdb.set_trace()

        return {
            "input_ids": output_sequence,
            "segment_ids_1": segment_ids_1,
            "segment_ids_2": segment_ids_2,
            "labels": labels,
            "position_ids": position_ids,
        }