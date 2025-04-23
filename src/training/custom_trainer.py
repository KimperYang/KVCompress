import torch
from transformers import Trainer

from src.data.attention import make_segment_mask_with_two_rules, make_anchor_attention

class CustomTrainerCompressAttn(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        mask = make_segment_mask_with_two_rules(
            source_segments_1=inputs['segment_ids_1'],
            target_segments_1=inputs['segment_ids_1'],
            source_segments_2=inputs['segment_ids_2'],
            target_segments_2=inputs['segment_ids_2'],
            dtype=torch.bfloat16,           # For printing, let's keep float
            add_causal_lm_mask=True     # No causal mask for clarity
        )
        outputs = model(input_ids = inputs['input_ids'], attention_mask = mask.unsqueeze(1), labels = inputs['labels'], position_ids = inputs['position_ids'])
        torch.cuda.empty_cache()
        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
class AnchorTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        mask = make_anchor_attention(
            source_segments_1=inputs['segment_ids_1'],
            target_segments_1=inputs['segment_ids_1'],
            source_segments_2=inputs['segment_ids_2'],
            target_segments_2=inputs['segment_ids_2'],
            dtype=torch.bfloat16,           # For printing, let's keep float
            allow_self=True     # No causal mask for clarity
        )
        outputs = model(input_ids = inputs['input_ids'], attention_mask = mask.unsqueeze(0).unsqueeze(0), labels = inputs['labels'])
        torch.cuda.empty_cache()
        return (outputs.loss, outputs) if return_outputs else outputs.loss
