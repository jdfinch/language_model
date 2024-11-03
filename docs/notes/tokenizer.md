


find the first slot unfilled-for-generation and record min_out num expected generated tokens

truncate segments after segment with slot unfilled-for-generation

sort segments by trunc_rank, trunc_side, and truncability (filter by truncability)

truncate segments based on max_sequences

tokenize all value sequences in remaining segments

compute min and max tokens of each slot

sort slots of all segments by trunc_rank, trunc_side, and truncability (filter by truncability and min!=max)

sort-merge segments-by-trunc-rank and slots-by-trunc-rank

current_length <- expected_out_tokens + sum(segment_template_lens) + sum(slot_max)

for item in sorted_trunc_ranking

    if current_length <= max_length

        tokens_to_recover = max_length - current_length

        for slot in reverse(back_through(slots_of(sorted_trunc_ranking)))

            recovered_tokens = original_slot_value_max - current_slot_value_max

            add recovered tokens back to slot in counts table (account for trunc_text)

            tokens_to_recover -= recovered_tokens

            if tokens_to_recover == 0

                break

        break

    if item is a segment

        remove the segment (and its slots from count tables)

    if item is a slot
        
        truncate the slot content (update count tables), make sure trunc_text is accounted for

actually truncate value segments by corresponding slot value max

put together the full final sequence


