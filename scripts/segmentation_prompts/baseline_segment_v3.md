# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn conversation into topically coherent units based on semantics, in a **streaming** fashion. The conversation may be a one-on-one between a user and an AI assistant, or a multi-party meeting among several human speakers. You are given the **recent turns** (roughly the last few minutes, already processed — provided as context only) and the **new turns** to segment now. Successive turns on the same topic should be grouped into the same segmentation unit, and a new segmentation unit should be created when a topic shift occurs.

- **Data**: Each turn is a single utterance by one speaker, started with "[Turn (Turn Number)]: (Speaker) ", carrying a **global** turn index. The first new turn may either **continue** the latest recent topic or **start** a new topic.

## Requirements

### Output Format

- Output the segmentation results in **jsonl lines file** format. Each dictionary represents a segment of the **new turns**, consisting of one or more turns on the same topic. Each dictionary should include the following keys:
    - **segment_id**: The index of this segment within the new turns, starting from 0.
    - **start_turn_number**: The **global** turn index of the **first** turn in this segment.
    - **end_turn_number**: The **global** turn index of the **last** turn in this segment.
    - **num_turns**: An integer = **end_turn_number** - **start_turn_number** + 1.
    - **continues_previous** (first segment only): `true` if the first new turn continues the latest recent topic, `false` if it starts a new topic. Omit for later segments.

Here is an example of the expected output:
```
<segmentation>
{{"segment_id": 0, "start_turn_number": 412, "end_turn_number": 417, "num_turns": 6, "continues_previous": false}}
{{"segment_id": 1, "start_turn_number": 418, "end_turn_number": 420, "num_turns": 3}}
...
</segmentation>
```

# Recent context (≈ last few minutes, already segmented — for reference only, do NOT re-segment)

{recent_context}

# Data (new turns to segment)

{text_to_be_segmented}

# Question

## Please generate the segmentation result for the **new turns** that meets the following requirements:

- **No Missing Turns**: The segments must cover all new turns without omission. The **start_turn_number** of the first segment equals the first new turn's index; the **end_turn_number** of the last segment equals the last new turn's index.
- **No Overlapping Turns**: Successive segments have no overlap. The **start_turn_number** of the next segment = **end_turn_number** + 1 of the previous.
- **Genuine Shifts Only**: Create a new segment only at a genuine topic shift; do not over-segment turns that elaborate, react to, or continue the same topic. Judge by **semantic topic only**, not by speaker change or short acknowledgements ("yeah", "okay", "right").
- **Accurate Counting**: The sum of **num_turns** across all segments equals the total number of new turns.
- Provide your segmentation result between the tags: <segmentation></segmentation>.

# Output

Now, provide the segmentation result based on the instructions above.
