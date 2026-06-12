# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn conversation into topically coherent units based on semantics. The conversation may be a dialogue between two participants or a meeting among several speakers. Successive turns on the same topic should be grouped into the same segmentation unit, and a new segmentation unit should be created when a topic shift occurs.

- **Data**: The input data is a series of turns separated by "\n\n". Each turn is a single utterance by one speaker, started with "[Turn (Turn Number)]: (Speaker) ".

## Requirements

### Output Format

- Output the segmentation results in **jsonl lines file** format. Each dictionary represents a segment, consisting of one or more turns on the same topic. Each dictionary should include the following keys:
    - **segment_id**: The index of this segment, starting from 0.
    - **start_turn_number**: The number of the **first** turn in this segment.
    - **end_turn_number**: The number of the **last** turn in this segment.
    - **num_turns**: An integer indicating the number of turns in this segment, calculated as **end_turn_number** - **start_turn_number** + 1.
    - **summary**: A brief summary (within 100 words) of this segment. The summary should be straightforward, without unnecessary prefixes such as "The conversation starts with", "The discussion shifts back to", or "The topic shifts to".

Here is an example of the expected output:
```
<segmentation>
{{"segment_id": 0, "start_turn_number": 0, "end_turn_number": 5, "num_turns": 6, "summary": "A brief summary of this segment."}}
{{"segment_id": 1, "start_turn_number": 6, "end_turn_number": 8, "num_turns": 3, "summary": "A brief summary of this segment."}}
...
</segmentation>
```

# Data

{text_to_be_segmented}

# Question

## Please generate the segmentation result from the input data that meets the following requirements:

- **No Missing Turns**: Ensure that the turn numbers cover all turns in the given conversation without omission. The **start_turn_number** of the next segment should be equal to **end_turn_number** + 1 of the previous segment. The **start_turn_number** of the first segment should be 0, and the **end_turn_number** of the last segment should equal the last turn number of the given input.
- **No Overlapping Turns**: Ensure that successive segments have no overlap in turns. The **start_turn_number** of the next segment should be equal to **end_turn_number** + 1 of the previous segment.
- **Concise, Complete and Straightforward Summaries**: The summaries should be concise, not exceeding 100 words, while retaining all key information from the segment. The summaries should be straightforward, without unnecessary prefixes such as "The conversation starts with ", "The discussion shifts back to " or "The topic shifts to ...".
- **Accurate Counting**: The sum of **num_turns** across all segments should equal the total number of turns in the input.
- Provide your segmentation result between the tags: <segmentation></segmentation>.

# Output

Now, provide the segmentation result based on the instructions above.
