# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn conversation into topically coherent units based on semantics. The conversation may be a dialogue between two participants or a meeting among several speakers. Successive turns on the same topic should be grouped into the same segmentation unit, and a new segmentation unit should be created when a topic shift occurs.

- **Data**: The input data is a series of turns separated by "\n\n". Each turn is a single utterance by one speaker, started with "[Turn (Turn Number)]: (Speaker) ".

# Data

{text_to_be_segmented}

# Question

You are shown the conversation up to and including the CURRENT turn, **[Turn {cur}]** (the last turn above). Using the same segmentation criterion as above — a new segmentation unit is created when a topic shift occurs, while successive turns on the same topic stay in the same unit — decide whether **[Turn {cur}]** begins a NEW segment (a topic shift occurs at this turn) or continues the SAME segment as the preceding turns.

Answer with exactly one word: NEW or SAME.

# Output
