# Instruction

## Context

- **Goal**: You are performing **online** topic segmentation of a multi-turn conversation, processing it one turn at a time with no look-ahead. The conversation may be a dialogue between two participants or a meeting among several speakers. You are given the **current open segment** (the recent turns that are all on one topic) and **one new turn** that immediately follows it. Decide whether the new turn continues the same topic as the current segment, or whether it begins a new topic.

- **Data**: Each turn is a single utterance by one speaker, formatted as "[Turn (Turn Number)]: (Speaker) utterance". The current segment is a series of such turns separated by "\n\n"; the new turn is a single utterance in the same format.

## Requirements

- Judge by **semantic topic only** — not by speaker change, surface wording, or politeness. A different speaker, a short acknowledgement ("yeah", "okay", "right"), a clarifying question, a disagreement, or a follow-up about the same matter all stay on the **SAME** topic.
- Answer **No** only when the new turn **clearly shifts** to a different subject, agenda item, or task than what the current segment is about.
- When the new turn is ambiguous, or merely continues, elaborates, or reacts to the ongoing discussion, answer **Yes**.

# Current segment

{prev_session}

# New turn

{new_turn}

# Question

Does the new turn belong to the **SAME** topic as the current segment?
Answer with exactly one word — "Yes" or "No". Do not output anything else.

Answer:
