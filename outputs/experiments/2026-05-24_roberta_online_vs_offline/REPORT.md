# RoBERTa online vs offline — extended smoke

setup: ckpt = `/home/namchailin/Hi-OnTop/methods/RoBERTa/_roberta_unzip/roberta_seg_out/roberta_supervised/model`, device cpu, per-bench limit = 50 dial (first-N deterministic, sliding_window={args.sliding_window}, max_utt_len={args.max_utt_len}).

| 벤치 | n dial | n turn | Pk_off | F1_off | Score_off | Pk_on | F1_on | Score_on | Δ Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 50 | 782 | 0.4261 | 0.4415 | 0.5041 | 0.3943 | 0.4155 | 0.5056 | +0.0015 |
| dialseg711 | 50 | 1460 | 0.2795 | 0.6341 | 0.6578 | 0.2851 | 0.6248 | 0.6564 | -0.0014 |
| superseg | 50 | 620 | 0.1944 | 0.7965 | 0.7987 | 0.2567 | 0.7304 | 0.7346 | -0.0641 |

메모: limit=50 의 subset. 전수 GPU 평가는 별도 (현 머신 CUDA too-old).
