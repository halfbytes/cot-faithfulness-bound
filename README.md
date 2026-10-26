# cot-faithfulness-bound

Measures information flow between a language model's residual stream and its chain-of-thought trace.

## Install

Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
```

## Run

```bash
export HF_TOKEN=...                   # GPQA-Diamond, Meta-Llama-3-8B and Gemma-2-9B are gated on the Hugging Face Hub
python main.py make-cryptobench
python main.py prepare-data --benchmarks gsm8k math_algebra arc_challenge strategyqa boardgameqa gpqa_diamond

CFG="--config configs/main.yaml"
python main.py generate             $CFG --models llama3_8b mistral_7b gemma2_9b
python main.py measure-gap          $CFG
python main.py measure-gap          $CFG --models mistral_7b gemma2_9b --benchmarks gpqa_diamond cryptobench
python main.py measure-gap          $CFG --benchmarks cryptobench --caps 128 256 1024 2048
python main.py measure-gap          $CFG --benchmarks cryptobench --n-pairs
python main.py perturbation         $CFG --models llama3_8b mistral_7b gemma2_9b
python main.py perturbation         $CFG --benchmarks cryptobench --caps 128 256 1024 2048
python main.py patching             $CFG
python main.py estimator-comparison $CFG
python main.py calibration          $CFG
python main.py complexity           $CFG
python main.py ib-frontier          $CFG
python main.py tables               $CFG --models llama3_8b mistral_7b gemma2_9b
python main.py measure-gap          --config configs/zero_shot.yaml
python main.py measure-gap          --config configs/posttrained.yaml
python main.py verify-cryptobench
```

## Layout

```
main.py          entry point for every command
theory/          capacity arithmetic and trace-complexity estimators
data/            benchmark loaders, CryptoBench generator, prompts
estimation/      critic scores, projections, calibration
experiments/     generation, gap, perturbation, patching, fine-tuning, aggregation
configs/         one configuration per experiment; configs/models/ pins each checkpoint
cryptobench/     released CryptoBench instances
```

| File | Contents |
|---|---|
| `theory/capacity.py` | Channel capacity, effective capacity, projection loss, recovery bound, ceilings |
| `theory/complexity.py` | Program-length and Lempel-Ziv trace-complexity estimators |
| `data/schema.py` | Instance schema, JSONL I/O, answer normalisation, benchmark base class |
| `data/benchmarks.py` | Public benchmark loaders |
| `data/cryptobench.py` | CryptoBench generator, answer checking, complexity report, release format |
| `data/prompts.py` | Few-shot exemplars and prompt construction |
| `data/registry.py` | Benchmark lookup, generation caps, data preparation |
| `estimation/mine.py` | Donsker-Varadhan and InfoNCE critic scores, convergence statistics |
| `estimation/representations.py` | Mean input embeddings, PCA projection, standardisation |
| `estimation/calibration.py` | Critic calibration on synthetic Gaussian pairs of known information |
| `experiments/config.py` | YAML loading with inheritance and overrides |
| `experiments/seeding.py` | Seeding of every stochastic source |
| `experiments/model.py` | Checkpoint loading, greedy decoding with a KV cache, answer extraction |
| `experiments/generation.py` | Trace generation, per-token statistics, effective capacity |
| `experiments/faithfulness.py` | Gap estimation, per-category gaps, sample-size sensitivity, estimator comparison |
| `experiments/perturbation.py` | Step resampling and sensitivity |
| `experiments/patching.py` | Attention-recovery estimation |
| `experiments/ib.py` | Bottleneck fine-tuning and the penalty sweep |
| `experiments/tables.py` | Aggregation into markdown and JSON tables |

## Configuration

| Key | Values |
|---|---|
| `seed` | `0` (global random seed) |
| `prompting.k_shot` | `8` (number of in-context demonstrations) or `0` (zero-shot) |
| `generation.per_benchmark_T.<name>` | `512` / `384` / `307` (maximum trace length per benchmark) |
| `mine.n_pairs` | `1000` (sample pairs per MINE seed) |
| `mine.seeds` | `[0, 1, 2, 3, 4]` (independent MINE seeds) |
| `perturbation.temperature` | `1.0` (sampling temperature for the resampled step) |
| `perturbation.max_step_tokens` | `64` (maximum tokens per resampled step) |
| `ib.betas` | `[0.0, 0.01, 0.1, 1.0, 10.0]` (bottleneck penalty weights) |
| `ib.lora_r` | `16` (LoRA rank) |
| `ib.lora_alpha` | `32` (LoRA scaling) |
| `ib.kl_reduction` | `mean` (bottleneck KL in bits averaged over hidden dimensions) or `sum` (summed over them) |
| `representations.h0_pca_fit` | `validation_positions` (state at every generated validation position) or `validation_prompt_final` (validation prompt-final states) |
| `representations.answer_source` | `model` (generated answer) or `gold` (reference answer) |
| `entropy.mode` | `nll` (surprisal of the emitted token) or `shannon` (next-token entropy) |
| `patching.mechanism` | `kv_cache` (attention-cache splice) or `token_prefix` (token substitution) |

## License

MIT (`LICENSE`). CryptoBench: `cryptobench/README.md`.
