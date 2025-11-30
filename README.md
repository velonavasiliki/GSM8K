# Synthetic Chain-of-Thought Distillation for Mathematical Reasoning

**End-to-end ML pipeline for improving small language model performance through knowledge distillation from multiple teacher models.**

This project implements a complete machine learning workflow that generates synthetic training data from three large language models (GPT-4o-mini, DeepSeek, Gemini), fine-tunes a compact 1.5B parameter student model (Qwen2.5-1.5B-Instruct), and demonstrates measurable improvements on the GSM8K mathematical reasoning benchmark.

## Key Results

**Full-scale experiment (7,473 training examples, 1,319 test examples):**

| Model | Accuracy | Improvement | Training Loss | Notes |
|-------|----------|-------------|---------------|-------|
| Qwen2.5-1.5B Base | 60.96% | baseline | - | No fine-tuning |
| **+ GPT-4o-mini CoT** | **67.17%** | **+6.21%** | 0.23 | Best performer |
| + DeepSeek CoT | 64.59% | +3.63% | 0.38 | Strong improvement |
| + Gemini CoT | 62.32% | +1.36% | 0.34 | Modest gain |
| + GSM8K Original | 47.31% | -13.65% | 0.40 | Catastrophic forgetting |

### Impact

- **10% relative improvement** over baseline using GPT-4o-mini synthetic data
- **Identified critical failure mode**: Training on original GSM8K data caused severe performance degradation (-13.65%), suggesting that explanation quality and presentation style significantly impact learning effectiveness, even when both datasets contain correct step-by-step solutions.

## Technical Implementation

### System Architecture

```
┌─────────────────┐
│  Teacher APIs   │  GPT-4o-mini, DeepSeek, Gemini
└────────┬────────┘
         │ Generate synthetic CoT
         ▼
┌─────────────────┐
│ 7,473 Examples  │  Parquet datasets with reasoning chains
└────────┬────────┘
         │ Fine-tune with LoRA
         ▼
┌─────────────────┐
│ Student Models  │  4 variants of Qwen2.5-1.5B
└────────┬────────┘
         │ Evaluate
         ▼
┌─────────────────┐
│ GSM8K Test Set  │  1,319 math problems
└─────────────────┘
```

### Pipeline Components

**1. Synthetic Data Generation**
- Orchestrated API calls to 3 teacher models with rate limiting and retry logic
- Implemented concurrent processing using Python multiprocessing
- Generated 100% successful completions across 7,473 examples per teacher
- Total generation time: ~40 hours across all teachers

| Teacher | API | Time | Throughput |
|---------|-----|------|------------|
| GPT-4o-mini | OpenAI | 17.1h | 7.3 ex/min |
| DeepSeek-chat | DeepSeek | 19.4h | 6.4 ex/min |
| Gemini 2.0 Flash | Google | 3.5h | 35.6 ex/min |

**2. Model Fine-Tuning**
- Implemented Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters
- Configured 4-bit quantization (NF4) for memory efficiency
- Trained 73.9M parameters (7.67% of total) instead of full 962M parameters
- Training time: 45-65 minutes per model on NVIDIA RTX 3090

**Configuration:**
```python
LoRA: rank=64, alpha=128, dropout=0.08
Optimizer: AdamW with cosine LR schedule (1e-5 peak, 10% warmup)
Batch: size=4, gradient_accumulation=4 (effective=16)
Precision: 4-bit NF4 quantization + bfloat16 compute
Epochs: 2 over full training set
```

**3. Evaluation Pipeline**
- Parallel evaluation on 2 GPUs using multiprocessing (2 models simultaneously)
- Reduced total evaluation time from ~35 hours to ~18 hours
- Greedy decoding with regex-based answer extraction
- Comprehensive metrics collection and sample prediction tracking

### Engineering Highlights

**Scalability & Efficiency:**
- Parallel GPU utilization for evaluation workloads
- Efficient data storage using Parquet format
- Modular codebase supporting easy experimentation with different teachers/hyperparameters

**Robustness:**
- Error handling with exponential backoff for API calls
- Checkpoint saving during training
- Incremental result persistence (saves results to JSON after each model completes)

**Reproducibility:**
- All hyperparameters version-controlled in config files
- Complete training logs with losses and metrics
- Deterministic evaluation with fixed random seeds

## Project Structure

```
GSM8K/
├── scripts/
│   ├── cot_gen.py          # Synthetic data generation pipeline
│   ├── fine_tune.py        # LoRA fine-tuning with multiprocessing support
│   ├── evaluate.py         # Parallel evaluation framework
│   └── config.py           # Centralized hyperparameter configuration
├── data/
│   ├── gpt_cot.parquet     # 7,473 GPT-4o-mini reasoning examples
│   ├── deepseek_cot.parquet
│   └── gemini_cot.parquet
├── results/
│   ├── evaluation_results.json  # Final metrics for all models
│   └── training_log.json        # Training metadata and losses
└── notebooks/
    └── GSM8K_Qwen.ipynb    # Initial prototyping (100 examples)
```
## Summary of frameworks and technologies

**Machine Learning:**
- Knowledge distillation and transfer learning
- Parameter-efficient fine-tuning (LoRA)
- Model quantization and optimization
- Benchmark evaluation and analysis

**ML Engineering:**
- End-to-end pipeline development (data generation → training → evaluation)
- Multi-GPU orchestration and parallel processing
- API integration with rate limiting and error handling
- Experiment tracking and reproducible workflows

**Technologies:**
- PyTorch, Transformers (HuggingFace), PEFT, BitsAndBytes
- Python multiprocessing for parallelization
- Cloud GPU infrastructure (RunPod)
- Data processing with Datasets library and Parquet

## Experimental Insights

### Small-Scale vs. Full-Scale

Initial 100-example experiment vs. full 7,473-example training revealed:

| Metric | Small (100) | Full (7,473) | Insight |
|--------|-------------|--------------|---------|
| GPT improvement | +7% | +6.21% | Consistent |
| DeepSeek improvement | +6% | +3.63% | Diminishing returns at scale |
| Gemini improvement | +3% | +1.36% | Limited transferability |
| GSM8K original | +3% | -13.65% | Explanation quality mismatch |

**Key Finding:** While both GSM8K original (compact notation: `<<48/2=24>>`) and synthetic CoT (structured explanations with clear steps) contain correct reasoning, the synthetic data's pedagogical clarity proved far more effective for knowledge transfer. This demonstrates that not just correctness, but the quality and structure of explanations critically impact fine-tuning success.

### Alignment with Current Research

Our observed catastrophic forgetting when fine-tuning on GSM8K original data aligns with recent literature documenting similar challenges. [Research on fine-tuning instruction-tuned models](https://arxiv.org/html/2506.17209v1) shows that GSM8K fine-tuning can cause substantial capability degradation through catastrophic forgetting, with practitioners [reporting difficulties](https://discuss.huggingface.co/t/bad-performance-finetuning-llama-chat-and-instruct-models-on-gsm8k/126058) achieving satisfactory results even after extensive hyperparameter tuning. Our work extends these findings by demonstrating that synthetic Chain-of-Thought data with improved pedagogical structure successfully avoids this degradation, achieving +6.21% improvement where the original format caused -13.65% degradation. This validates recent work on [mitigating forgetting through data quality](https://openreview.net/forum?id=13HPTmZKbM) rather than architectural modifications alone.

## Setup and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create `.env` file:
```
OPENAI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
GEMINI_API_KEY=your_key
```

### Running the Pipeline

```bash
# Full pipeline
./run_pipeline.sh

# Or run components individually
python scripts/cot_gen.py                    # Generate synthetic data
python scripts/fine_tune.py --teacher all    # Train all models
python scripts/evaluate.py --teacher all     # Evaluate all models
```

## References

**Datasets & Models:**
- [GSM8K Dataset](https://github.com/openai/grade-school-math) - 8K grade school math problems
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) - Student model
- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) - Math-specialized variants

**Related Research:**
- [An Empirical Study of Catastrophic Forgetting in LLMs](https://arxiv.org/abs/2308.08747) - Understanding performance degradation during fine-tuning
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Original CoT research

## License

MIT License

---

**Project Highlights:**
- Complete ML pipeline from data generation to evaluation
- Multi-GPU orchestration and parallel processing
- 10% performance improvement through knowledge distillation
- Robust error handling with retry logic and result persistence
- Reproducible experiments with version-controlled configs
