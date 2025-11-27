"""
Fine-tune Qwen2.5-1.5B on synthetic CoT data from teacher models.

Output:
    models/{teacher}/adapter/  - LoRA adapter weights
    results/training_log.json  - Training metrics and hyperparameters
"""

import os
import json
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import Dict

import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Load environment variables and configuration
from dotenv import load_dotenv
load_dotenv()
from config import TRAINING_CONFIG, DATA_DIR, MODELS_DIR, RESULTS_DIR, TEACHER_CONFIG

def get_bnb_config() -> BitsAndBytesConfig:
    """Create quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=TRAINING_CONFIG.use_4bit,
        bnb_4bit_use_double_quant=TRAINING_CONFIG.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=TRAINING_CONFIG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, TRAINING_CONFIG.bnb_4bit_compute_dtype)
    )

def get_lora_config() -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=TRAINING_CONFIG.lora_r,
        lora_alpha=TRAINING_CONFIG.lora_alpha,
        target_modules=TRAINING_CONFIG.lora_target_modules,
        lora_dropout=TRAINING_CONFIG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

def get_training_args(output_dir: str) -> TrainingArguments:
    """Create training arguments from config."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG.num_epochs,
        per_device_train_batch_size=TRAINING_CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG.gradient_accumulation_steps,

        # Optimization
        learning_rate=TRAINING_CONFIG.learning_rate,
        lr_scheduler_type=TRAINING_CONFIG.lr_scheduler_type,
        warmup_ratio=TRAINING_CONFIG.warmup_ratio,
        weight_decay=TRAINING_CONFIG.weight_decay,

        # Memory optimization
        fp16=TRAINING_CONFIG.fp16,
        bf16=TRAINING_CONFIG.bf16,
        gradient_checkpointing=TRAINING_CONFIG.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": TRAINING_CONFIG.use_reentrant},

        # Logging
        logging_steps=TRAINING_CONFIG.logging_steps,
        logging_first_step=True,

        # Saving
        save_strategy=TRAINING_CONFIG.save_strategy,
        save_steps=TRAINING_CONFIG.save_steps,
        save_total_limit=TRAINING_CONFIG.save_total_limit,

        # Evaluation
        eval_strategy=TRAINING_CONFIG.eval_strategy,
        eval_steps=TRAINING_CONFIG.eval_steps,

        # Other
        optim=TRAINING_CONFIG.optim,
        report_to="none",
        seed=TRAINING_CONFIG.seed,
    )

def load_tokenizer():
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_CONFIG.student_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer

def load_model_for_training(bnb_config: BitsAndBytesConfig):
    """Load model with quantization for training."""
    print(f"Loading model: {TRAINING_CONFIG.student_model}")
    model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG.student_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    print(f"Model loaded successfully")
    return model

def load_synthetic_dataset(teacher: str, tokenizer) -> Dataset:
    """Load synthetic CoT data from parquet or GSM8K directly and format for training."""

    # Handle GSM8K original data (no synthetic CoT)
    if teacher == "gsm8k":
        print("Loading GSM8K original training data...")
        dataset = load_dataset("openai/gsm8k", "main")
        train_data = dataset["train"]
        print(f"Loaded {len(train_data)} examples from GSM8K")

        # Format for training
        def format_example(example):
            question = example['question']
            answer = example['answer']  # Original GSM8K answer

            messages = [
                {"role": "user", "content": f"Solve this grade school math problem step by step. Show your reasoning clearly. End with the final numerical answer after '####'.\n\n{question}"},
                {"role": "assistant", "content": answer}
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

        # Apply formatting
        formatted_dataset = train_data.map(format_example, remove_columns=train_data.column_names)
        print(f"Dataset formatted with {len(formatted_dataset)} examples")
        return formatted_dataset

    # Handle synthetic CoT data from teacher models
    else:
        data_path = os.path.join(DATA_DIR, f"{teacher}_cot.parquet")
        print(f"Loading dataset from: {data_path}")

        # Load parquet file
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df)} examples")

        # Format for training
        def format_example(row):
            question = row['question']
            synthetic_answer = row['synthetic_answer']

            messages = [
                {"role": "user", "content": f"Solve this grade school math problem step by step. Show your reasoning clearly. End with the final numerical answer after '####'.\n\n{question}"},
                {"role": "assistant", "content": synthetic_answer}
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

        # Convert to dataset
        formatted_data = [format_example(row) for _, row in df.iterrows()]
        dataset = Dataset.from_list(formatted_data)

        print(f"Dataset formatted with {len(dataset)} examples")
        return dataset

def train_model(teacher: str, tokenizer) -> Dict:
    """Train model on teacher's synthetic data and return training info."""
    print(f"\n{'='*60}")
    print(f"TRAINING ON {teacher.upper()} TEACHER DATA")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    # Setup paths
    output_dir = os.path.join(MODELS_DIR, f"qwen-{teacher}")
    adapter_path = os.path.join(output_dir, "adapter")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    train_dataset = load_synthetic_dataset(teacher, tokenizer)

    # Load model with quantization
    bnb_config = get_bnb_config()
    model = load_model_for_training(bnb_config)

    # Apply LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Setup training
    training_args = get_training_args(output_dir)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
        max_length=TRAINING_CONFIG.max_seq_length,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save adapter
    print(f"\nSaving adapter to: {adapter_path}")
    trainer.model.save_pretrained(adapter_path)

    # Calculate training time
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60  # minutes

    # Gather training info
    training_info = {
        "teacher": teacher,
        "timestamp": start_time.isoformat(),
        "dataset": {
            "source": os.path.join(DATA_DIR, f"{teacher}_cot.parquet"),
            "train_size": len(train_dataset)
        },
        "model": {
            "student": TRAINING_CONFIG.student_model,
            "teacher": getattr(TEACHER_CONFIG, f"{teacher}_model")
        },
        "hyperparameters": {
            "learning_rate": TRAINING_CONFIG.learning_rate,
            "batch_size": TRAINING_CONFIG.per_device_train_batch_size,
            "gradient_accumulation_steps": TRAINING_CONFIG.gradient_accumulation_steps,
            "effective_batch_size": TRAINING_CONFIG.per_device_train_batch_size * TRAINING_CONFIG.gradient_accumulation_steps,
            "num_epochs": TRAINING_CONFIG.num_epochs,
            "max_seq_length": TRAINING_CONFIG.max_seq_length,
            "lora_r": TRAINING_CONFIG.lora_r,
            "lora_alpha": TRAINING_CONFIG.lora_alpha,
            "lora_dropout": TRAINING_CONFIG.lora_dropout,
        },
        "training": {
            "total_steps": train_result.global_step,
            "time_minutes": round(training_time, 2),
            "final_loss": round(train_result.training_loss, 4),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percent": round(100 * trainable_params / total_params, 2)
        },
        "output_path": adapter_path
    }

    # Clean up
    del model, trainer
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Training completed in {training_time:.1f} minutes")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Adapter saved to: {adapter_path}")
    print(f"{'='*60}\n")

    return training_info


def save_training_log(training_info: Dict):
    """Save training information to JSON log."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "training_log.json")

    # Load existing log or create new
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {}

    # Add new training info
    teacher = training_info["teacher"]
    log_data[teacher] = training_info

    # Save updated log
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"Training log saved to: {log_path}")


def train_on_gpu(teacher: str, gpu_id: int):
    """Train a single teacher model on a specific GPU."""
    # Set which GPU to use for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"\n[GPU {gpu_id}] Starting training for {teacher.upper()}")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Train model
    training_info = train_model(teacher, tokenizer)
    save_training_log(training_info)

    print(f"\n[GPU {gpu_id}] Completed training for {teacher.upper()}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on synthetic CoT data or GSM8K original data")
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        choices=["gpt", "deepseek", "gemini", "gsm8k", "all"],
        help="Which teacher model's data to use for training (or 'gsm8k' for original answers)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train all teachers in parallel on separate GPUs (requires multi-GPU instance)"
    )
    args = parser.parse_args()

    # Determine which teachers to train
    if args.teacher == "all":
        teachers = TEACHER_CONFIG.teachers + ["gsm8k"]  # Include GSM8K original data
    else:
        teachers = [args.teacher]

    # Parallel training on multiple GPUs
    if args.parallel and len(teachers) > 1:
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"PARALLEL TRAINING MODE")
        print(f"Available GPUs: {num_gpus}")
        print(f"Teachers to train: {len(teachers)}")
        print(f"{'='*60}\n")

        if num_gpus < len(teachers):
            print(f"⚠ Warning: You have {num_gpus} GPUs but {len(teachers)} teachers.")
            print(f"Some teachers will share GPUs or train sequentially.")

        # Launch parallel processes
        processes = []
        for i, teacher in enumerate(teachers):
            gpu_id = i % num_gpus  # Distribute teachers across available GPUs
            p = mp.Process(target=train_on_gpu, args=(teacher, gpu_id))
            p.start()
            processes.append(p)
            print(f"Launched {teacher.upper()} training on GPU {gpu_id}")

        # Wait for all processes to complete
        print(f"\nWaiting for all training processes to complete...")
        for p in processes:
            p.join()

        print("\n" + "="*60)
        print("PARALLEL TRAINING COMPLETED SUCCESSFULLY")
        print(f"Models saved to: {MODELS_DIR}")
        print(f"Training logs saved to: {RESULTS_DIR}/training_log.json")
        print("="*60)

    # Sequential training (original behavior)
    else:
        if args.parallel:
            print("⚠ --parallel flag ignored (only one teacher or not supported)")

        tokenizer = load_tokenizer()

        # Train on each teacher sequentially
        for teacher in teachers:
            training_info = train_model(teacher, tokenizer)
            save_training_log(training_info)

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Models saved to: {MODELS_DIR}")
        print(f"Training logs saved to: {RESULTS_DIR}/training_log.json")
        print("="*60)


if __name__ == "__main__":
    main()