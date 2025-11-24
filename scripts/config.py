"""
Configuration file
"""
import os
from dataclasses import dataclass
from typing import List

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Teacher model configurations
@dataclass
class TeacherConfig:
    """Configuration for teacher models used in CoT generation."""
    teachers: List[str] = None
    gpt_model: str = "gpt-4o-mini"
    deepseek_model: str = "deepseek-chat"
    gemini_model: str = "gemini-2.0-flash"
    temperature: float = 0.3
    max_tokens: int = 1024

    def __post_init__(self):
        if self.teachers is None:
            self.teachers = ["gpt", "deepseek", "gemini"]

# Spark configurations
@dataclass
class SparkConfig:
    """Configuration for Spark session."""
    app_name: str = "GSM8K-SyntheticCoT"
    num_workers: int = 3
    driver_memory: str = "4g"
    shuffle_partitions: int = 3

# Fine-tuning configurations
@dataclass
class TrainingConfig:
    """
    Configuration for fine-tuning.
    """
    student_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Training
    per_device_train_batch_size: int = 4       # Increased from 2 
    gradient_accumulation_steps: int = 4       # Effective batch size = 16
    learning_rate: float = 1e-5                # Conservative for full data
    num_epochs: int = 2                        # Reduced from 3 to avoid overfitting
    max_seq_length: int = 512

    # LoRA parameters
    lora_r: int = 64                           # Increased from 32
    lora_alpha: int = 128                      # Maintains 2:1 ratio with r
    lora_dropout: float = 0.08
    lora_target_modules: List[str] = None

    # Quantization
    use_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Optimizer & Scheduler
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1                  # Scales automatically with dataset
    weight_decay: float = 0.01

    # Memory optimization
    gradient_checkpointing: bool = True
    use_reentrant: bool = False

    # Logging and saving
    logging_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    eval_strategy: str = "steps"
    eval_steps: int = 500

    # Other
    fp16: bool = False
    bf16: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

# Evaluation configurations
@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    test_size: int = 1319                      # Full test set (or subset)
    batch_size: int = 8
    max_new_tokens: int = 1024                 # Match notebook

# Default instances
TEACHER_CONFIG = TeacherConfig()
SPARK_CONFIG = SparkConfig()
TRAINING_CONFIG = TrainingConfig()
EVAL_CONFIG = EvalConfig()
