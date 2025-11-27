"""
Evaluate fine-tuned Qwen models on GSM8K test set.

Output:
    results/evaluation_results.json  - Accuracy and sample predictions per teacher
"""

import os
import re
import json
import argparse
from typing import Dict, List
import multiprocessing as mp

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Load environment variables and configuration
from dotenv import load_dotenv
load_dotenv()
from config import TRAINING_CONFIG, EVAL_CONFIG, MODELS_DIR, RESULTS_DIR, TEACHER_CONFIG


def get_bnb_config() -> BitsAndBytesConfig:
    """Create quantization configuration for inference."""
    return BitsAndBytesConfig(
        load_in_4bit=TRAINING_CONFIG.use_4bit,
        bnb_4bit_use_double_quant=TRAINING_CONFIG.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=TRAINING_CONFIG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, TRAINING_CONFIG.bnb_4bit_compute_dtype)
    )


def load_base_model(tokenizer):
    """Load base model without any adapter."""
    print(f"\nLoading base model: {TRAINING_CONFIG.student_model}")

    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG.student_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    print("Base model loaded successfully")
    return model


def load_model_with_adapter(teacher: str, tokenizer):
    """Load base model and apply LoRA adapter."""
    print(f"\nLoading base model: {TRAINING_CONFIG.student_model}")

    bnb_config = get_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG.student_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Load adapter
    adapter_path = os.path.join(MODELS_DIR, f"qwen-{teacher}", "adapter")
    print(f"Loading adapter from: {adapter_path}")

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("Model loaded successfully")
    return model


def extract_answer(text: str) -> str:
    """Extract numerical answer from model output.

    Looks for patterns like:
    - #### 42
    - The answer is 42
    - = 42 (at end)
    """
    # Try to find #### pattern first (GSM8K format)
    pattern = r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).replace(",", "")

    # Try "answer is X" pattern
    pattern = r"(?:answer is|answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)"
    match = re.search(pattern, text.lower())
    if match:
        return match.group(1).replace(",", "")

    # Try to find last number with = or : before it
    pattern = r"[=:]\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$"
    match = re.search(pattern, text)
    if match:
        return match.group(1).replace(",", "")

    # Last resort: find last number in text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove commas and whitespace
    answer = answer.replace(",", "").strip()

    # Convert to float and back to handle decimals
    try:
        num = float(answer)
        # If it's a whole number, return as int
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer


def evaluate_model(teacher: str, model, tokenizer) -> Dict:
    """Evaluate model on GSM8K test set."""
    print(f"\n{'='*60}")
    print(f"EVALUATING {teacher.upper()} MODEL")
    print(f"{'='*60}\n")

    # Load test dataset
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    # Use subset if specified
    if EVAL_CONFIG.test_size < len(test_data):
        test_data = test_data.select(range(EVAL_CONFIG.test_size))

    print(f"Evaluating on {len(test_data)} examples")

    # Evaluate
    correct = 0
    total = 0
    sample_predictions = []

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        question = example["question"]
        ground_truth = example["answer"].split("####")[-1].strip()

        # Format prompt
        messages = [
            {"role": "user", "content": f"Solve this grade school math problem step by step. Show your reasoning clearly. End with the final numerical answer after '####'.\n\n{question}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EVAL_CONFIG.max_new_tokens,
                do_sample=False,  # Greedy decoding for evaluation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and extract answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        predicted_answer = extract_answer(response)

        # Normalize and compare
        predicted_norm = normalize_answer(predicted_answer)
        ground_truth_norm = normalize_answer(ground_truth)

        is_correct = predicted_norm == ground_truth_norm
        if is_correct:
            correct += 1
        total += 1

        # Save first 5 examples as samples
        if i < 5:
            sample_predictions.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "full_response": response[:500],  # Truncate for readability
                "correct": is_correct
            })

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results for {teacher.upper()}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"{'='*60}\n")

    return {
        "teacher": teacher,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "sample_predictions": sample_predictions
    }


def save_evaluation_results(results: List[Dict]):
    """Save evaluation results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")

    # Load existing results if file exists
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    # Add new results
    for r in results:
        results_dict[r["teacher"]] = r

    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Evaluation results saved to: {results_path}")


def evaluate_on_gpu(teacher: str, gpu_id: int):
    """Evaluate a single model on a specific GPU."""
    # Set which GPU to use for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"\n[GPU {gpu_id}] Starting evaluation for {teacher.upper()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_CONFIG.student_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and evaluate model
    if teacher == "base":
        model = load_base_model(tokenizer)
    else:
        model = load_model_with_adapter(teacher, tokenizer)

    results = evaluate_model(teacher, model, tokenizer)

    # Save results immediately
    save_evaluation_results([results])

    # Clean up
    del model
    torch.cuda.empty_cache()

    print(f"\n[GPU {gpu_id}] Completed evaluation for {teacher.upper()}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on GSM8K")
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        choices=["gpt", "deepseek", "gemini", "gsm8k", "all", "base"],
        help="Which teacher model's fine-tuned student to evaluate (use 'base' for baseline, 'gsm8k' for GSM8K-finetuned)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate models in parallel on separate GPUs (requires multi-GPU instance)"
    )
    args = parser.parse_args()

    # Determine which teachers to evaluate
    all_teachers = []
    if args.teacher in ["base", "all"]:
        all_teachers.append("base")

    if args.teacher == "all":
        all_teachers.extend(TEACHER_CONFIG.teachers + ["gsm8k"])
    elif args.teacher not in ["base", "all"]:
        all_teachers.append(args.teacher)

    # Parallel evaluation on multiple GPUs
    if args.parallel and len(all_teachers) > 1:
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"PARALLEL EVALUATION MODE")
        print(f"Available GPUs: {num_gpus}")
        print(f"Models to evaluate: {len(all_teachers)}")
        print(f"{'='*60}\n")

        # Process in batches of num_gpus (evaluate num_gpus models at a time)
        all_results = []
        for i in range(0, len(all_teachers), num_gpus):
            batch = all_teachers[i:i+num_gpus]
            processes = []

            for j, teacher in enumerate(batch):
                gpu_id = j % num_gpus
                p = mp.Process(target=evaluate_on_gpu, args=(teacher, gpu_id))
                p.start()
                processes.append(p)
                print(f"Launched {teacher.upper()} evaluation on GPU {gpu_id}")

            # Wait for batch to complete
            for p in processes:
                p.join()

            print(f"\nCompleted batch {i//num_gpus + 1}/{(len(all_teachers)-1)//num_gpus + 1}")

        print("\n" + "="*60)
        print("PARALLEL EVALUATION COMPLETED")
        print("="*60)

    # Sequential evaluation (original behavior)
    else:
        if args.parallel:
            print("⚠ --parallel flag ignored (only one model or not supported)")

        # Load tokenizer once
        tokenizer = AutoTokenizer.from_pretrained(TRAINING_CONFIG.student_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        all_results = []

        for teacher in all_teachers:
            print("\n" + "="*60)
            if teacher == "base":
                print("EVALUATING BASE MODEL (NO FINE-TUNING)")
                print("="*60)
                model = load_base_model(tokenizer)
            else:
                print(f"EVALUATING {teacher.upper()} MODEL")
                print("="*60)
                model = load_model_with_adapter(teacher, tokenizer)

            results = evaluate_model(teacher, model, tokenizer)
            all_results.append(results)

            # Save after each model
            save_evaluation_results([results])

            # Clean up
            del model
            torch.cuda.empty_cache()

    # Load all results from file for summary
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results_dict = json.load(f)

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for teacher, result in all_results_dict.items():
            model_name = "BASE (no fine-tuning)" if teacher == 'base' else teacher.upper()
            print(f"{model_name}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

        # Print improvements if base was evaluated
        if 'base' in all_results_dict and len(all_results_dict) > 1:
            base_result = all_results_dict['base']
            print("\n" + "-"*60)
            print("IMPROVEMENTS OVER BASE MODEL")
            print("-"*60)
            for teacher, result in all_results_dict.items():
                if teacher != 'base':
                    improvement = result['accuracy'] - base_result['accuracy']
                    sign = "+" if improvement >= 0 else ""
                    print(f"{teacher.upper()}: {sign}{improvement:.2%} ({sign}{improvement*100:.2f} points)")
        print("="*60)


if __name__ == "__main__":
    main()
