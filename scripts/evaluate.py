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

    # Organize results by teacher
    results_dict = {r["teacher"]: r for r in results}

    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Evaluation results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on GSM8K")
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        choices=["gpt", "deepseek", "gemini", "gsm8k", "all", "base"],
        help="Which teacher model's fine-tuned student to evaluate (use 'base' for baseline, 'gsm8k' for GSM8K-finetuned)"
    )
    args = parser.parse_args()

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_CONFIG.student_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    all_results = []

    # Evaluate base model if requested
    if args.teacher in ["base", "all"]:
        print("\n" + "="*60)
        print("EVALUATING BASE MODEL (NO FINE-TUNING)")
        print("="*60)
        base_model = load_base_model(tokenizer)
        base_results = evaluate_model("base", base_model, tokenizer)
        all_results.append(base_results)

        # Clean up
        del base_model
        torch.cuda.empty_cache()

    # Determine which teachers to evaluate
    if args.teacher == "all":
        teachers = TEACHER_CONFIG.teachers + ["gsm8k"]  # Include GSM8K-finetuned model
    elif args.teacher == "base":
        teachers = []
    else:
        teachers = [args.teacher]

    # Evaluate each teacher's model
    for teacher in teachers:
        model = load_model_with_adapter(teacher, tokenizer)
        results = evaluate_model(teacher, model, tokenizer)
        all_results.append(results)

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save results
    save_evaluation_results(all_results)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for result in all_results:
        model_name = "BASE (no fine-tuning)" if result['teacher'] == 'base' else result['teacher'].upper()
        print(f"{model_name}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    # Print improvements if base was evaluated
    base_result = next((r for r in all_results if r['teacher'] == 'base'), None)
    if base_result and len(all_results) > 1:
        print("\n" + "-"*60)
        print("IMPROVEMENTS OVER BASE MODEL")
        print("-"*60)
        for result in all_results:
            if result['teacher'] != 'base':
                improvement = result['accuracy'] - base_result['accuracy']
                sign = "+" if improvement >= 0 else ""
                print(f"{result['teacher'].upper()}: {sign}{improvement:.2%} ({sign}{improvement*100:.2f} points)")
    print("="*60)


if __name__ == "__main__":
    main()
