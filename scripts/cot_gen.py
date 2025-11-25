"""
Generate synthetic chain-of-thought datasets using three teacher models in parallel.
Each teacher model processes the GSM8K dataset sequentially (no rate limit issues).

Output:
    data/gpt_cot.parquet
    data/deepseek_cot.parquet
    data/gemini_cot.parquet
"""

import os
import time
import random
import multiprocessing as mp
from typing import Any, Dict, List
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
from openai import OpenAI
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

from config import TEACHER_CONFIG, DATA_DIR, RESULTS_DIR

# Use configuration from config.py
OUTPUT_DIR = DATA_DIR
TEACHERS = TEACHER_CONFIG.teachers
TEACHER_GPT = TEACHER_CONFIG.gpt_model
TEACHER_DEEPSEEK = TEACHER_CONFIG.deepseek_model
TEACHER_GEMINI = TEACHER_CONFIG.gemini_model
TEACHER_TEMPERATURE = TEACHER_CONFIG.temperature
TEACHER_MAX_TOKENS = TEACHER_CONFIG.max_tokens


def retry_with_backoff(func, max_retries: int = 5, initial_delay: float = 2.0) -> Any:
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Check if it's a rate limit error
            error_msg = str(e).lower()
            if '429' in error_msg or 'rate limit' in error_msg or 'quota' in error_msg:
                # Longer backoff for rate limits
                delay = initial_delay * (3 ** attempt)
                print(f"    Rate limit hit, backing off for {delay:.1f}s...")
            else:
                # Standard exponential backoff for other errors
                delay = initial_delay * (2 ** attempt)
            time.sleep(delay)


def create_api_client(teacher: str):
    """Create API client for the specified teacher model."""
    if teacher == "gpt":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), None
    elif teacher == "deepseek":
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        ), None
    elif teacher == "gemini":
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        return None, genai.GenerativeModel(TEACHER_GEMINI)
    else:
        raise ValueError(f"Unknown teacher: {teacher}")


def generate_synthetic_answer(question: str, teacher: str, client, model) -> str:
    """Generate a synthetic answer for a question using the specified teacher model."""
    prompt_template = (
        "Solve this grade school math problem step by step. "
        "Show your reasoning clearly. End with the final numerical answer after '####'.\n\n"
        "Problem: {}\n\nSolution:"
    )
    prompt = prompt_template.format(question)
    model_config = {"temperature": TEACHER_TEMPERATURE, "max_tokens": TEACHER_MAX_TOKENS}

    def make_api_call():
        if teacher in ["gpt", "deepseek"]:
            model_name = TEACHER_GPT if teacher == "gpt" else TEACHER_DEEPSEEK
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **model_config
            )
            return resp.choices[0].message.content
        else:  # gemini
            resp = model.generate_content(prompt)
            return resp.text

    try:
        result = retry_with_backoff(make_api_call)

        # Add delay AFTER successful API call to respect rate limits
        # Paid tiers have generous limits, so we can use shorter delays
        base_delay = 0.1 if teacher == "gemini" else 0.2
        jitter = random.uniform(0, 0.1)
        time.sleep(base_delay + jitter)

        return result
    except Exception as e:
        return f"ERROR: {str(e)}"


def process_teacher(teacher: str) -> Dict:
    """Process all examples for a single teacher and save results."""
    print(f"\n[{teacher.upper()}] Starting generation...")
    start_time = time.time()

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]
    total_examples = len(train_data)
    print(f"[{teacher.upper()}] Loaded {total_examples} training examples")

    # Create API client
    client, model = create_api_client(teacher)

    # Process each example sequentially
    results = []
    errors = 0
    consecutive_errors = 0
    max_consecutive_errors = 10      # Stop after 10 consecutive failures
    error_rate_threshold = 0.5       # Stop if error rate > 50% in any 100-example window
    window_size = 100

    for i, example in enumerate(train_data):
        question = example["question"]
        answer = example["answer"]

        # Generate synthetic answer
        synthetic_answer = generate_synthetic_answer(question, teacher, client, model)

        if synthetic_answer.startswith("ERROR:"):
            errors += 1
            consecutive_errors += 1

            # Check for consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n[{teacher.upper()}] ❌ STOPPING: {max_consecutive_errors} consecutive API failures.")
                print(f"[{teacher.upper()}] This likely indicates an API key issue, network problem, or severe rate limiting.")
                print(f"[{teacher.upper()}] Processed {i + 1}/{total_examples} examples before stopping.")
                break
        else:
            consecutive_errors = 0  # Reset on success

        results.append({
            "question": question,
            "answer": answer,
            "synthetic_answer": synthetic_answer,
            "teacher": teacher
        })

        # Check rolling window error rate every 100 examples
        if (i + 1) % window_size == 0:
            # Count errors in the last 100 examples
            window_start = max(0, i + 1 - window_size)
            window_errors = sum(1 for r in results[window_start:i + 1] if r['synthetic_answer'].startswith("ERROR:"))
            window_error_rate = window_errors / window_size

            if window_error_rate > error_rate_threshold:
                print(f"\n[{teacher.upper()}] ❌ STOPPING: Error rate too high in last {window_size} examples ({window_error_rate:.1%} > {error_rate_threshold:.0%}).")
                print(f"[{teacher.upper()}] Please check your API key, rate limits, and network connection.")
                print(f"[{teacher.upper()}] Processed {i + 1}/{total_examples} examples before stopping.")
                break

        # Progress update every 100 examples
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total_examples - i - 1) / rate if rate > 0 else 0
            overall_error_rate = errors / (i + 1)
            print(f"[{teacher.upper()}] Progress: {i + 1}/{total_examples} "
                  f"({100 * (i + 1) / total_examples:.1f}%) - "
                  f"Errors: {errors} ({overall_error_rate:.1%}) - "
                  f"ETA: {remaining / 60:.1f} min")

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, f"{teacher}_cot.parquet")
    df.to_parquet(output_path, index=False)

    # Calculate statistics
    total_time = time.time() - start_time
    success_count = total_examples - errors

    print(f"[{teacher.upper()}] Completed! {success_count}/{total_examples} successful. "
          f"Saved to {output_path}")
    print(f"[{teacher.upper()}] Total time: {total_time / 60:.1f} minutes")

    # Save statistics to results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats_path = os.path.join(RESULTS_DIR, f"{teacher}_cot_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Teacher: {teacher.upper()}\n")
        f.write(f"Total examples: {total_examples}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Success rate: {100 * success_count / total_examples:.2f}%\n")
        f.write(f"Total time: {total_time / 60:.1f} minutes\n")
        f.write(f"Output file: {output_path}\n")

    print(f"[{teacher.upper()}] Statistics saved to {stats_path}")

    return {
        "teacher": teacher,
        "total": total_examples,
        "successful": success_count,
        "errors": errors,
        "time_minutes": total_time / 60
    }


def main():
    """Generate synthetic CoT datasets for all teachers in parallel."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("SYNTHETIC CoT GENERATION")
    print("="*60)
    print(f"Teachers: {', '.join([t.upper() for t in TEACHERS])}")
    print(f"Running teachers in parallel (one API call per teacher at a time)")
    print("="*60 + "\n")

    # Run teachers in parallel using multiprocessing
    with mp.Pool(processes=len(TEACHERS)) as pool:
        results = pool.map(process_teacher, TEACHERS)

    # Print summary
    print("\n" + "="*60)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*60)
    for result in results:
        print(f"{result['teacher'].upper()}: "
              f"{result['successful']}/{result['total']} successful "
              f"({100 * result['successful'] / result['total']:.1f}%) - "
              f"Time: {result['time_minutes']:.1f} min")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
