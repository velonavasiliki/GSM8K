"""
Generate synthetic chain-of-thought datasets using three teacher models sequentially with PySpark.
Each teacher model processes the GSM8K dataset with 3 parallel workers.

Output:
    data/gpt_cot.parquet
    data/deepseek_cot.parquet
    data/gemini_cot.parquet
"""

import os
import time
import random
from typing import Any
from dotenv import load_dotenv
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from openai import OpenAI
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

from config import TEACHER_CONFIG, SPARK_CONFIG, DATA_DIR, RESULTS_DIR
# Use configuration from config.py
OUTPUT_DIR = DATA_DIR
TEACHERS = TEACHER_CONFIG.teachers
TEACHER_GPT = TEACHER_CONFIG.gpt_model
TEACHER_DEEPSEEK = TEACHER_CONFIG.deepseek_model
TEACHER_GEMINI = TEACHER_CONFIG.gemini_model
TEACHER_TEMPERATURE = TEACHER_CONFIG.temperature
TEACHER_MAX_TOKENS = TEACHER_CONFIG.max_tokens

def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0) -> Any:
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            time.sleep(delay)

def create_spark_session():
    """Create a Spark session using configuration from config.py."""
    return SparkSession.builder \
        .appName(SPARK_CONFIG.app_name) \
        .master(f"local[{SPARK_CONFIG.num_workers}]") \
        .config("spark.driver.memory", SPARK_CONFIG.driver_memory) \
        .config("spark.sql.shuffle.partitions", str(SPARK_CONFIG.shuffle_partitions)) \
        .getOrCreate()

def process_partition(partition, teacher: str):
    """Process a partition of questions with a single API client per partition."""
    
    prompt_template = (
        "Solve this grade school math problem step by step. "
        "Show your reasoning clearly. End with the final numerical answer after '####'.\n\n"
        "Problem: {}\n\nSolution:"
    )
    
    model_config = {"temperature": TEACHER_TEMPERATURE, "max_tokens": TEACHER_MAX_TOKENS}
    
    # Initialize client once per partition
    client = None
    model = None
    
    if teacher == "gpt":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif teacher == "deepseek":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif teacher == "gemini":
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(TEACHER_GEMINI)
    
    # Process each row in the partition
    for row in partition:
        prompt = prompt_template.format(row.question)
        
        try:
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
            
            # Add jitter to prevent thundering herd
            base_delay = 0.3 if teacher == "gemini" else 0.5  # Gemini is more permissive
            jitter = random.uniform(0, 0.2)
            time.sleep(base_delay + jitter)
            synthetic_answer = retry_with_backoff(make_api_call)
                
        except Exception as e:
            synthetic_answer = f"ERROR: {str(e)}"
        
        yield (row.question, row.answer, synthetic_answer, teacher)


def process_teacher(spark, df, teacher: str, output_path: str):
    """Process all examples for a single teacher and save results."""
    print(f"\n[{teacher.upper()}] Starting generation...")
    
    # Broadcast environment variables to workers
    env_vars = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }
    broadcast_env = spark.sparkContext.broadcast(env_vars)
    
    # Use mapPartitions for efficient processing
    def partition_wrapper(partition):
        # Set env vars in worker
        env = broadcast_env.value
        for key, value in env.items():
            if value:
                os.environ[key] = value
        return process_partition(partition, teacher)
    
    # Apply mapPartitions
    result_rdd = df.rdd.mapPartitions(partition_wrapper)
    
    # Convert back to DataFrame
    schema = StructType([
        StructField("question", StringType(), True),
        StructField("answer", StringType(), True),
        StructField("synthetic_answer", StringType(), True),
        StructField("teacher", StringType(), True)
    ])
    result_df = spark.createDataFrame(result_rdd, schema)
    
    # Write to parquet
    result_df.write.mode("overwrite").parquet(output_path)
    
    # Count results
    total = result_df.count()
    errors = result_df.filter(result_df["synthetic_answer"].startswith("ERROR:")).count()
    
    print(f"[{teacher.upper()}] Completed! {total - errors}/{total} successful. Saved to {output_path}")

    # Save statistics to results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats_path = os.path.join(RESULTS_DIR, f"{teacher}_cot_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Teacher: {teacher.upper()}\n")
        f.write(f"Total examples: {total}\n")
        f.write(f"Successful: {total - errors}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Success rate: {(total - errors) / total * 100:.2f}%\n")
        f.write(f"Output file: {output_path}\n")

    print(f"[{teacher.upper()}] Statistics saved to {stats_path}")

    # Clean up broadcast variable
    broadcast_env.unpersist()

def main():
    # Create output directory and Spark session
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Initializing Spark...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]
    print(f"Loaded {len(train_data)} training examples")

    # Convert to Spark DataFrame
    pdf = train_data.to_pandas()
    df = spark.createDataFrame(pdf)

    # Cache the DataFrame since we'll use it 3 times
    df.cache()

    # Process each teacher sequentially - Spark handles parallelization within each
    for teacher in TEACHERS:
        output_path = os.path.join(OUTPUT_DIR, f"{teacher}_cot.parquet")
        process_teacher(spark, df, teacher, output_path)

    # Cleanup
    df.unpersist()
    spark.stop()

    print("\n" + "="*50)
    print("All datasets generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*50)


if __name__ == "__main__":
    main()
