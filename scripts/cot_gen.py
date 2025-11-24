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
from dotenv import load_dotenv
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from openai import OpenAI
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEACHERS = ["gpt", "deepseek", "gemini"]
TEACHER_GPT = "gpt-4o-mini"
TEACHER_DEEPSEEK = "deepseek-chat"
TEACHER_GEMINI = "gemini-2.0-flash"
TEACHER_TEMPERATURE = 0.3
TEACHER_MAX_TOKENS = 1024

def retry_with_backoff(func, max_retries=3, initial_delay=1.0):
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
    """Create a Spark session."""
    return SparkSession.builder \
        .appName("GSM8K-SyntheticCoT") \
        .master("local[3]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "3") \
        .getOrCreate()

def process_partition(partition, teacher: str):
    """Process a partition of questions with a single API client per partition."""
    
    prompt_template = (
        "Solve this grade school math problem step by step. "
        "Show your reasoning clearly. End with the final numerical answer after '####'.\n\n"
        "Problem: {}\n\nSolution:"
    )
    
    config = {"temperature": TEACHER_TEMPERATURE, "max_tokens": TEACHER_MAX_TOKENS}
    
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
                        **config
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
    
    # Save statistics to txt file
    stats_path = output_path.replace(".parquet", "_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Teacher: {teacher.upper()}\n")
        f.write(f"Total examples: {total}\n")
        f.write(f"Successful: {total - errors}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Success rate: {(total - errors) / total * 100:.2f}%\n")
        f.write(f"Output file: {output_path}\n")

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
