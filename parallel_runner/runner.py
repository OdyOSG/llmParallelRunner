import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Optional
from llm_invocation import main


def safe_main_main(*args, **kwargs) -> pd.DataFrame:
    """
    Calls main.main with a retry mechanism to handle transient concurrent write errors.
    
    Retries the main.main call up to max_retries if a concurrent write error is detected.
    
    Parameters:
        *args, **kwargs: Arguments passed directly to main.main.
        
    Returns:
        pd.DataFrame: The result from main.main.
        
    Raises:
        Exception: If the function fails after the maximum number of retries.
    """
    max_retries = 3
    delay = 1  # seconds delay between retries
    attempt = 0

    while attempt < max_retries:
        try:
            # Attempt to process the chunk.
            return main.main(*args, **kwargs)
        except Exception as error:
            error_message = str(error).lower()
            # Check for concurrent write or transaction conflict errors.
            if "concurrent write" in error_message or "transaction conflict" in error_message:
                attempt += 1
                print(f"Concurrent write error encountered; retrying attempt {attempt}/{max_retries} after {delay} second(s)...")
                time.sleep(delay)
            else:
                # If it's not a concurrent write error, re-raise the exception.
                raise

    raise Exception("Failed to process chunk after multiple retries due to concurrent write errors.")


def parallel_process_df(
    df: pd.DataFrame,
    n_splits: Optional[int] = None,
    max_workers: int = 4,
    max_main_retries: int = 3,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Processes a DataFrame concurrently by splitting it into chunks, processing each chunk
    in parallel using safe_main_main with retry logic, and concatenating the results.
    
    Additionally, the function runs the processing multiple times (up to max_main_retries attempts)
    if not all rows have been processed. After each attempt, it:
      - Ensures that pmcid values are strings.
      - Queries the output table (specified by 'table_name') via the Spark session (passed as 'spark')
        to retrieve the list of processed pmcid ids.
      - Determines how many rows (and texts in the 'methods' column) remain unprocessed.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame to process. Must contain columns 'pmcid' and 'methods'.
        n_splits (Optional[int]): Number of chunks to split into. Defaults to max_workers.
        max_workers (int): Maximum number of parallel workers.
        max_main_retries (int): Number of total processing attempts if some rows remain unprocessed.
        **kwargs: Additional keyword arguments passed to main.main.
            Must include:
              - spark: SparkSession object used to query the output table.
              - table_name: Name of the table where processed pmcid ids are stored.
              
    Returns:
        pd.DataFrame: A concatenated DataFrame of the results from all attempts.
    """
    # Ensure pmcid is always a string.
    df['pmcid'] = df['pmcid'].astype(str)
    
    # Validate required parameters in kwargs.
    if 'spark' not in kwargs or 'table_name' not in kwargs:
        raise ValueError("Both 'spark' and 'table_name' must be provided in kwargs.")

    spark = kwargs['spark']
    table_name = kwargs['table_name']
    
    overall_results = []
    remaining_df = df.copy()
    total_initial = df.shape[0]
    
    for attempt in range(max_main_retries):
        if remaining_df.empty:
            print("All rows have been processed.")
            break
        
        print(f"Main run attempt {attempt + 1}/{max_main_retries} with {remaining_df.shape[0]} rows to process.")
        
        if n_splits is None:
            n_splits = max_workers
        
        # Split the remaining DataFrame into roughly equal chunks.
        chunks = np.array_split(remaining_df, n_splits)
        results = [None] * n_splits
        
        # Process each chunk concurrently.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(safe_main_main, df=chunk, **kwargs): index
                for index, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as error:
                    print(f"Error processing chunk {index}: {error}")
                    results[index] = pd.DataFrame()
        
        # Concatenate the results from this processing attempt.
        attempt_df = pd.concat(results, ignore_index=True)
        overall_results.append(attempt_df)
        
        # Query the processed pmcid ids from the output table.
        query = f"SELECT DISTINCT(pmcid) FROM {table_name}"
        processed_df = spark.sql(query).toPandas()
        processed_ids = set(processed_df['pmcid'].astype(str).tolist())
        
        # Determine which rows from the original DataFrame have not yet been processed.
        remaining_df = df[~df['pmcid'].isin(processed_ids)].copy()
        
        # Count missing texts (empty string or NaN) in the remaining rows.
        missing_texts_count = remaining_df['methods'].isna().sum() + (remaining_df['methods'] == "").sum()
        print(f"After attempt {attempt + 1}, processed {total_initial - remaining_df.shape[0]}/{total_initial} pmcid ids. Missing texts count: {missing_texts_count}.")
        
        if remaining_df.empty:
            print("All rows processed successfully.")
            break
        else:
            print(f"Retrying for the remaining {remaining_df.shape[0]} rows...")
            time.sleep(1)  # Optional delay between attempts.
    
    if overall_results:
        final_df = pd.concat(overall_results, ignore_index=True)
    else:
        final_df = pd.DataFrame()
    
    return final_df
