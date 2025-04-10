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
    **kwargs: Any
) -> pd.DataFrame:
    """
    Processes a DataFrame concurrently by splitting it into chunks,
    processing each chunk in parallel using safe_main_main with retry logic,
    and then concatenates the result.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame to process.
        n_splits (Optional[int]): Number of chunks to split into. Defaults to max_workers.
        max_workers (int): Maximum number of parallel workers.
        **kwargs: Additional keyword arguments passed to main.main.
        
    Returns:
        pd.DataFrame: The concatenated result from processing each chunk.
    """
    if n_splits is None:
        n_splits = max_workers

    # Split the DataFrame into roughly equal chunks.
    chunks = np.array_split(df, n_splits)
    results = [None] * n_splits

    # Process each chunk in parallel.
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
                # If an error persists, return an empty DataFrame for this chunk.
                results[index] = pd.DataFrame()

    # Concatenate all processed chunks into a single DataFrame.
    final_df = pd.concat(results, ignore_index=True)
    return final_df
