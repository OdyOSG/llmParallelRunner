import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, Optional
from llm_invocation import main

# Create a lock to serialize Delta table writes
delta_write_lock = threading.Lock()

def safe_main_main(*args, **kwargs) -> pd.DataFrame:
    """
    Wraps main.main such that Delta table operations are executed serially.
    """
    with delta_write_lock:
        return main.main(*args, **kwargs)

def parallel_process_df(
    df: pd.DataFrame,
    n_splits: Optional[int] = None,
    max_workers: int = 4,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Processes a DataFrame concurrently by splitting it into chunks,
    processing each chunk in parallel using safe_main_main (which wraps
    main.main with a lock to avoid concurrent Delta table writes), and
    then concatenates the result.

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
                results[index] = pd.DataFrame()  # Return an empty DataFrame if processing fails.

    # Concatenate all processed chunks into a single DataFrame.
    final_df = pd.concat(results, ignore_index=True)
    return final_df
