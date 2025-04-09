import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from llmInvocation import main


def parallel_process_df(
    df: pd.DataFrame,
    n_splits: Optional[int] = None,
    max_workers: int = 4,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Processes a concatenated pandas DataFrame concurrently by splitting it into chunks and
    applying the main.main function to each chunk.

    All calls receive the same table_name provided in kwargs, so that every chunk is processed
    under the same table name. It is assumed that main.main handles merging or appending data
    appropriately.

    Parameters:
        df (pd.DataFrame): The concatenated input DataFrame to process.
        n_splits (Optional[int]): The number of chunks to split the DataFrame into. If not provided,
            it defaults to max_workers.
        max_workers (int): The maximum number of parallel workers. Defaults to 4.
        **kwargs: Additional keyword arguments to pass to main.main (e.g., api_key, temperature,
            azure_endpoint, api_version, text_column, table_name, spark, llm_model).

    Returns:
        pd.DataFrame: The concatenated result DataFrame obtained by combining the outputs from processing each chunk.
    """
    if n_splits is None:
        n_splits = max_workers

    # Split the DataFrame into approximately equal chunks.
    chunks = np.array_split(df, n_splits)
    results = [None] * n_splits

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(main.main, df=chunk, **kwargs): index
            for index, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as error:
                print(f"Error processing chunk {index}: {error}")
                results[index] = pd.DataFrame()  # On error, use an empty DataFrame.

    # Concatenate all processed chunks into a single DataFrame.
    final_df = pd.concat(results, ignore_index=True)
    return final_df
