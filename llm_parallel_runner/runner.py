import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
from llmInvocation import main


def parallel_process_dfs(
    dfs_dict: Dict[str, pd.DataFrame],
    max_workers: int = 4,
    **kwargs: Any
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Processes multiple pandas DataFrames concurrently using the main.main function.

    Parameters:
        dfs_dict (Dict[str, pd.DataFrame]): A dictionary where keys are unique identifiers
            (e.g., table names) and values are the corresponding pandas DataFrames.
        max_workers (int, optional): The maximum number of parallel workers. Defaults to 4.
        **kwargs: Additional keyword arguments to pass to main.main (e.g., api_key, temperature,
            azure_endpoint, api_version, text_column, spark, llm_model).

    Returns:
        Dict[str, Optional[pd.DataFrame]]: A dictionary mapping each key from `dfs_dict` to its
            resulting DataFrame. If an error occurs for a particular DataFrame, the corresponding
            value is set to None.
    """
    results: Dict[str, Optional[pd.DataFrame]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(main.main, df=df, **kwargs): key
            for key, df in dfs_dict.items()
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as error:
                print(f"Error processing DataFrame for key '{key}': {error}")
                results[key] = None

    return results
