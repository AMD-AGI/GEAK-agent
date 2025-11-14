from time import time


def get_time_spent(start_time: float) -> str:
    """
    Returns the time spent since the start_time in a human-readable format.
    
    Args:
        start_time (float): The starting time in seconds.
        
    Returns:
        str: A string representing the time spent in a human-readable format.
    """
    elapsed_time = time() - start_time
    if elapsed_time < 60:
        return f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        return f"{elapsed_time / 60:.2f} minutes"
    else:
        return f"{elapsed_time / 3600:.2f} hours"