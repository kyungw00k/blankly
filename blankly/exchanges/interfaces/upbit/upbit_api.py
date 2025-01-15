import pyupbit

def create_upbit_client(auth):
    """
    Create a new upbit client
    Args:
        auth (Auth): Blankly auth object
    Returns:
        upbit.Upbit: The authenticated upbit client
    """
    return pyupbit.Upbit(auth.keys['ACCESS_KEY'], auth.keys['SECRET_KEY']) 