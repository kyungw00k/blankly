from blankly.exchanges.exchange import Exchange
from blankly.exchanges.auth.auth_constructor import AuthConstructor
from blankly.exchanges.interfaces.upbit.upbit_api import create_upbit_client

class Upbit(Exchange):
    def __init__(self, portfolio_name=None, keys_path="keys.json", settings_path=None):
        # Initialize with the exchange name
        Exchange.__init__(self, "upbit", portfolio_name, settings_path)
        
        # Load the auth from the keys file
        auth = AuthConstructor(keys_path, portfolio_name, 'upbit', ['ACCESS_KEY', 'SECRET_KEY'])
        
        # Create authenticated client
        calls = create_upbit_client(auth)
        
        # Add preferences to the calls object
        calls.preferences = self.preferences
        
        # Always finish the method with this function
        super().construct_interface_and_cache(calls)
    
    def get_exchange_state(self):
        return self.interface.get_products()
        
    def get_asset_state(self, symbol):
        return self.interface.get_account(symbol)
        
    def get_direct_calls(self):
        return self.calls 