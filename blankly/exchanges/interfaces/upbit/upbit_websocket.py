from typing import List, Callable
from pyupbit.websocket_api import WebSocketManager

class Tickers:
    def __init__(self, symbol: str, stream: str = "ticker"):
        """Initialize websocket connection
        Args:
            symbol (str): Trading pair (e.g. 'BTC-KRW')
            stream (str): Stream type ('ticker' or 'orderbook')
        """
        self.symbol = symbol
        self.stream = stream
        self.callbacks: List[Callable] = []
        self.ws: WebSocketManager = None
        
    def append_callback(self, callback: Callable) -> None:
        """Add a callback function to handle incoming messages"""
        self.callbacks.append(callback)
        
    def connect(self) -> None:
        """Connect to websocket and start receiving data"""
        if self.ws is not None:
            self.close()
            
        # Convert BTC-KRW to KRW-BTC format for Upbit
        if '-' in self.symbol:
            base, quote = self.symbol.split('-')
            if quote == 'KRW':
                symbol = f'KRW-{base}'
                
        self.ws = WebSocketManager(self.stream, [symbol])
        
        # Start message processing in a separate thread
        def message_handler():
            while True:
                data = self.ws.get()
                if data == 'ConnectionClosedError':
                    self.ws.terminate()
                    break
                for callback in self.callbacks:
                    callback(data)
                    
        import threading
        self.thread = threading.Thread(target=message_handler)
        self.thread.daemon = True
        self.thread.start()
        
    def close(self) -> None:
        """Close websocket connection"""
        if self.ws is not None:
            self.ws.terminate()
            self.ws = None 