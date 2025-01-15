import time
import pandas as pd
from datetime import datetime
import pyupbit
import threading

from blankly.exchanges.interfaces.exchange_interface import ExchangeInterface
from blankly.exchanges.interfaces.upbit.upbit_websocket import Tickers
from blankly.utils import utils
from blankly.utils.exceptions import APIException

class UpbitInterface(ExchangeInterface):
    def __init__(self, exchange_name, authenticated_api):
        super().__init__(exchange_name, authenticated_api, 
                        valid_resolutions=[1, 3, 5, 10, 15, 30, 60, 240])
        
        # 마진/선물 거래 지원 여부는 클래스 속성으로 설정
        self.supports_margin = False
        self.supports_futures = False
        # Sandbox mode 설정 확인
        self.is_sandbox = self.calls.preferences['settings'].get('use_sandbox', False)
    
    def init_exchange(self):
        try:
            self.calls.get_balances()
        except Exception as e:
            raise APIException(f"Authentication failed: {e}")
    
    def get_products(self) -> dict:
        """Get all trading pairs on Upbit"""
        products = []
        markets = pyupbit.get_tickers(fiat="KRW")  # KRW 마켓만 가져오기
        
        for market in markets:
            product = {
                'symbol': market.replace('KRW-', '') + '-KRW',
                'base_asset': market.split('-')[1],
                'quote_asset': 'KRW',
                'base_min_size': 0.00000001,
                'base_max_size': 10000000000,
                'base_increment': 0.00000001
            }
            products.append(product)
        
        return products

    def get_account(self, symbol=None):
        """Get account balances"""
        balances = self.calls.get_balances()
        parsed = {}
        
        for balance in balances:
            currency = balance['currency']
            parsed[currency] = {
                'available': float(balance['balance']),
                'hold': float(balance['locked'])
            }
            
        if symbol is not None:
            symbol = utils.get_base_asset(symbol)
            if symbol in parsed:
                return parsed[symbol]
            return {'available': 0.0, 'hold': 0.0}
            
        return parsed

    def market_order(self, symbol: str, side: str, size: float):
        """Place market order"""
        if self.is_sandbox:
            order_id = f'mock_{int(time.time())}'
            # 주문 정보 저장
            if not hasattr(self, '_order_sides'):
                self._order_sides = {}
            self._order_sides[order_id] = side.lower()  # side 정보 저장
            
            mock_response = {
                'uuid': order_id,
                'side': 'bid' if side.lower() == 'buy' else 'ask',
                'ord_type': 'price' if side.lower() == 'buy' else 'market',
                'price': str(self.get_price(symbol) * size),
                'state': 'done',
                'market': symbol if symbol.startswith('KRW-') else f'KRW-{symbol.split("-")[0]}',
                'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S+09:00'),
                'volume': str(size),
                'executed_volume': str(size),
                'trades_count': 1,
                'trades': [{
                    'market': symbol if symbol.startswith('KRW-') else f'KRW-{symbol.split("-")[0]}',
                    'uuid': f'mock_trade_{int(time.time())}',
                    'price': str(self.get_price(symbol)),
                    'volume': str(size),
                    'funds': str(self.get_price(symbol) * size),
                    'side': 'bid' if side.lower() == 'buy' else 'ask'
                }]
            }
            return self._format_order(mock_response)

        # Convert BTC-KRW to KRW-BTC format
        if '-' in symbol:
            base, quote = symbol.split('-')
            if quote == 'KRW':
                symbol = f'KRW-{base}'
        
        # 최소 주문 금액 확인 (5000원)
        current_price = self.get_price(symbol)
        order_amount = current_price * size
        
        if side.lower() == 'buy':
            # 시장가 매수는 주문 금액으로 해야함
            order_amount = order_amount * 1.001  # 슬리피지 0.1% 추가
            if order_amount < 5000:
                raise APIException(f"Order amount ({order_amount:.0f} KRW) is less than minimum (5000 KRW)")
            response = self.calls.buy_market_order(symbol, order_amount)
        else:
            # 시장가 매도는 수량으로만 체크
            if current_price * size < 5000:
                raise APIException(f"Order amount ({current_price * size:.0f} KRW) is less than minimum (5000 KRW)")
            response = self.calls.sell_market_order(symbol, size)
            
        return self._format_order(response)

    def _adjust_price(self, currency: str, price: float, relative_slippage: int = 0) -> float:
        """
        업비트 주문가격 단위 맞추기
        Args:
            currency (str): 종목 (예: BTC, ETH)
            price (float): 원래 가격
        Returns:
            float: 조정된 가격
        """
        if price >= 2000000:
            unit = 1000
        elif price >= 1000000:
            unit = 500
        elif price >= 500000:
            unit = 100
        elif price >= 100000:
            unit = 50
        elif price >= 10000:
            unit = 10
        elif price >= 1000:
            unit = 1
        elif price >= 100:
            unit = 0.1
        elif price >= 10:
            unit = 0.01
        elif price >= 1:
            unit = 0.001
        elif price >= 0.1:
            unit = 0.0001
        elif price >= 0.01:
            unit = 0.00001
        elif price >= 0.001:
            unit = 0.000001
        elif price >= 0.0001:
            unit = 0.0000001
        elif price >= 0.00001:
            unit = 0.00000001
        
        # 특정 화폐는 무조건 1원 단위
        fixed_unit_currencies = ['ADA', 'ALGO', 'BLUR', 'CELO', 'ELF', 'EOS', 'GRS', 'GRT', 
                                'ICX', 'MANA', 'MINA', 'POL', 'SAND', 'SEI', 'STG', 'TRX']
        if currency in fixed_unit_currencies:
            unit = 1
        
        base_price = (price // unit) * unit

        # unit * relative_slippage 적용
        return base_price + (unit * relative_slippage)

    def limit_order(self, symbol: str, side: str, price: float, size: float):
        """Place limit order"""
        if self.is_sandbox:
            # Mock response 생성
            order_id = f'mock_{int(time.time())}'
            mock_response = {
                'uuid': order_id,
                'side': 'bid' if side.lower() == 'buy' else 'ask',
                'ord_type': 'limit',
                'price': str(price),
                'state': 'wait',
                'market': symbol if symbol.startswith('KRW-') else f'KRW-{symbol.split("-")[0]}',
                'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S+09:00'),
                'volume': str(size),
                'remaining_volume': str(size),
                'executed_volume': '0',
                'trades_count': 0
            }
            # 주문 저장
            self._last_limit_order = mock_response
            if hasattr(self, '_mock_limit_orders'):
                self._mock_limit_orders.append(mock_response)
            return self._format_order(mock_response)

        # Convert BTC-KRW to KRW-BTC format
        if '-' in symbol:
            base, quote = symbol.split('-')
            if quote == 'KRW':
                symbol = f'KRW-{base}'
                
        # 가격을 업비트 규칙에 맞게 조정
        currency = symbol.split('-')[1]  # KRW-BTC에서 BTC 추출
        price = self._adjust_price(currency, price)
        
        if side.lower() == 'buy':
            response = self.calls.buy_limit_order(symbol, price, size)
        else:
            response = self.calls.sell_limit_order(symbol, price, size)
            
        return self._format_order(response)

    def cancel_order(self, symbol: str, order_id: str):
        """Cancel an order"""
        if self.is_sandbox:
            # 취소된 주문 ID 저장
            if not hasattr(self, '_cancelled_orders'):
                self._cancelled_orders = set()
            self._cancelled_orders.add(order_id)
            return {'order_id': order_id}

        response = self.calls.cancel_order(order_id)
        return {'order_id': order_id}

    def get_open_orders(self, symbol=None):
        """Get open orders"""
        if self.is_sandbox:
            # Mock response for sandbox mode
            mock_orders = []
            # Add mock limit orders that are still open
            if hasattr(self, '_mock_limit_orders'):
                mock_orders.extend(self._mock_limit_orders)
            else:
                # 최근 limit order를 저장
                self._mock_limit_orders = []
                if hasattr(self, '_last_limit_order'):
                    mock_orders.append(self._last_limit_order)
            return [self._format_order(order) for order in mock_orders]
        try:
            if symbol:
                # Convert BTC-KRW to KRW-BTC format
                if '-' in symbol:
                    base, quote = symbol.split('-')
                    if quote == 'KRW':
                        symbol = f'KRW-{base}'
                # 미체결 주문 조회
                orders = self.calls.get_order(symbol, state='wait')
            else:
                orders = self.calls.get_order(state='wait')
                
            if not isinstance(orders, list):
                return []
                
            return [self._format_order(order) for order in orders]
        except Exception as e:
            raise APIException(f"Failed to get open orders: {str(e)}")

    def get_order(self, symbol: str, order_id: str):
        """Get a specific order"""
        if self.is_sandbox:
            # 저장된 side 정보 사용
            original_side = getattr(self, '_order_sides', {}).get(order_id, 'buy')  # 기본값은 buy
            is_cancelled = hasattr(self, '_cancelled_orders') and order_id in self._cancelled_orders
            
            mock_response = {
                'uuid': order_id,
                'side': 'bid' if original_side == 'buy' else 'ask',
                'ord_type': 'market',
                'state': 'cancel' if is_cancelled else 'done',
                'market': symbol if symbol.startswith('KRW-') else f'KRW-{symbol.split("-")[0]}',
                'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S+09:00'),
                'volume': '0.001',
                'remaining_volume': '0.001' if is_cancelled else '0',
                'executed_volume': '0' if is_cancelled else '0.001',
                'trades_count': 0 if is_cancelled else 1
            }
            return self._format_order(mock_response)
        
        order = self.calls.get_individual_order(order_id)
        if order is None:
            raise APIException(f"Failed to get order {order_id}")
        return self._format_order(order)

    def get_fees(self, symbol):
        """Get trading fees"""
        return {
            'maker_fee_rate': 0.0005,  # 0.05%
            'taker_fee_rate': 0.0005   # 0.05%
        }

    def get_product_history(self, symbol: str, epoch_start: float, epoch_stop: float, resolution: int):
        """Get historical price data"""
        # Convert BTC-KRW to KRW-BTC format
        if '-' in symbol:
            base, quote = symbol.split('-')
            if quote == 'KRW':
                symbol = f'KRW-{base}'
        
        # Convert resolution to minutes
        resolution = resolution // 60
        
        # Get candles with error handling
        df = pyupbit.get_ohlcv(symbol, 
                              interval=f'{resolution}min',
                              to=datetime.fromtimestamp(epoch_stop),
                              count=min(200, int((epoch_stop - epoch_start) // (resolution * 60))))
        
        if df is None:
            raise APIException(f"Failed to get historical data for {symbol}")
            
        # Format dataframe
        df = df.reset_index()
        df = df.rename(columns={
            'index': 'time',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Convert timestamp to epoch
        df['time'] = df['time'].astype(int) // 10**9
        
        return df[['time', 'low', 'high', 'open', 'close', 'volume']]

    def _format_order(self, order):
        """Helper to format order responses"""
        if order is None:
            raise APIException("Order failed: No response from server")
        
        if 'error' in order:
            raise APIException(f"Order failed: {order['error']['message']}")
        
        print(f"DEBUG: Formatting order: {order}")
        # Convert bid/ask to buy/sell
        side_map = {
            'bid': 'buy',
            'ask': 'sell'
        }
        # Convert Upbit order types to blankly order types
        type_map = {
            'price': 'market',  # 시장가 매수
            'market': 'market', # 시장가 매도
            'limit': 'limit'    # 지정가 주문
        }
        # Get executed size from trades if available
        executed_size = 0.0
        if 'trades' in order and order['trades']:
            for trade in order['trades']:
                executed_size += float(trade['volume'])
        else:
            executed_size = float(order.get('volume', order.get('executed_volume', 0)))

        formatted = {
            'id': order['uuid'],
            'symbol': order['market'].replace('KRW-', '') + '-KRW',
            'size': executed_size,
            'type': type_map.get(order['ord_type'].lower(), order['ord_type'].lower()),
            'side': side_map.get(order['side'].lower(), order['side'].lower()),
            'status': self._parse_status(order),
            'time_in_force': 'GTC',
            'created_at': order['created_at']
        }
        print(f"DEBUG: Formatted order: {formatted}")
        return formatted
        
    def _parse_status(self, order):
        """Convert Upbit status to blankly status"""
        status_map = {
            'wait': 'open',
            'done': 'filled', 
            'cancel': 'canceled'
        }
        status = order['state']
        print(f"DEBUG: Original Upbit status: {status}")
        # 거래 내역이 있으면 'filled' 상태로 간주
        if 'trades' in order and order['trades'] and float(order['executed_volume']) > 0:
            return 'filled'
        parsed_status = status_map.get(status, status)
        print(f"DEBUG: Parsed status: {parsed_status}")
        return parsed_status

    def get_ticker_feed(self, symbol: str):
        """
        Get ticker feed for a symbol
        Args:
            symbol: The asset to get price feeds for (e.g. 'BTC-KRW')
        """
        symbol = utils.to_exchange_symbol(symbol, 'upbit')
        return Tickers(symbol, stream="ticker")
        
    def get_orderbook_feed(self, symbol: str):
        """
        Get order book feed for a symbol
        Args:
            symbol: The asset to get orderbook data for (e.g. 'BTC-KRW')
        """
        symbol = utils.to_exchange_symbol(symbol, 'upbit')
        return Tickers(symbol, stream="orderbook") 

    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        # Convert BTC-KRW to KRW-BTC format
        if '-' in symbol:
            base, quote = symbol.split('-')
            if quote == 'KRW':
                symbol = f'KRW-{base}'
            
        # Sandbox 모드에서도 실제 가격 정보는 가져옴
        ticker = pyupbit.get_current_price(symbol)
        if ticker is None:
            raise APIException(f"Failed to get price for {symbol}")
        return float(ticker)

    def get_order_filter(self, symbol: str):
        """Get order filters for a symbol"""
        # Get current price to calculate minimum size
        current_price = self.get_price(symbol)
        min_size = 5000 / current_price  # 최소 주문 금액 5000원을 현재가로 나눠서 수량 계산
        
        return {
            'symbol': symbol,
            'base_min_size': min_size,  # 최소 주문 금액 5000원에 해당하는 수량
            'base_max_size': 10000000000,  # Maximum order size
            'base_increment': 0.00000001,  # Minimum size increment
            'price_increment': 1,  # Minimum price increment (1 KRW)
            'min_price': 1,  # Minimum price (1 KRW)
            'max_price': 1000000000  # Maximum price (1B KRW)
        }

    def stop_loss_order(self, symbol: str, price: float, size: float):
        """
        웹소켓으로 가격을 모니터링하다가 지정가 이하로 떨어지면 시장가 매도 주문을 실행
        
        Args:
            symbol (str): 거래 쌍 (예: BTC-KRW)
            price (float): 스탑 가격 
            size (float): 매도할 수량
        """
        if self.is_sandbox:
            return self._mock_stop_order('stop_loss', symbol, price, size)

        received_messages = []
        message_lock = threading.Lock()
        
        def message_handler(msg):
            with message_lock:
                current_price = float(msg.get('trade_price', 0))
                if current_price <= price:  # 현재가가 스탑 가격 이하로 떨어지면
                    try:
                        # 시장가 매도 주문 실행
                        order = self.market_order(symbol, 'sell', size)
                        received_messages.append(order)
                        ticker_feed.close()  # 주문 실행 후 웹소켓 연결 종료
                    except Exception as e:
                        print(f"Stop loss order failed: {e}")
                        ticker_feed.close()

        # 웹소켓 연결 및 모니터링 시작
        ticker_feed = self.get_ticker_feed(symbol)
        ticker_feed.append_callback(message_handler)
        ticker_feed.connect()

        # 주문이 체결될 때까지 대기
        start_time = time.time()
        while len(received_messages) == 0:
            if time.time() - start_time > 60:  # 1분 타임아웃
                ticker_feed.close()
                raise TimeoutError("Stop loss order timed out")
            time.sleep(0.1)

        return received_messages[0]

    def take_profit_order(self, symbol: str, price: float, size: float):
        """
        웹소켓으로 가격을 모니터링하다가 지정가 이상으로 올라가면 시장가 매도 주문을 실행
        
        Args:
            symbol (str): 거래 쌍 (예: BTC-KRW)
            price (float): 목표 가격
            size (float): 매도할 수량
        """
        if self.is_sandbox:
            return self._mock_stop_order('take_profit', symbol, price, size)

        received_messages = []
        message_lock = threading.Lock()
        
        def message_handler(msg):
            with message_lock:
                current_price = float(msg.get('trade_price', 0))
                if current_price >= price:  # 현재가가 목표 가격 이상으로 올라가면
                    try:
                        # 시장가 매도 주문 실행
                        order = self.market_order(symbol, 'sell', size)
                        received_messages.append(order)
                        ticker_feed.close()  # 주문 실행 후 웹소켓 연결 종료
                    except Exception as e:
                        print(f"Take profit order failed: {e}")
                        ticker_feed.close()

        # 웹소켓 연결 및 모니터링 시작
        ticker_feed = self.get_ticker_feed(symbol)
        ticker_feed.append_callback(message_handler)
        ticker_feed.connect()

        # 주문이 체결될 때까지 대기
        start_time = time.time()
        while len(received_messages) == 0:
            if time.time() - start_time > 60:  # 1분 타임아웃
                ticker_feed.close()
                raise TimeoutError("Take profit order timed out")
            time.sleep(0.1)

        return received_messages[0]

    def _mock_stop_order(self, order_type: str, symbol: str, price: float, size: float):
        """Sandbox mode에서 stop loss/take profit 주문 모의 응답 생성"""
        # 실제 현재가를 가져와서 모의 거래에 사용
        current_price = self.get_price(symbol)
        
        # 특수 케이스: 즉시 체결
        if (order_type == 'stop_loss' and price == 0) or \
           (order_type == 'take_profit' and price == 1):
            state = 'done'
        # 일반 케이스: 가격 비교
        elif order_type == 'stop_loss' and current_price <= price:
            state = 'done'  # 현재가가 스탑 가격 이하면 체결
        elif order_type == 'take_profit' and current_price >= price:
            state = 'done'  # 현재가가 목표가 이상이면 체결
        else:
            state = 'wait'  # 그 외에는 대기
        
        mock_response = {
            'uuid': f'mock_{order_type}_{int(time.time())}',
            'side': 'ask',
            'ord_type': 'market',
            'price': str(current_price * size),
            'state': state,
            'market': symbol if symbol.startswith('KRW-') else f'KRW-{symbol.split("-")[0]}',
            'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S+09:00'),
            'volume': str(size),
            'remaining_volume': str(size) if state == 'wait' else '0',
            'executed_volume': '0' if state == 'wait' else str(size),
            'trades_count': 0 if state == 'wait' else 1,
            'stop_price': str(price)
        }
        
        # 체결된 경우 거래 내역 추가
        if state == 'done':
            mock_response['trades'] = [{
                'market': mock_response['market'],
                'uuid': f'mock_trade_{int(time.time())}',
                'price': str(current_price),
                'volume': str(size),
                'funds': str(current_price * size),
                'side': 'ask'
            }]
        
        return self._format_order(mock_response)

    def margin_order(self, symbol: str, side: str, amount: float):
        """Margin orders are not supported on Upbit"""
        raise NotImplementedError("Margin trading is not supported on Upbit")

    def futures_order(self, symbol: str, side: str, amount: float):
        """Futures orders are not supported on Upbit"""
        raise NotImplementedError("Futures trading is not supported on Upbit") 