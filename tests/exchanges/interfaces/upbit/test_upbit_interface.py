"""
    Upbit interface unit tests
    Copyright (C) 2023 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import time
from pathlib import Path
import json
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import threading

import pytest

import blankly
from blankly.exchanges.interfaces.upbit.upbit_interface import UpbitInterface


@pytest.fixture
def upbit_interface():
    keys_file_path = Path("tests/config/keys.json").resolve()
    settings_file_path = Path("tests/config/settings.json").resolve()

    upbit = blankly.Upbit(keys_path=keys_file_path,
                         settings_path=settings_file_path,
                         portfolio_name='upbit test portfolio')

    return upbit.interface


def test_get_exchange(upbit_interface: UpbitInterface) -> None:
    assert upbit_interface.get_exchange_type() == 'upbit'


def test_get_products(upbit_interface: UpbitInterface) -> None:
    products = upbit_interface.get_products()
    assert len(products) > 0
    
    # Check if BTC market exists
    btc_market = None
    for product in products:
        if product['base_asset'] == 'BTC' and product['quote_asset'] == 'KRW':
            btc_market = product
            break
    
    assert btc_market is not None
    assert 'symbol' in btc_market
    assert 'base_min_size' in btc_market
    assert 'base_max_size' in btc_market
    assert 'base_increment' in btc_market


def test_get_account(upbit_interface: UpbitInterface) -> None:
    account = upbit_interface.get_account()
    assert isinstance(account, dict)
    
    # Test specific symbol query
    btc_account = upbit_interface.get_account('BTC-KRW')
    assert 'available' in btc_account
    assert 'hold' in btc_account
    assert isinstance(btc_account['available'], float)
    assert isinstance(btc_account['hold'], float)


def test_get_price(upbit_interface: UpbitInterface) -> None:
    price = upbit_interface.get_price('BTC-KRW')
    assert isinstance(price, float)
    assert price > 0


def test_market_order_flow(upbit_interface: UpbitInterface) -> None:
    """Test the full order flow: place market order -> check status -> get order"""
    # Skip if not in sandbox mode
    if not upbit_interface.is_sandbox:
        pytest.skip("Test requires sandbox mode")
        
    # Get initial account balance
    initial_balance = upbit_interface.get_account('KRW')['available']
    
    # Place small market buy order
    symbol = 'BTC-KRW'
    current_price = upbit_interface.get_price(symbol)
    print(f"DEBUG: Current price: {current_price}")
    order_amount = 5500  # 5500원 주문
    size = order_amount / current_price
    print(f"DEBUG: Order size: {size}")
    
    # Place buy order
    buy_order = upbit_interface.market_order(symbol, 'buy', size)
    print(f"DEBUG: Buy order response: {buy_order}")
    assert buy_order is not None
    
    # Wait for order to complete
    time.sleep(1)
    
    # Check order status
    order_status = upbit_interface.get_order(symbol, buy_order['id'])
    print(f"DEBUG: Order status response: {order_status}")
    assert order_status['status'] in ['filled', 'closed']
    assert order_status['side'] == 'buy'
    assert order_status['type'] == 'market'
    
    # Place sell order to return to initial state
    sell_size = float(order_status['size'])  # 체결 상태에서 실제 수량 가져오기
    sell_order = upbit_interface.market_order(symbol, 'sell', sell_size)
    assert sell_order is not None
    
    time.sleep(1)
    
    # Verify sell order
    sell_status = upbit_interface.get_order(symbol, sell_order['id'])
    assert sell_status['status'] in ['filled', 'closed']
    assert sell_status['side'] == 'sell'


def test_limit_order_flow(upbit_interface: UpbitInterface) -> None:
    """Test limit order placement and cancellation"""
    # Skip if not in sandbox mode
    if not upbit_interface.is_sandbox:
        pytest.skip("Test requires sandbox mode")
        
    symbol = 'BTC-KRW'
    
    # Get current price
    current_price = upbit_interface.get_price(symbol)
    
    # Place limit buy order 10 ticks below current price
    limit_price = upbit_interface._adjust_price('BTC', current_price, -10)  # 10 ticks below current price
    size = 5500 / limit_price
    
    limit_order = upbit_interface.limit_order(symbol, 'buy', limit_price, size)
    assert limit_order is not None
    
    # 주문 ID 저장
    order_id = limit_order['id']
    
    time.sleep(1)
    
    # Verify open order
    open_orders = upbit_interface.get_open_orders(symbol)
    assert len(open_orders) > 0
    assert any(order['id'] == order_id for order in open_orders)
    
    # Cancel order
    cancel_resp = upbit_interface.cancel_order(symbol, order_id)
    assert cancel_resp is not None
    
    time.sleep(1)
    
    # Verify order was cancelled
    cancelled_status = upbit_interface.get_order(symbol, order_id)
    assert cancelled_status['status'] in ['canceled', 'cancelled']


def test_get_fees(upbit_interface: UpbitInterface) -> None:
    fees = upbit_interface.get_fees('BTC-KRW')
    assert 'maker_fee_rate' in fees
    assert 'taker_fee_rate' in fees
    assert isinstance(fees['maker_fee_rate'], float)
    assert isinstance(fees['taker_fee_rate'], float)


def test_get_product_history(upbit_interface: UpbitInterface) -> None:
    """Test historical price data retrieval"""
    symbol = 'BTC-KRW'
    
    # Test with 1 minute resolution
    current_time = time.time()
    one_day_ago = current_time - (24 * 60 * 60)
    
    df = upbit_interface.get_product_history(symbol, one_day_ago, current_time, 60)
    
    assert len(df) > 0
    assert all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'volume'])
    assert df['time'].is_monotonic_increasing  # Verify timestamps are in order 


def test_websocket_feeds(upbit_interface: UpbitInterface) -> None:
    """Test websocket ticker and orderbook feeds"""
    symbol = 'BTC-KRW'

    # Test ticker feed
    ticker_feed = upbit_interface.get_ticker_feed(symbol)
    assert ticker_feed is not None

    # Mock message handler with threading lock
    received_messages = []
    message_lock = threading.Lock()
    
    def message_handler(msg: Dict[str, Any]):
        print(f"DEBUG: Received ticker message: {msg}")
        with message_lock:
            received_messages.append(msg)

    # Subscribe to feed
    ticker_feed.append_callback(message_handler)
    ticker_feed.connect()

    # Wait for some messages with timeout
    start_time = time.time()
    while len(received_messages) == 0:
        if time.time() - start_time > 5:  # 5초 타임아웃
            break
        time.sleep(0.1)

    # Cleanup
    ticker_feed.close()
    
    # Verify we received ticker data
    assert len(received_messages) > 0 


def test_stop_loss_sandbox(upbit_interface: UpbitInterface) -> None:
    """Test stop loss order in sandbox mode"""
    symbol = 'BTC-KRW'
    current_price = upbit_interface.get_price(symbol)
    
    # Set stop price slightly above current price to trigger immediately
    stop_price = current_price * 1.01
    size = 0.001  # Small test size
    
    # Place stop loss order
    order = upbit_interface.stop_loss_order(symbol, stop_price, size)
    
    # Verify order format
    assert order is not None
    assert order['symbol'] == symbol
    assert order['side'] == 'sell'
    assert order['type'] == 'market'
    assert float(order['size']) == size
    assert 'id' in order
    assert order['status'] in ['wait', 'filled']


def test_take_profit_sandbox(upbit_interface: UpbitInterface) -> None:
    """Test take profit order in sandbox mode"""
    symbol = 'BTC-KRW'
    current_price = upbit_interface.get_price(symbol)
    
    # Set take profit price slightly below current price to trigger immediately
    take_profit_price = current_price * 0.99
    size = 0.001  # Small test size
    
    # Place take profit order
    order = upbit_interface.take_profit_order(symbol, take_profit_price, size)
    
    # Verify order format
    assert order is not None
    assert order['symbol'] == symbol
    assert order['side'] == 'sell'
    assert order['type'] == 'market'
    assert float(order['size']) == size
    assert 'id' in order
    assert order['status'] in ['wait', 'filled']


# @pytest.mark.skip(reason="This test requires real price movement")
def test_stop_loss_live(upbit_interface: UpbitInterface) -> None:
    """Test stop loss order with real price monitoring"""
    symbol = 'BTC-KRW'
    current_price = upbit_interface.get_price(symbol)
    
    # Set stop price very close to current price (0.01% below)
    stop_price = current_price * 0.9999  # 0.01% below current price
    size = 0.001  # Small test size
    
    try:
        # Place stop loss order and wait for small price movement
        order = upbit_interface.stop_loss_order(symbol, stop_price, size)
        
        # Verify order execution
        assert order is not None
        assert order['symbol'] == symbol
        assert order['side'] == 'sell'
        assert order['type'] == 'market'
        assert float(order['size']) == size
        assert order['status'] in ['open', 'filled']  # Allow both states
        
    except TimeoutError:
        pytest.skip("Price didn't reach stop loss level within timeout")


# @pytest.mark.skip(reason="This test requires real price movement")
def test_take_profit_live(upbit_interface: UpbitInterface) -> None:
    """Test take profit order with real price monitoring"""
    symbol = 'BTC-KRW'
    current_price = upbit_interface.get_price(symbol)
    
    # Set take profit price very close to current price (0.01% above)
    take_profit_price = current_price * 1.0001  # 0.01% above current price
    size = 0.001  # Small test size
    
    try:
        # Place take profit order and wait for small price movement
        order = upbit_interface.take_profit_order(symbol, take_profit_price, size)
        
        # Verify order execution
        assert order is not None
        assert order['symbol'] == symbol
        assert order['side'] == 'sell'
        assert order['type'] == 'market'
        assert float(order['size']) == size
        assert order['status'] in ['open', 'filled']  # Allow both states
        
    except TimeoutError:
        pytest.skip("Price didn't reach take profit level within timeout")


def test_stop_loss(upbit_interface: UpbitInterface) -> None:
    """Test stop loss order with immediate execution by setting price to 0"""
    # Skip if not in sandbox mode
    if not upbit_interface.is_sandbox:
        pytest.skip("Test requires sandbox mode")
        
    symbol = 'BTC-KRW'
    current_price = upbit_interface.get_price(symbol)
    
    # Calculate minimum size for 5500 KRW (5000원 + 여유 500원)
    size = 5500 / current_price
    
    # Set stop price to 0 to trigger immediately
    stop_price = 0
    
    try:
        # Place stop loss order - should execute immediately
        order = upbit_interface.stop_loss_order(symbol, stop_price, size)
        
        # Verify immediate execution
        assert order is not None
        assert order['symbol'] == symbol
        assert order['side'] == 'sell'
        assert order['type'] == 'market'
        assert float(order['size']) >= (5000 / current_price)  # Verify minimum size
        assert order['status'] == 'filled'  # Should be filled immediately
        
    except TimeoutError:
        pytest.fail("Stop loss order should have executed immediately")


@patch('pyupbit.get_current_price')
def test_take_profit(mock_get_price, upbit_interface: UpbitInterface) -> None:
    """Test take profit order with immediate execution by setting very low price"""
    # Mock price response
    mock_get_price.return_value = 143673000.0
    
    # Skip if not in sandbox mode
    if not upbit_interface.is_sandbox:
        pytest.skip("Test requires sandbox mode")
        
    symbol = 'BTC-KRW'
    current_price = upbit_interface.get_price(symbol)
    
    # Calculate minimum size for 5500 KRW (5000원 + 여유 500원)
    size = 5500 / current_price
    
    # Set take profit price very low to trigger immediately
    take_profit_price = 1  # 1원으로 설정하여 즉시 체결되도록
    
    try:
        # Place take profit order - should execute immediately
        order = upbit_interface.take_profit_order(symbol, take_profit_price, size)
        
        # Verify immediate execution
        assert order is not None
        assert order['symbol'] == symbol
        assert order['side'] == 'sell'
        assert order['type'] == 'market'
        assert float(order['size']) >= (5000 / current_price)  # Verify minimum size
        assert order['status'] == 'filled'  # Should be filled immediately
        
    except TimeoutError:
        pytest.fail("Take profit order should have executed immediately") 