# -*- coding: utf-8 -*-
file_path = r"C:\Users\ACER\QuantDinger\backend_api_python\app\services\mt5_trading\client.py"
content = open(file_path, 'r', encoding='utf-8').read()

old = """            # Determine filling mode based on symbol properties
            # Different brokers support different filling modes
            print(f"DEBUG: symbol_info.filling_mode raw = {symbol_info.filling_mode}", flush=True)

            filling_mode = mt5.ORDER_FILLING_IOC  # Default
            if symbol_info.filling_mode & mt5.ORDER_FILLING_IOC:
                filling_mode = mt5.ORDER_FILLING_IOC
            elif symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
                filling_mode = mt5.ORDER_FILLING_FOK
            elif symbol_info.filling_mode & mt5.ORDER_FILLING_RETURN:
                filling_mode = mt5.ORDER_FILLING_RETURN
            print(f"DEBUG: chosen filling_mode = {filling_mode}", flush=True)

            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_float,  # Use validated and rounded volume
                "type": order_type,
                "price": price,
                "deviation": deviation,
                "magic": self.config.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            # Send order
            result = mt5.order_send(request)
            print(f"DEBUG: full result = {result}", flush=True)

            if result is None:
                error = mt5.last_error()
                return OrderResult(
                    success=False,
                    message=f"Order send failed: {error}"
                )

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return OrderResult(
                    success=False,
                    order_id=result.order if hasattr(result, 'order') else 0,
                    status=str(result.retcode),
                    message=f"Order rejected: {result.comment}",
                    raw=result._asdict() if hasattr(result, '_asdict') else {}
                )"""

new = """            # Determine filling mode: try all supported modes in order until one works.
            # Some brokers (esp. shared MetaQuotes-Demo) report a filling_mode bitmask
            # that does not match what the trade server actually accepts.
            print(f"DEBUG: symbol_info.filling_mode raw = {symbol_info.filling_mode}", flush=True)

            filling_modes_to_try = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
            result = None
            for fm in filling_modes_to_try:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume_float,
                    "type": order_type,
                    "price": price,
                    "deviation": deviation,
                    "magic": self.config.magic_number,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": fm,
                }
                result = mt5.order_send(request)
                rc = result.retcode if result is not None else None
                cm = result.comment if result is not None else None
                print(f"DEBUG: tried filling_mode={fm} -> retcode={rc} comment={cm}", flush=True)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    break

            if result is None:
                error = mt5.last_error()
                return OrderResult(
                    success=False,
                    message=f"Order send failed: {error}"
                )

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return OrderResult(
                    success=False,
                    order_id=result.order if hasattr(result, 'order') else 0,
                    status=str(result.retcode),
                    message=f"Order rejected: {result.comment}",
                    raw=result._asdict() if hasattr(result, '_asdict') else {}
                )"""

count = content.count(old)
print('matches found:', count)

if count > 0:
    content = content.replace(old, new)
    open(file_path, 'w', encoding='utf-8').write(content)
    print('done')
else:
    print('NO MATCH - need manual inspection')
