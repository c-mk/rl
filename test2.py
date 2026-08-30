import schwab
from schwab.auth import easy_client
import httpx2

c = easy_client(
    api_key='RNz4Esx0WFVNTlGiP57GpJGO4XqLoAONtqC3dnrrG1MaXgJQ',
    app_secret = '5SNiAxva0QjrwnGbcN0LEsQn33TBBj4OmjmPTutx3yw09UPtDDCebtBSAAnn3JzS',
    callback_url='https://127.0.0.1:8182',
    token_path='/tmp/token.json')

resp = c.get_price_history_every_day('AAPL')
assert resp.status_code == httpx2.codes.OK

esp = c.get_quote('SPY')
print(resp.status_code)   # 200 — just the status
data = resp.json()        # <-- this is where the actual quote data is
print(data)

history = resp.json()
resp = c.get_price_history_every_day('SPY')
history = (resp.json())
print(history)

