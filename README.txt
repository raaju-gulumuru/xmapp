XM Signal Monitor v4 - manual signals only
========================================

WARNING
This app does not guarantee correct signals or profits. It is a technical-analysis
prototype. Test with XM demo account and historical/forward testing first. It never
places, modifies, or closes trades.


Requirements
-----------
Windows, Python 3.10+ (use the Python version compatible with your installed MT5)
XM MetaTrader 5 desktop terminal installed, open and logged in.


Setup
-----
1. Open XM MT5 and sign into a DEMO account. Keep terminal open.
2. Open PowerShell in this folder.
3. Install packages:
   py -m pip install -r requirements.txt
4. Launch:
   py -m streamlit run xm_signal_monitor_v4.py
5. Browser opens the dashboard. Check symbol names in MT5 Market Watch.


Notes
-----
- Uses closed candles for indicator signals; dashboard refreshes every 15 seconds.
- BUY/SELL labels are setup candidates, not recommendations or probabilities.
- Current daily P/L input is manual and is not automatically calculated from deal history.
- Telegram alerts are sent only when you click the button and configure bot token/chat ID.
- Lot sizing uses MT5 order_calc_profit. If broker minimum lot exceeds risk, sizing is
  rejected instead of forcing the minimum.
- Symbol matching is conservative; if multiple suffix candidates exist, check names.
- Demo mode is a UI placeholder and does not simulate realistic signal performance.
- This version does not provide historical backtesting, automatic signal CSV logging,
  automatic daily P/L calculation, or automatic order execution.


===========================================================================

py -3.12 -m venv .venv

.\.venv\Scripts\python.exe --version

.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install MetaTrader5

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m streamlit run xm_signal_monitor_v4.py