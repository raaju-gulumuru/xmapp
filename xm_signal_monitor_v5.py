"""
XM Signal Monitor v5 - manual trading signals only (no order execution).

Connect to an open XM MetaTrader 5 terminal.
This application only monitors, calculates and displays signals;
it never places or modifies orders.

Spread rules:
- BTCUSD: allow spread up to and including 50.0 MT5 price units.
- GOLD: retain the original M15 ATR-based spread filter (12% of ATR).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# -------------------- settings --------------------

SYMBOL_CANDIDATES = {
    "GOLD": ["GOLD", "XAUUSD", "GOLD.", "XAUUSD.", "XAUUSD#"],
    "BTCUSD": ["BTCUSD", "BTCUSD.", "BTCUSD#", "BTC/USD"],
}

TF = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}

# BTC spread is measured as tick.ask - tick.bid, in the broker's quote
# price units. Change this value if your XM terminal reports a different scale.
BTC_MAX_SPREAD = 50.0

# GOLD keeps the original ATR-relative spread rule.
GOLD_MAX_SPREAD_ATR_MULTIPLIER = 0.12


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat(
        [
            df.high - df.low,
            (df.high - prev).abs(),
            (df.low - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, down = df.high.diff(), -df.low.diff()
    plus = pd.Series(
        np.where((up > down) & (up > 0), up, 0), index=df.index
    ).ewm(alpha=1 / n, adjust=False).mean()
    minus = pd.Series(
        np.where((down > up) & (down > 0), down, 0), index=df.index
    ).ewm(alpha=1 / n, adjust=False).mean()

    a = atr(df, n)
    pdi = 100 * plus / a.replace(0, np.nan)
    mdi = 100 * minus / a.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def macd_hist(close: pd.Series) -> pd.Series:
    macd = ema(close, 12) - ema(close, 26)
    return macd - ema(macd, 9)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = ema(x.close, 20)
    x["ema50"] = ema(x.close, 50)
    x["ema200"] = ema(x.close, 200)
    x["rsi"] = rsi(x.close)
    x["atr"] = atr(x)
    x["adx"] = adx(x)
    x["macd_h"] = macd_hist(x.close)
    x["bb_mid"] = x.close.rolling(20).mean()
    x["bb_std"] = x.close.rolling(20).std()
    x["bb_up"] = x.bb_mid + 2 * x.bb_std
    x["bb_dn"] = x.bb_mid - 2 * x.bb_std
    x["prior_high20"] = x.high.shift(1).rolling(20).max()
    x["prior_low20"] = x.low.shift(1).rolling(20).min()
    return x


def trend_label(df: pd.DataFrame) -> str:
    if len(df) < 210:
        return "INSUFFICIENT DATA"

    x = enrich(df).iloc[-1]
    if x.ema50 > x.ema200 and x.close > x.ema50:
        return "BULLISH"
    if x.ema50 < x.ema200 and x.close < x.ema50:
        return "BEARISH"
    return "NEUTRAL"


def market_regime(m15: pd.DataFrame) -> str:
    x = enrich(m15).iloc[-1]
    if pd.isna(x.adx) or pd.isna(x.atr):
        return "UNKNOWN"
    if x.adx >= 25:
        return "TRENDING"
    if x.adx < 18:
        return "RANGING"
    return "TRANSITION"


def score_setup(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    m5: pd.DataFrame,
    spread: float,
    risk_pct: float,
    daily_locked: bool,
    min_rr: float,
    symbol_label: str,
    btc_max_spread: float,
) -> dict:
    """Closed-candle rule engine. Signal is deliberately conservative."""

    M, H, Q, E = (enrich(d).iloc[-1] for d in (m15, h1, h4, m5))
    trend_h1, trend_h4 = trend_label(h1), trend_label(h4)
    regime = market_regime(m15)

    score_buy, score_sell = 0, 0
    reasons_buy, reasons_sell, blockers = [], [], []

    # Higher-timeframe alignment is mandatory.
    if trend_h1 == "BULLISH":
        score_buy += 30
        reasons_buy.append("H1 trend bullish")
    if trend_h1 == "BEARISH":
        score_sell += 30
        reasons_sell.append("H1 trend bearish")
    if trend_h4 == "BULLISH":
        score_buy += 15
        reasons_buy.append("H4 trend bullish")
    if trend_h4 == "BEARISH":
        score_sell += 15
        reasons_sell.append("H4 trend bearish")

    # M15 setup: EMA pullback/reclaim or close through previous 20-bar boundary.
    pullback_buy = (
        M.low <= M.ema20 and M.close > M.ema20 and M.close > M.open
    )
    pullback_sell = (
        M.high >= M.ema20 and M.close < M.ema20 and M.close < M.open
    )
    breakout_buy = M.close > M.prior_high20 and M.close > M.open
    breakout_sell = M.close < M.prior_low20 and M.close < M.open

    if pullback_buy:
        score_buy += 20
        reasons_buy.append("M15 bullish EMA20 reclaim")
    if pullback_sell:
        score_sell += 20
        reasons_sell.append("M15 bearish EMA20 rejection")
    if breakout_buy:
        score_buy += 20
        reasons_buy.append("M15 20-candle breakout")
    if breakout_sell:
        score_sell += 20
        reasons_sell.append("M15 20-candle breakdown")

    # Momentum alignment.
    if M.rsi >= 52 and M.macd_h > 0:
        score_buy += 15
        reasons_buy.append("M15 RSI/MACD bullish")
    if M.rsi <= 48 and M.macd_h < 0:
        score_sell += 15
        reasons_sell.append("M15 RSI/MACD bearish")

    if M.adx >= 18:
        if M.close > M.ema50:
            score_buy += 10
            reasons_buy.append("M15 above EMA50 with ADX")
        if M.close < M.ema50:
            score_sell += 10
            reasons_sell.append("M15 below EMA50 with ADX")

    if E.close > E.ema20 and E.macd_h > 0:
        score_buy += 10
        reasons_buy.append("M5 entry momentum agrees")
    if E.close < E.ema20 and E.macd_h < 0:
        score_sell += 10
        reasons_sell.append("M5 entry momentum agrees")

    # Hard safety blockers.
    if daily_locked:
        blockers.append("Daily loss lock is active")
    if pd.isna(M.atr) or M.atr <= 0:
        blockers.append("ATR unavailable")
    if trend_h1 in ("NEUTRAL", "INSUFFICIENT DATA"):
        blockers.append("H1 trend not established")
    if trend_h4 in ("NEUTRAL", "INSUFFICIENT DATA"):
        blockers.append("H4 trend not established")
    if regime == "RANGING":
        blockers.append("Ranging market: trend setups disabled")

    # Instrument-specific spread filter.
    if spread <= 0 or not np.isfinite(spread):
        blockers.append("Spread unavailable")
    elif symbol_label == "BTCUSD":
        if spread > btc_max_spread:
            blockers.append(
                f"BTC spread too high: {spread:.2f} "
                f"(maximum allowed: {btc_max_spread:.2f})"
            )
    elif symbol_label == "GOLD":
        if not pd.isna(M.atr) and spread > M.atr * GOLD_MAX_SPREAD_ATR_MULTIPLIER:
            blockers.append("Spread too high versus M15 ATR")

    direction = "WAIT"
    reasons = []

    if score_buy >= 75 and score_buy > score_sell:
        direction, reasons = "BUY", reasons_buy
    elif score_sell >= 75 and score_sell > score_buy:
        direction, reasons = "SELL", reasons_sell

    if blockers:
        direction = "WAIT"

    return {
        "signal": direction,
        "score_buy": score_buy,
        "score_sell": score_sell,
        "reasons": reasons,
        "blockers": blockers,
        "trend_h1": trend_h1,
        "trend_h4": trend_h4,
        "regime": regime,
        "rsi": float(M.rsi),
        "adx": float(M.adx),
        "atr": float(M.atr),
        "m15": M,
        "m5": E,
        "min_rr": min_rr,
    }


def price_levels(
    signal: str,
    entry: float,
    atr_value: float,
    candle_low: float,
    candle_high: float,
    tick_size: float,
    min_rr: float,
    tp2_rr: float = 2.0,
) -> Optional[dict]:
    values = [entry, atr_value, candle_low, candle_high]
    if (
        signal not in ("BUY", "SELL")
        or not all(np.isfinite(v) for v in values)
        or atr_value <= 0
    ):
        return None

    # Swing + volatility stop. Keep stop at least 1.2 ATR from entry.
    if signal == "BUY":
        raw_sl = min(candle_low, entry - 1.2 * atr_value)
        sl = raw_sl - max(tick_size, 0.05 * atr_value)
        risk = entry - sl
        tp1, tp2 = entry + risk, entry + tp2_rr * risk
    else:
        raw_sl = max(candle_high, entry + 1.2 * atr_value)
        sl = raw_sl + max(tick_size, 0.05 * atr_value)
        risk = sl - entry
        tp1, tp2 = entry - risk, entry - tp2_rr * risk

    if risk <= 0 or (abs(tp2 - entry) / risk) < min_rr:
        return None

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_dist": risk,
    }


def mt5_connect():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None, (
            "MetaTrader5 Python package missing. "
            "Install MetaTrader5 on Windows Python."
        )

    if not mt5.initialize():
        return None, (
            f"MT5 initialize failed: {mt5.last_error()}. "
            "Open the XM MT5 terminal and log in."
        )

    return mt5, None


def resolve_symbol(mt5, label: str) -> Optional[str]:
    all_symbols = mt5.symbols_get() or []
    names = {s.name.upper(): s.name for s in all_symbols}

    for candidate in SYMBOL_CANDIDATES[label]:
        if candidate.upper() in names:
            name = names[candidate.upper()]
            mt5.symbol_select(name, True)
            return name

    if label == "GOLD":
        matches = [
            s.name for s in all_symbols
            if "XAU" in s.name.upper() or s.name.upper().startswith("GOLD")
        ]
    else:
        matches = [
            s.name for s in all_symbols
            if "BTC" in s.name.upper() and "USD" in s.name.upper()
        ]

    if len(matches) == 1:
        mt5.symbol_select(matches[0], True)
        return matches[0]

    return None


def fetch_rates(mt5, symbol: str, timeframe, bars: int = 450) -> Optional[pd.DataFrame]:
    # Start at shift 1: use completed candles only.
    raw = mt5.copy_rates_from_pos(symbol, timeframe, 1, bars)
    if raw is None or len(raw) < 220:
        return None

    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.time, unit="s", utc=True)
    return df[["open", "high", "low", "close", "tick_volume"]].copy()


def position_size(
    mt5,
    symbol: str,
    signal: str,
    entry: float,
    sl: float,
    balance: float,
    risk_pct: float,
) -> Optional[float]:
    if signal not in ("BUY", "SELL") or entry == sl or balance <= 0:
        return None

    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    order_type = (
        mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    )
    profit = mt5.order_calc_profit(order_type, symbol, 1.0, entry, sl)
    if profit is None:
        return None

    loss_per_lot = abs(float(profit))
    if loss_per_lot <= 0:
        return None

    risk_cash = balance * risk_pct / 100
    raw_lots = risk_cash / loss_per_lot

    # Normalize down, never up to broker minimum if that would violate risk cap.
    step = info.volume_step or 0.01
    lots = math.floor(raw_lots / step) * step
    if lots < info.volume_min:
        return None

    lots = min(lots, info.volume_max)
    return round(lots, 8)


def telegram_alert(text: str, token: str, chat_id: str) -> str:
    if not token.strip() or not chat_id.strip():
        return "Telegram not configured"

    try:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=8,
        )
        resp.raise_for_status()
        return "Telegram alert sent"
    except Exception as e:
        return f"Telegram alert failed: {e}"


def main():
    st.set_page_config(page_title="XM Signal Monitor v5", layout="wide")

    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1000, key="xm_auto_refresh")
    except ImportError:
        st.caption("Auto-refresh package unavailable; use Refresh now.")

    st.title("XM Signal Monitor v5")
    st.caption(
        "Manual signals only • No trade execution • "
        "Test on demo first • Signals are not guaranteed"
    )

    with st.sidebar:
        st.header("Risk settings")
        risk_pct = st.number_input(
            "Risk per trade (%)",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
        min_rr = st.selectbox(
            "Minimum reward/risk", [1.5, 2.0], index=0
        )
        daily_loss_limit = st.number_input(
            "Daily loss lock (%)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.25,
        )

        st.subheader("Spread limits")
        btc_max_spread = st.number_input(
            "BTCUSD maximum spread",
            min_value=1.0,
            max_value=1000.0,
            value=BTC_MAX_SPREAD,
            step=1.0,
            help=(
                "Spread is calculated as MT5 ask minus bid. "
                "A spread equal to this limit is allowed; a higher spread is blocked."
            ),
        )
        st.caption(
            "GOLD retains the ATR-based spread filter. "
            "BTCUSD uses the fixed limit above."
        )

        telegram_token = st.text_input(
            "Telegram bot token", type="password"
        )
        telegram_chat = st.text_input("Telegram chat ID")
        st.caption("Tokens are not stored by this app. Avoid sharing them.")

        demo_mode = st.toggle(
            "Demo mode (synthetic data)", value=False
        )
        if st.button("Refresh now", use_container_width=True):
            st.rerun()

    st.info(
        "Signal decisions use completed candles. "
        "Dashboard refresh is separate from candle confirmation."
    )

    if demo_mode:
        st.warning(
            "Demo mode uses synthetic prices. "
            "Do not interpret these signals as live market advice."
        )
        st.metric("Data source", "Synthetic demo")
        st.write(
            "Demo signal engine is intended for UI/testing; "
            "use MT5 data for market analysis."
        )
        return

    mt5, error = mt5_connect()
    if error:
        st.error(error)
        st.markdown(
            "**Setup:** Open XM MT5 → log in → install the matching "
            "Python package → rerun this app."
        )
        st.code(
            "py -m pip install MetaTrader5 streamlit pandas numpy requests "
            "streamlit-autorefresh\n"
            "py -m streamlit run xm_signal_monitor_v5.py",
            language="powershell",
        )
        return

    try:
        account = mt5.account_info()
        if account is None:
            st.error("Could not read MT5 account details.")
            return

        balance = float(account.balance)
        equity = float(account.equity)

        st.caption(
            f"Connected to MT5 · Account currency: {account.currency} · "
            f"Balance: {balance:,.2f} · Equity: {equity:,.2f}"
        )

        daily_pnl = st.number_input(
            "Today's realized P/L (account currency; enter from MT5 history)",
            value=0.0,
            help=(
                "Manual input for this initial release. "
                "Daily lock is based on this amount divided by balance."
            ),
        )
        daily_locked = daily_pnl <= -(balance * daily_loss_limit / 100)

        if daily_locked:
            st.error(
                f"Daily loss limit reached ({daily_loss_limit:.2f}%). "
                "Signals are blocked until next day."
            )

        cards = st.columns(2)

        for col, label in zip(cards, ["GOLD", "BTCUSD"]):
            with col:
                st.subheader(label)

                symbol = resolve_symbol(mt5, label)
                if not symbol:
                    st.warning(
                        f"{label}: symbol not found. Check Market Watch naming."
                    )
                    continue

                tick = mt5.symbol_info_tick(symbol)
                info = mt5.symbol_info(symbol)
                if tick is None or info is None:
                    st.error("No quote or symbol metadata.")
                    continue

                spread = float(tick.ask - tick.bid)

                m5 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_M5)
                m15 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_M15)
                h1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_H1)
                h4 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_H4)

                if any(d is None for d in [m5, m15, h1, h4]):
                    st.warning(
                        "Insufficient history. Load M5/M15/H1/H4 "
                        "candles in MT5 and retry."
                    )
                    continue

                setup = score_setup(
                    m15,
                    h1,
                    h4,
                    m5,
                    spread,
                    risk_pct,
                    daily_locked,
                    min_rr,
                    label,
                    float(btc_max_spread),
                )

                signal = setup["signal"]
                entry = (
                    float(tick.ask if signal == "BUY" else tick.bid)
                    if signal in ("BUY", "SELL")
                    else None
                )

                levels = None
                lots = None

                if signal in ("BUY", "SELL"):
                    levels = price_levels(
                        signal,
                        entry,
                        setup["atr"],
                        float(setup["m15"].low),
                        float(setup["m15"].high),
                        float(info.trade_tick_size or info.point),
                        min_rr,
                    )
                    if levels:
                        lots = position_size(
                            mt5,
                            symbol,
                            signal,
                            levels["entry"],
                            levels["sl"],
                            balance,
                            risk_pct,
                        )
                    if levels is None or lots is None:
                        signal = "WAIT"

                if signal == "BUY":
                    st.success(
                        f"**{signal} SETUP** · Score {setup['score_buy']}/100"
                    )
                elif signal == "SELL":
                    st.error(
                        f"**{signal} SETUP** · Score {setup['score_sell']}/100"
                    )
                else:
                    st.warning(
                        f"**WAIT** · Buy score {setup['score_buy']} / "
                        f"Sell score {setup['score_sell']}"
                    )

                st.write(
                    f"**MT5 symbol:** `{symbol}` · "
                    f"**Bid:** {tick.bid:.{info.digits}f} · "
                    f"**Ask:** {tick.ask:.{info.digits}f} · "
                    f"**Spread:** {spread:.{info.digits}f}"
                )

                if label == "BTCUSD":
                    st.caption(
                        f"BTC spread limit: {btc_max_spread:.2f} "
                        f"(equal to limit is allowed)"
                    )

                st.write(
                    f"**H1:** {setup['trend_h1']} · "
                    f"**H4:** {setup['trend_h4']} · "
                    f"**Regime:** {setup['regime']}"
                )
                st.write(
                    f"**M15 RSI:** {setup['rsi']:.1f} · "
                    f"**ADX:** {setup['adx']:.1f} · "
                    f"**ATR:** {setup['atr']:.{info.digits}f}"
                )

                if levels and signal in ("BUY", "SELL"):
                    metrics = st.columns(4)
                    for metric, (name, value) in zip(
                        metrics,
                        [
                            ("Entry", levels["entry"]),
                            ("SL", levels["sl"]),
                            ("TP1", levels["tp1"]),
                            ("TP2", levels["tp2"]),
                        ],
                    ):
                        metric.metric(name, f"{value:.{info.digits}f}")

                    st.metric(
                        "Suggested lots (risk-based)",
                        lots if lots else "Rejected: broker min lot exceeds risk",
                    )
                    st.caption(
                        f"Calculated risk budget: "
                        f"{balance * risk_pct / 100:.2f} {account.currency} "
                        f"({risk_pct:.2f}%). Verify all values and spread before entry."
                    )

                with st.expander(
                    "Signal reasons / blockers", expanded=True
                ):
                    st.write(
                        "**Reasons:**",
                        setup["reasons"] or "No setup confirmation",
                    )
                    st.write(
                        "**Blockers:**",
                        setup["blockers"] or "None",
                    )

                if signal in ("BUY", "SELL") and st.button(
                    f"Send {label} signal to Telegram",
                    key=f"tg_{label}",
                ):
                    msg = (
                        f"XM SIGNAL MONITOR\n"
                        f"{label} {signal}\n"
                        f"Entry {levels['entry']}\n"
                        f"SL {levels['sl']}\n"
                        f"TP1 {levels['tp1']}\n"
                        f"TP2 {levels['tp2']}\n"
                        f"Lots {lots}\n"
                        f"Score {setup['score_buy'] if signal == 'BUY' else setup['score_sell']}\n"
                        f"Signal-only; verify in MT5"
                    )
                    st.info(
                        telegram_alert(msg, telegram_token, telegram_chat)
                    )

                st.caption(
                    "Signal is informational. Check quote freshness, news, "
                    "broker spread, and candle close before acting."
                )
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
