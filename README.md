# Exness MT5 — AI Trading Bot (Gemma‑3 API)

A streamlined, event-driven trading assistant that connects MetaTrader 5 (MT5) with
Google AI Studio's Gemma-3 models. The bot collects live market data, engineers
features, requests a decision payload from the LLM, enforces strict local
risk-management rules, and forwards executable orders to the broker.

> **Important**: This project is for educational purposes. Trade on a demo account
> before considering live markets.

## Highlights

- **Event-driven loop** — decisions are requested only on new bars and when market
  state changes, protecting the Gemma-3 free-tier quota.
- **Guardrails before every order** — spread, session filters, risk/reward, cooldown,
  ATR-based clamps, daily loss kill-switch, and one-position policy are enforced locally.
- **LLM constrained to JSON** — the language model only returns
  `{action, entry, sl, tp, confidence}`; order management remains fully deterministic
  inside the bot.
- **Persistent telemetry** — every decision is logged to SQLite and mirrored to CSV
  for later analysis or model fine-tuning.

## Quick Start

1. Install the official MT5 desktop terminal and log into your **Exness** account.
2. Locate your terminal path (e.g. `C:\\Program Files\\MetaTrader 5\\terminal64.exe`).
3. Duplicate `.env.example` to `.env` and populate the variables.
4. Review `config.yaml` to tune symbols, risk, and schedule.
5. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Launch the bot:

   ```bash
   python -m src.main
   ```

### Platform notes

- Windows is recommended as it runs the native MT5 terminal. Linux users can rely on
  Wine but must provide the path to the terminal executable.
- The project does **not** execute without the MT5 desktop terminal running and
  logged in. Keep the terminal active while the bot is operating.

## Configuration Overview

Key options from `config.yaml`:

- `symbol` — trading instrument (default `XAUUSDm`).
- `timeframe` — bar timeframe used for feature construction and decisions.
- `action_interval_seconds` — minimum interval between LLM requests.
- `risk` — spread limits, maximum risk per trade, and stop-loss bounds.
- `confirmations` — multi-timeframe trend/volatility gates (ATR bounds, M5/M15 filters).
- `clamps` — ATR-driven price guard that restricts LLM entries/SL/TP around market price.
- `exits` — breakeven trigger and ATR trailing distance for live position management.
- `safety` — daily maximum loss percentage (equity-based kill switch).
- `sessions` — broker-time trading windows (supports overnight ranges).
- `cooldowns.after_order_seconds` — cooldown between filled trades.
- `ai` — Gemma model, endpoint, timeout, and retry policy.

Refer to the inline comments in `config.yaml` for the full list of knobs.

## Disclaimer

Trading involves significant risk. Historical performance does not guarantee
future results. Always validate behaviour on a demo account before deploying to
live capital.
