# LLM Trading Backtest

Backtest of a strategy where **an LLM decides the direction** (long / short /
close / hold) on each candle, and the **code controls the risk**. It runs on real
Binance historical data, with real fees, and compares the result against buy &
hold.

> Project premise: a strategy that doesn't beat buy & hold is worthless. The
> backtest exists to test that hypothesis honestly, not to confirm it.

## How it works

```
ccxt ──► historical candles (Binance)
   │
   ▼
indicators           RSI, EMAs, MACD, ATR computed per candle
   │
   ▼
context ──► LLM      structured JSON output: LONG / SHORT / CLOSE / HOLD
   │
   ▼
simulator            1 position at a time, no leverage, real Binance fee
   │
   ▼
comparison           strategy  ×  buy & hold
```

**Separation of concerns:** the LLM only opines on direction. Position size,
exposure limits and fees belong to the code — the model never controls risk.

## Modes

Set `BT_PROVIDER` in `.env`:

- **`offline`** — rule-based baseline (EMA + MACD), **no API key needed**. Runs
  immediately and serves as an honest benchmark.
- **`anthropic`** — Claude decides (`ANTHROPIC_API_KEY`).
- **`openai`** — any OpenAI-compatible endpoint: DeepSeek, OpenAI, Groq,
  OpenRouter, local Ollama (`DEEPSEEK_API_KEY`, etc.).

With no key configured it falls back to `offline` automatically — the project
always runs. For OpenAI-compatible providers the output schema goes in the prompt;
in Anthropic mode it uses native structured output.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
python backtest_llm.py          # runs offline out of the box
```

Configurable via `.env`: symbol, timeframe, number of candles, model and provider.

## Implementation detail

The HTTP client is configured to trust the **Windows certificate store** — the
`httpx` used by the SDKs ignores corporate proxy certificates by default, which
breaks calls on enterprise networks.

## Stack

Python · `ccxt` · `pandas` · Anthropic API / OpenAI-compatible APIs · `httpx`
