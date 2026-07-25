"""
Backtest de estrategia "LLM decide a direcao" em dados historicos.

Para cada candle: calcula indicadores (RSI, EMAs, MACD, ATR), monta o contexto,
pergunta ao Claude (saida JSON estruturada) se deve ficar LONG / SHORT / fechar
/ manter, e simula o resultado com taxa real da Binance. O LLM decide direcao;
o CODIGO controla risco (1 posicao por vez, sem alavancagem).

Compara com buy & hold — uma estrategia que nao bate o buy & hold nao presta.

MODOS (BT_PROVIDER no .env):
  - offline   -> baseline rule-based (EMA+MACD), SEM chave. Roda de imediato.
  - anthropic -> Claude decide (ANTHROPIC_API_KEY; console.anthropic.com)
  - openai    -> compativel: DeepSeek/OpenAI/Groq/OpenRouter (DEEPSEEK_API_KEY etc.)
  Sem chave, cai automaticamente pra offline — o projeto sempre roda.

USO:
    python backtest_llm.py
    (configuravel por .env: BT_SYMBOL, BT_TIMEFRAME, BT_CANDLES, BT_MODEL, ...)
"""
from __future__ import annotations

import csv
import json
import os
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path

import ccxt
import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---- config (tudo ajustavel por .env) --------------------------------------
SYMBOL     = os.environ.get("BT_SYMBOL", "BTC/USDT")
TIMEFRAME  = os.environ.get("BT_TIMEFRAME", "4h")
CANDLES    = int(os.environ.get("BT_CANDLES", "150"))   # quantos candles decidir
WARMUP     = int(os.environ.get("BT_WARMUP", "50"))     # candles p/ os indicadores
FEE_PCT    = float(os.environ.get("BT_FEE_PCT", "0.1"))  # taxa Binance taker/lado
START_CASH = float(os.environ.get("BT_START_CASH", "1000"))
ALLOW_SHORT = os.environ.get("BT_ALLOW_SHORT", "1") == "1"

# provedor do LLM: "anthropic" (Claude) ou "openai" (compativel: DeepSeek, OpenAI,
# Groq, OpenRouter, Ollama local...). Para DeepSeek: BT_PROVIDER=openai.
PROVIDER   = os.environ.get("BT_PROVIDER", "anthropic").lower()
# base_url so vale para o modo "openai-compativel". Default: DeepSeek.
BASE_URL   = os.environ.get("BT_BASE_URL", "https://api.deepseek.com")
_DEFAULT_MODEL = "deepseek-chat" if PROVIDER == "openai" else "claude-haiku-4-5"
MODEL      = os.environ.get("BT_MODEL", _DEFAULT_MODEL)

CSV_FILE = Path(__file__).parent / "backtest_decisions.csv"

SYSTEM = (
    "Voce e um trader sistematico. A cada candle recebe indicadores tecnicos e a "
    "posicao atual, e decide a acao. Responda SO com a acao pedida, sem texto extra.\n"
    "Acoes: 'long' (ficar comprado), 'short' (ficar vendido), 'close' (zerar a "
    "posicao), 'hold' (manter como esta). Pense no risco/retorno; nao opere a toa — "
    "'hold' costuma ser a resposta certa quando nao ha sinal claro."
)

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["long", "short", "close", "hold"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["action", "confidence", "reason"],
    "additionalProperties": False,
}

# para o modo openai-compativel (DeepSeek etc.), o schema vai no prompt
SCHEMA_HINT = (
    '\nResponda APENAS com um objeto JSON valido, sem markdown, neste formato exato:\n'
    '{"action": "long"|"short"|"close"|"hold", "confidence": <numero 0..1>, '
    '"reason": "<curto>"}'
)


def _windows_ssl_context() -> ssl.SSLContext:
    """Confia nos certs da loja do Windows (o httpx do anthropic ignora a loja do SO)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    loaded = False
    for store in ("ROOT", "CA"):
        try:
            for cert_der, enc, _ in ssl.enum_certificates(store):
                if enc == "x509_asn":
                    try:
                        ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(cert_der))
                        loaded = True
                    except ssl.SSLError:
                        pass
        except (AttributeError, OSError):
            pass
    if not loaded:
        ctx.load_default_certs()
    return ctx


# ---- dados + indicadores ----------------------------------------------------

def fetch_data(n: int) -> pd.DataFrame:
    ex = ccxt.binance()
    raw = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=n)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    # ATR(14)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def context_for(df: pd.DataFrame, i: int, position) -> str:
    """Monta o texto enviado ao LLM para a decisao no candle i."""
    row = df.iloc[i]
    recent = df["close"].iloc[i - 5:i + 1].round(2).tolist()
    pos_txt = "flat (sem posicao)"
    if position["dir"]:
        ret = ((row["close"] / position["entry"] - 1) * 100
               * (1 if position["dir"] == "long" else -1))
        pos_txt = f"{position['dir']} desde {position['entry']:.2f} (PnL aberto {ret:+.1f}%)"
    return (
        f"Par: {SYMBOL}  Timeframe: {TIMEFRAME}\n"
        f"Preco atual: {row['close']:.2f}\n"
        f"Ultimos 6 closes: {recent}\n"
        f"EMA20: {row['ema20']:.2f}  EMA50: {row['ema50']:.2f}  "
        f"(tendencia: {'alta' if row['ema20'] > row['ema50'] else 'baixa'})\n"
        f"RSI(14): {row['rsi']:.1f}\n"
        f"MACD: {row['macd']:.2f}  sinal: {row['macd_signal']:.2f}  "
        f"({'cruzou pra cima' if row['macd'] > row['macd_signal'] else 'cruzou pra baixo'})\n"
        f"ATR(14): {row['atr']:.2f} ({row['atr'] / row['close'] * 100:.1f}% do preco)\n"
        f"Posicao atual: {pos_txt}\n"
        + ("" if ALLOW_SHORT else "Obs: SHORT nao permitido neste mercado; use long/close/hold.\n")
        + "Qual a acao?"
    )


def _parse_decision(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):  # alguns modelos embrulham em markdown
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    return json.loads(text)


def decide_anthropic(client, ctx: str) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": ctx}],
        output_config={"format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return _parse_decision(text)


def decide_openai(client, ctx: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM + SCHEMA_HINT},
            {"role": "user", "content": ctx},
        ],
        response_format={"type": "json_object"},
    )
    return _parse_decision(resp.choices[0].message.content)


def decide_offline(row) -> dict:
    """Baseline rule-based (sem LLM): tendencia por EMA + confirmacao do MACD.
    Serve como benchmark honesto e deixa o projeto rodavel sem chave de API."""
    up = row["ema20"] > row["ema50"] and row["macd"] > row["macd_signal"]
    down = row["ema20"] < row["ema50"] and row["macd"] < row["macd_signal"]
    if up:
        return {"action": "long", "confidence": 0.6, "reason": "EMA20>EMA50 e MACD pra cima"}
    if down:
        return {"action": "short" if ALLOW_SHORT else "close", "confidence": 0.6,
                "reason": "EMA20<EMA50 e MACD pra baixo"}
    return {"action": "hold", "confidence": 0.3, "reason": "sem sinal claro"}


# ---- simulacao --------------------------------------------------------------

def main():
    ssl_ctx = _windows_ssl_context()
    client = None
    offline = (PROVIDER == "offline")

    if not offline and PROVIDER == "openai":
        key = (os.environ.get("BT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
        if not key:
            print("[aviso] sem chave de LLM (DEEPSEEK_API_KEY/OPENAI_API_KEY) — "
                  "rodando em modo OFFLINE (baseline rule-based).")
            offline = True
        else:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=BASE_URL,
                            http_client=httpx.Client(verify=ssl_ctx, timeout=60))
            decide = decide_openai
    elif not offline:  # anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[aviso] sem ANTHROPIC_API_KEY — rodando em modo OFFLINE (baseline rule-based).")
            offline = True
        else:
            from anthropic import Anthropic, DefaultHttpxClient
            client = Anthropic(http_client=DefaultHttpxClient(verify=ssl_ctx))
            decide = decide_anthropic

    total = CANDLES + WARMUP
    print(f"Baixando {total} candles de {SYMBOL} {TIMEFRAME}...")
    df = add_indicators(fetch_data(total)).reset_index(drop=True)
    start_i = len(df) - CANDLES

    cash = START_CASH
    position = {"dir": None, "entry": 0.0}
    trades = []
    fee = FEE_PCT / 100

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["dt", "price", "rsi", "action", "confidence",
                                "position", "equity", "reason"])

    def close_position(price):
        nonlocal cash
        ret = (price / position["entry"] - 1) * (1 if position["dir"] == "long" else -1)
        cash *= (1 + ret) * (1 - fee)   # realiza + taxa de saida
        trades.append(ret)
        position["dir"] = None

    def open_position(direction, price):
        nonlocal cash
        cash *= (1 - fee)               # taxa de entrada
        position["dir"] = direction
        position["entry"] = price

    label = "offline: baseline" if offline else f"{PROVIDER}: {MODEL}"
    print(f"Backtestando {CANDLES} candles  [{label}]...\n")
    for n, i in enumerate(range(start_i, len(df)), 1):
        row = df.iloc[i]
        price = row["close"]
        if offline:
            d = decide_offline(row)
        else:
            try:
                d = decide(client, context_for(df, i, position))
            except Exception as e:
                code = getattr(e, "status_code", None)
                if code in (401, 402, 403):
                    print(f"\n[aviso] API rejeitou (HTTP {code}: sem saldo/chave invalida). "
                          "Trocando pra modo OFFLINE (baseline) pelo resto do backtest.\n")
                    offline = True
                    d = decide_offline(row)
                else:
                    print(f"  [warn] candle {n}: {str(e)[:80]} — hold")
                    d = {"action": "hold", "confidence": 0, "reason": "erro"}

        action = d["action"]
        if action == "short" and not ALLOW_SHORT:
            action = "close"

        # aplica a decisao
        if action == "close" and position["dir"]:
            close_position(price)
        elif action == "long":
            if position["dir"] == "short":
                close_position(price)
            if not position["dir"]:
                open_position("long", price)
        elif action == "short" and ALLOW_SHORT:
            if position["dir"] == "long":
                close_position(price)
            if not position["dir"]:
                open_position("short", price)
        # hold: nada

        # equity marcada a mercado (inclui posicao aberta)
        if position["dir"]:
            mtm = (price / position["entry"] - 1) * (1 if position["dir"] == "long" else -1)
            equity = cash * (1 + mtm)
        else:
            equity = cash

        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                f"{row['dt']}", f"{price:.2f}", f"{row['rsi']:.1f}", action,
                f"{d['confidence']:.2f}", position["dir"] or "flat",
                f"{equity:.2f}", d["reason"][:80],
            ])
        if n % 10 == 0 or n == CANDLES:
            print(f"  {n}/{CANDLES}  {row['dt']:%m-%d %H:%M}  {action:5} "
                  f"pos={position['dir'] or 'flat':5} equity={equity:.2f}")
        if not offline:
            time.sleep(0.3)   # respeita rate-limit do LLM; offline nao precisa

    # fecha posicao no fim p/ contabilizar
    if position["dir"]:
        close_position(df.iloc[-1]["close"])
    final_equity = cash

    # ---- resultado + benchmark ----
    bh_ret = (df.iloc[-1]["close"] / df.iloc[start_i]["close"] - 1) * 100
    strat_ret = (final_equity / START_CASH - 1) * 100
    wins = sum(1 for t in trades if t > 0)
    print("\n" + "=" * 56)
    print(f"RESULTADO ({SYMBOL} {TIMEFRAME}, {CANDLES} candles)")
    mode_txt = "baseline" if offline else f"{PROVIDER}/{MODEL}"
    print(f"  Estrategia ({mode_txt}): {strat_ret:+.1f}%   (equity {START_CASH:.0f} -> {final_equity:.0f})")
    print(f"  Buy & hold     : {bh_ret:+.1f}%")
    print(f"  Trades         : {len(trades)}  (win {wins}/{len(trades)} = "
          f"{wins / len(trades) * 100:.0f}%)" if trades else "  Trades: 0")
    print(f"  Veredito       : {'BATEU o buy&hold' if strat_ret > bh_ret else 'NAO bateu o buy&hold'}")
    print("=" * 56)
    print(f"\nDetalhe candle a candle em: {CSV_FILE.name}")


if __name__ == "__main__":
    main()
