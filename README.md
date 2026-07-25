# LLM Trading Backtest

Backtest de uma estratégia em que **um LLM decide a direção** (long / short /
fechar / manter) a cada candle, e o **código controla o risco**. Roda sobre
dados históricos reais da Binance, com taxas reais, e compara o resultado
contra buy & hold.

> Premissa do projeto: uma estratégia que não bate o buy & hold não presta.
> O backtest existe para testar essa hipótese com honestidade, não para
> confirmá-la.

## Como funciona

```
ccxt ──► candles históricos (Binance)
   │
   ▼
indicadores          RSI, EMAs, MACD, ATR calculados por candle
   │
   ▼
contexto ──► LLM     saída JSON estruturada: LONG / SHORT / CLOSE / HOLD
   │
   ▼
simulador            1 posição por vez, sem alavancagem, taxa real da Binance
   │
   ▼
comparação           estratégia  ×  buy & hold
```

**Separação de responsabilidades:** o LLM só opina sobre direção. Tamanho de
posição, limite de exposição e taxas são do código — o modelo nunca controla
risco.

## Multi-provedor

Funciona com **Anthropic (Claude)** ou qualquer endpoint **compatível com
OpenAI** — DeepSeek, OpenAI, Groq, OpenRouter, Ollama local. Basta trocar
`BT_PROVIDER` e `BT_BASE_URL` no `.env`, útil para comparar custo e qualidade
de decisão entre modelos.

Para provedores compatíveis-OpenAI, o schema de saída vai no prompt; no modo
Anthropic, usa saída estruturada nativa.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
python backtest_llm.py
```

Configurável por `.env`: símbolo, timeframe, número de candles, modelo e
provedor.

## Detalhe de implementação

O cliente HTTP é configurado para confiar na **loja de certificados do
Windows** — o `httpx` usado pelos SDKs ignora o certificado de proxies
corporativos por padrão, o que quebra as chamadas em rede empresarial.

## Stack

Python · `ccxt` · `pandas` · Anthropic API / OpenAI-compatible APIs · `httpx`
