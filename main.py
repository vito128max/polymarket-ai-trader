import os
import json
import re
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


# ============================================================
# CONFIG
# ============================================================

GAMMA_URL = os.getenv(
    "GAMMA_URL",
    "https://gamma-api.polymarket.com/markets"
)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

INITIAL_BALANCE = float(
    os.getenv("PAPER_BALANCE", "1000")
)

MAX_POSITION_PCT = float(
    os.getenv("MAX_POSITION_PCT", "0.05")
)

MIN_EDGE = float(
    os.getenv("MIN_EDGE", "0.10")
)

MIN_CONFIDENCE = float(
    os.getenv("MIN_CONFIDENCE", "0.65")
)

SCAN_INTERVAL = int(
    os.getenv("SCAN_INTERVAL", "60")
)

MAX_MARKETS = int(
    os.getenv("MAX_MARKETS", "25")
)

AI_MARKETS_PER_SCAN = int(
    os.getenv("AI_MARKETS_PER_SCAN", "5")
)

ALLOWED_CHAT_ID = os.getenv(
    "TELEGRAM_ALLOWED_CHAT_ID",
    ""
).strip()


# ============================================================
# STORAGE
# ============================================================

DATA_DIR = Path(
    os.getenv("DATA_DIR", "/app/data")
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STATE_FILE = DATA_DIR / "state.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def default_state():
    return {
        "cash": INITIAL_BALANCE,
        "positions": [],
        "history": [],
        "auto": False
    }


def load_state():

    if not STATE_FILE.exists():
        return default_state()

    try:
        return json.loads(
            STATE_FILE.read_text()
        )

    except Exception:
        return default_state()


STATE = load_state()

STATE.setdefault(
    "cash",
    INITIAL_BALANCE
)

STATE.setdefault(
    "positions",
    []
)

STATE.setdefault(
    "history",
    []
)

STATE.setdefault(
    "auto",
    False
)


def save_state():

    temp_file = STATE_FILE.with_suffix(
        ".tmp"
    )

    temp_file.write_text(
        json.dumps(
            STATE,
            ensure_ascii=False,
            indent=2
        )
    )

    temp_file.replace(
        STATE_FILE
    )


# ============================================================
# CLIENTS
# ============================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# TELEGRAM ACCESS CONTROL
# ============================================================

def allowed(update: Update):

    if not ALLOWED_CHAT_ID:
        return True

    return (
        str(update.effective_chat.id)
        == ALLOWED_CHAT_ID
    )


async def reply(
    update: Update,
    text: str
):

    await update.effective_message.reply_text(
        text
    )


# ============================================================
# POLYMARKET MARKET DATA
# ============================================================

async def fetch_markets():

    params = {
        "closed": "false",
        "limit": 100
    }

    async with httpx.AsyncClient(
        timeout=20
    ) as client:

        response = await client.get(
            GAMMA_URL,
            params=params
        )

        response.raise_for_status()

        data = response.json()

    markets = (
        data
        if isinstance(data, list)
        else data.get("markets", [])
    )

    result = []

    for market in markets:

        try:

            prices = market.get(
                "outcomePrices"
            )

            outcomes = market.get(
                "outcomes"
            )

            if isinstance(
                prices,
                str
            ):
                prices = json.loads(
                    prices
                )

            if isinstance(
                outcomes,
                str
            ):
                outcomes = json.loads(
                    outcomes
                )

            if (
                not isinstance(prices, list)
                or len(prices) < 2
            ):
                continue

            if (
                not isinstance(outcomes, list)
                or len(outcomes) < 2
            ):
                outcomes = [
                    "Yes",
                    "No"
                ]

            yes_price = float(
                prices[0]
            )

            no_price = float(
                prices[1]
            )

            if not (
                0.01
                < yes_price
                < 0.99
            ):
                continue

            result.append({

                "id": str(
                    market.get("id", "")
                ),

                "slug": market.get(
                    "slug",
                    ""
                ),

                "question": market.get(
                    "question",
                    ""
                ),

                "description": (
                    market.get(
                        "description"
                    )
                    or ""
                )[:1500],

                "yes_price": yes_price,

                "no_price": no_price,

                "liquidity": float(
                    market.get(
                        "liquidityNum"
                    )
                    or market.get(
                        "liquidity"
                    )
                    or 0
                ),

                "volume24h": float(
                    market.get(
                        "volume24hr"
                    )
                    or 0
                ),

                "endDate": market.get(
                    "endDate"
                )
            })

        except Exception:

            continue

    result.sort(
        key=lambda x: (
            x["volume24h"],
            x["liquidity"]
        ),
        reverse=True
    )

    return result[:MAX_MARKETS]


# ============================================================
# OPENAI ANALYSIS
# ============================================================

def ai_analyze(market):

    prompt = f"""
You are the analysis engine for a PAPER-TRADING
Polymarket bot.

IMPORTANT:
This is simulated trading only.
Do not claim certainty.
Do not give financial advice.

Analyze ONLY the supplied market data.

Estimate a fair probability for YES.

Return ONLY valid JSON with exactly:

{{
  "yes_probability": number,
  "confidence": number,
  "decision": "BUY_YES" | "BUY_NO" | "PASS",
  "reason": "short string"
}}

Rules:

- yes_probability must be between 0 and 1.
- confidence must be between 0 and 1.
- BUY_YES means YES appears materially underpriced.
- BUY_NO means NO appears materially underpriced.
- PASS means there is not enough edge.
- Do not invent facts.
- Do not use information that is not contained
  in the supplied market data.

Market question:
{market["question"]}

Description:
{market["description"]}

Current YES price:
{market["yes_price"]:.4f}

Current NO price:
{market["no_price"]:.4f}

Liquidity:
{market["liquidity"]:.2f}

24h volume:
{market["volume24h"]:.2f}

End date:
{market["endDate"]}
"""

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )

    text = (
        response.output_text
        or ""
    ).strip()

    match = re.search(
        r"\{.*\}",
        text,
        re.S
    )

    if not match:
        raise ValueError(
            "AI did not return JSON"
        )

    data = json.loads(
        match.group(0)
    )

    probability = min(
        1,
        max(
            0,
            float(
                data["yes_probability"]
            )
        )
    )

    confidence = min(
        1,
        max(
            0,
            float(
                data["confidence"]
            )
        )
    )

    decision = str(
        data["decision"]
    ).upper()

    if decision not in {
        "BUY_YES",
        "BUY_NO",
        "PASS"
    }:

        decision = "PASS"

    return {

        "yes_probability":
            probability,

        "confidence":
            confidence,

        "decision":
            decision,

        "reason":
            str(
                data.get(
                    "reason",
                    ""
                )
            )[:600]
    }


# ============================================================
# PORTFOLIO
# ============================================================

def equity(markets_by_id):

    position_value = 0.0

    for position in STATE["positions"]:

        market = markets_by_id.get(
            str(
                position["market_id"]
            )
        )

        price = position[
            "entry_price"
        ]

        if market:

            if position["side"] == "YES":

                price = market[
                    "yes_price"
                ]

            else:

                price = market[
                    "no_price"
                ]

        position_value += (
            position["shares"]
            * price
        )

    return (
        STATE["cash"]
        + position_value
    )


# ============================================================
# PAPER TRADING
# ============================================================

def paper_buy(
    market,
    side,
    ai
):

    if side == "YES":

        price = market[
            "yes_price"
        ]

    else:

        price = market[
            "no_price"
        ]

    max_cost = (
        STATE["cash"]
        * MAX_POSITION_PCT
    )

    if max_cost < 1:

        return None

    shares = (
        max_cost
        / price
    )

    cost = (
        shares
        * price
    )

    STATE["cash"] -= cost

    if side == "YES":

        edge = (
            ai["yes_probability"]
            - market["yes_price"]
        )

    else:

        edge = (
            (1 - ai["yes_probability"])
            - market["no_price"]
        )

    position = {

        "market_id":
            market["id"],

        "question":
            market["question"],

        "side":
            side,

        "entry_price":
            price,

        "shares":
            shares,

        "cost":
            cost,

        "ai_probability":
            ai["yes_probability"],

        "confidence":
            ai["confidence"],

        "edge":
            edge,

        "opened_at":
            now()
    }

    STATE["positions"].append(
        position
    )

    STATE["history"].append({

        "type":
            "BUY",

        **position

    })

    save_state()

    return position


# ============================================================
# FORMAT AI RESULT
# ============================================================

def format_analysis(
    market,
    ai
):

    edge_yes = (
        ai["yes_probability"]
        - market["yes_price"]
    )

    edge_no = (
        (1 - ai["yes_probability"])
        - market["no_price"]
    )

    return (
        f"🧠 {market['question'][:180]}\n\n"

        f"YES 市价: "
        f"{market['yes_price']:.3f}\n"

        f"AI YES 概率: "
        f"{ai['yes_probability']:.3f}\n\n"

        f"YES Edge: "
        f"{edge_yes:+.1%}\n"

        f"NO Edge: "
        f"{edge_no:+.1%}\n\n"

        f"置信度: "
        f"{ai['confidence']:.0%}\n"

        f"AI 决策: "
        f"{ai['decision']}\n\n"

        f"理由:\n"
        f"{ai['reason']}"
    )


# ============================================================
# SCANNER
# ============================================================

async def scan_once():

    markets = await fetch_markets()

    candidates = markets[
        :AI_MARKETS_PER_SCAN
    ]

    signals = []

    for market in candidates:

        try:

            ai = await asyncio.to_thread(
                ai_analyze,
                market
            )

            signals.append(
                (
                    market,
                    ai
                )
            )

        except Exception as error:

            signals.append(
                (
                    market,
                    {
                        "decision":
                            "PASS",

                        "reason":
                            f"AI error: {error}",

                        "yes_probability":
                            market[
                                "yes_price"
                            ],

                        "confidence":
                            0
                    }
                )
            )

    trades = []

    for market, ai in signals:

        if (
            ai["confidence"]
            < MIN_CONFIDENCE
        ):
            continue

        edge_yes = (
            ai["yes_probability"]
            - market["yes_price"]
        )

        edge_no = (
            (1 - ai["yes_probability"])
            - market["no_price"]
        )

        side = None
        edge = 0

        if (
            ai["decision"]
            == "BUY_YES"
            and edge_yes
            >= MIN_EDGE
        ):

            side = "YES"
            edge = edge_yes

        elif (
            ai["decision"]
            == "BUY_NO"
            and edge_no
            >= MIN_EDGE
        ):

            side = "NO"
            edge = edge_no

        if not side:
            continue

        existing = any(

            position[
                "market_id"
            ] == market["id"]

            and position[
                "side"
            ] == side

            for position
            in STATE["positions"]
        )

        if existing:
            continue

        position = paper_buy(
            market,
            side,
            ai
        )

        if position:

            trades.append(
                (
                    market,
                    ai,
                    position,
                    edge
                )
            )

    return (
        markets,
        signals,
        trades
    )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    await reply(
        update,

        "🤖 Polymarket AI Paper Trader\n\n"

        "模式：PAPER 模拟交易\n"
        "不会连接钱包\n"
        "不会发送真实订单\n\n"

        f"初始余额："
        f"${INITIAL_BALANCE:.2f}\n"

        f"单仓上限："
        f"{MAX_POSITION_PCT:.1%}\n"

        f"最小 Edge："
        f"{MIN_EDGE:.1%}\n"

        f"最低 AI 置信度："
        f"{MIN_CONFIDENCE:.0%}\n\n"

        "/status - 系统状态\n"
        "/scan - 扫描并模拟交易\n"
        "/analyze - AI 分析市场\n"
        "/positions - 查看持仓\n"
        "/pnl - 查看盈亏\n"
        "/auto - 开启自动扫描\n"
        "/stop - 停止自动扫描\n"
        "/id - 查看 Telegram Chat ID"
    )


async def id_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await reply(
        update,
        f"Chat ID: {update.effective_chat.id}"
    )


async def status_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    markets = await fetch_markets()

    markets_by_id = {
        market["id"]: market
        for market in markets
    }

    current_equity = equity(
        markets_by_id
    )

    pnl = (
        current_equity
        - INITIAL_BALANCE
    )

    await reply(
        update,

        "🟢 PAPER TRADING\n\n"

        f"现金: "
        f"${STATE['cash']:.2f}\n"

        f"持仓: "
        f"{len(STATE['positions'])}\n"

        f"当前权益: "
        f"${current_equity:.2f}\n"

        f"PnL: "
        f"${pnl:+.2f}\n"

        f"自动扫描: "
        f"{'ON' if STATE.get('auto') else 'OFF'}"
    )


async def positions_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    if not STATE["positions"]:

        await reply(
            update,
            "目前没有 PAPER 持仓。"
        )

        return

    markets = await fetch_markets()

    markets_by_id = {
        market["id"]: market
        for market in markets
    }

    lines = [
        "📊 PAPER 持仓"
    ]

    for position in STATE["positions"]:

        market = markets_by_id.get(
            position["market_id"]
        )

        if market:

            if position["side"] == "YES":

                current_price = market[
                    "yes_price"
                ]

            else:

                current_price = market[
                    "no_price"
                ]

        else:

            current_price = position[
                "entry_price"
            ]

        value = (
            position["shares"]
            * current_price
        )

        pnl = (
            value
            - position["cost"]
        )

        lines.append(

            f"\n"
            f"{position['side']} | "
            f"{position['question'][:100]}\n"

            f"入场: "
            f"{position['entry_price']:.3f}\n"

            f"当前: "
            f"{current_price:.3f}\n"

            f"价值: "
            f"${value:.2f}\n"

            f"PnL: "
            f"${pnl:+.2f}"
        )

    await reply(
        update,
        "\n".join(lines)[:3900]
    )


async def pnl_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    markets = await fetch_markets()

    markets_by_id = {
        market["id"]: market
        for market in markets
    }

    current_equity = equity(
        markets_by_id
    )

    pnl = (
        current_equity
        - INITIAL_BALANCE
    )

    await reply(

        update,

        "💰 PAPER PnL\n\n"

        f"初始资金: "
        f"${INITIAL_BALANCE:.2f}\n"

        f"当前权益: "
        f"${current_equity:.2f}\n"

        f"总 PnL: "
        f"${pnl:+.2f}\n"

        f"现金: "
        f"${STATE['cash']:.2f}\n"

        f"持仓数量: "
        f"{len(STATE['positions'])}"
    )


async def analyze_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    await reply(
        update,
        "🔎 正在读取 Polymarket 市场并调用 GPT……"
    )

    markets = await fetch_markets()

    output = []

    for market in markets[
        :AI_MARKETS_PER_SCAN
    ]:

        try:

            ai = await asyncio.to_thread(
                ai_analyze,
                market
            )

            output.append(
                format_analysis(
                    market,
                    ai
                )
            )

        except Exception as error:

            output.append(
                "❌ "
                + market["question"][:120]
                + "\n"
                + str(error)
            )

    if not output:

        output = [
            "没有找到可分析的市场。"
        ]

    await reply(
        update,
        "\n\n".join(output)[:3900]
    )


async def scan_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    await reply(
        update,
        "⚙️ 正在扫描……"
    )

    markets, signals, trades = (
        await scan_once()
    )

    lines = [

        f"扫描市场: "
        f"{len(markets)}",

        f"AI 分析: "
        f"{len(signals)}"
    ]

    if trades:

        lines.append(
            "\n🟢 本轮 PAPER 交易:"
        )

        for (
            market,
            ai,
            position,
            edge
        ) in trades:

            lines.append(

                f"\n"
                f"BUY {position['side']}\n"

                f"Edge: "
                f"{edge:.1%}\n"

                f"成本: "
                f"${position['cost']:.2f}\n"

                f"{market['question'][:150]}"
            )

    else:

        lines.append(
            "\n本轮没有达到风控条件的交易。"
        )

    await reply(
        update,
        "\n".join(lines)[:3900]
    )


async def auto_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    STATE["auto"] = True

    save_state()

    await reply(
        update,

        f"🟢 自动扫描已开启\n"
        f"扫描间隔：{SCAN_INTERVAL} 秒"
    )


async def stop_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not allowed(update):
        return

    STATE["auto"] = False

    save_state()

    await reply(
        update,
        "⛔ 自动扫描已停止。"
    )


# ============================================================
# BACKGROUND AUTO TRADING
# ============================================================

async def auto_loop(
    application
):

    while True:

        try:

            if STATE.get("auto"):

                (
                    markets,
                    signals,
                    trades
                ) = await scan_once()

                if (
                    ALLOWED_CHAT_ID
                    and trades
                ):

                    message = (
                        "🤖 自动扫描产生 "
                        "PAPER 交易：\n"
                    )

                    for (
                        market,
                        ai,
                        position,
                        edge
                    ) in trades:

                        message += (
                            f"\n"
                            f"BUY {position['side']} "
                            f"Edge={edge:.1%}\n"
                            f"{market['question'][:120]}\n"
                        )

                    await application.bot.send_message(
                        chat_id=int(
                            ALLOWED_CHAT_ID
                        ),
                        text=message[:3900]
                    )

        except Exception as error:

            print(
                "AUTO LOOP ERROR:",
                repr(error)
            )

        await asyncio.sleep(
            SCAN_INTERVAL
        )


# ============================================================
# TELEGRAM STARTUP
# ============================================================

async def post_init(
    application
):

    application.bot_data[
        "auto_task"
    ] = asyncio.create_task(
        auto_loop(
            application
        )
    )

    await application.bot.set_my_commands([

        (
            "start",
            "启动/帮助"
        ),

        (
            "status",
            "系统状态"
        ),

        (
            "scan",
            "扫描并模拟交易"
        ),

        (
            "analyze",
            "AI分析"
        ),

        (
            "positions",
            "查看持仓"
        ),

        (
            "pnl",
            "查看盈亏"
        ),

        (
            "auto",
            "开启自动扫描"
        ),

        (
            "stop",
            "停止自动扫描"
        ),

        (
            "id",
            "查看Chat ID"
        )
    ])


async def post_shutdown(
    application
):

    task = application.bot_data.get(
        "auto_task"
    )

    if task:

        task.cancel()

        try:

            await task

        except asyncio.CancelledError:

            pass


# ============================================================
# MAIN
# ============================================================

def main():

    application = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "id",
            id_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "status",
            status_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "positions",
            positions_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "pnl",
            pnl_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "analyze",
            analyze_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "scan",
            scan_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "auto",
            auto_cmd
        )
    )

    application.add_handler(
        CommandHandler(
            "stop",
            stop_cmd
        )
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":

    main()
