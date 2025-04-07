from dotenv import load_dotenv
from openai import OpenAI
import os
import inspect
from pydantic import TypeAdapter
import requests
import yfinance as yf
import json


def get_symbol(company: str) -> str:
    """
    Retrieve the stock symbol for a specified company using the Yahoo Finance API.
    :param company: The name of the company for which to retrieve the stock symbol, e.g., 'Nvidia'.
    :output: The stock symbol for the specified company.
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company, "country": "United States"}
    user_agents = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
    res = requests.get(
        url=url,
        params=params,
        headers=user_agents)

    try:
        data = res.json()
        symbol = data['quotes'][0]['symbol']
        return symbol
    except (KeyError, IndexError) as e:
        return ""


def get_stock_price(symbol: str):
    """
    Retrieve the most recent stock price data for a specified company using the Yahoo Finance API via the yfinance Python library.
    :param symbol: The stock symbol for which to retrieve data, e.g., 'NVDA' for Nvidia.
    :output: A dictionary containing the most recent stock price data.
    """
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1d", interval="1m")
    latest = hist.iloc[-1]
    return {
        "timestamp": str(latest.name),
        "open": latest["Open"],
        "high": latest["High"],
        "low": latest["Low"],
        "close": latest["Close"],
        "volume": latest["Volume"]
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": inspect.getdoc(get_symbol),
            "parameters": TypeAdapter(get_symbol).json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": inspect.getdoc(get_stock_price),
            "parameters": TypeAdapter(get_stock_price).json_schema(),
        },
    }
]

FUNCTION_MAP = {
    "get_symbol": get_symbol,
    "get_stock_price": get_stock_price
}


load_dotenv()
# Đọc từ file .env cùng thư mục, nhưng đừng commit nha!
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


def get_completion(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        # Để temparature=0 để kết quả ổn định sau nhiều lần chạy
        temperature=0
    )
    return response


# Bắt đầu làm bài tập từ line này!

SYSTEM_PROMPT ="""
You are a helpful customer support assistant. Use the supplied tools to assist the user. Follow these rules:
    1. Always format prices with currency symbols (e.g., $150.50, ¥500)
    2. The response will be two part: brief stock price and rap sentence.
    3. First part is brief stock price and focused on the financial data, with format:
    - Highlight significant price changes
    - Include the currency and market status
    - Format large numbers with appropriate separators (e.g., 1,234,567)
    - Use the format: "Symbol: Price (Change%) | Market: Status"
    4. Second part is a rap sentence to advise user:
    - Use a fun and engaging tone
    - An advise user for investment.
    - Use slang and informal language
    - Use a rap-like rhythm and rhyme scheme
    - Use ad-libs and sound effects to enhance the rap
    4. If there's an error, explain it simply and suggest alternatives
    6. Answer the question in English or Vietnamese base on user question language.
"""

isExit = False
while not isExit:
    question = input("Enter your question (type \"exit\" to end the conversation): ")
    if question.lower() == "exit":
        isExit = True
        print("Goodbye! Exiting the program.")
        break
    elif question == "":
        continue

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},   
        {"role": "user", "content": question}
    ]

    response = get_completion(messages)
    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

    # Loop cho tới khi model báo stop và đưa ra kết quả
    while finish_reason != "stop":
        tool_call = first_choice.message.tool_calls[0]

        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)

        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps({"result": result})
        })

        print(messages)

        # Chờ kết quả từ LLM
        response = get_completion(messages)
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    # In ra kết quả sau khi đã thoát khỏi vòng lặp
    print("========================================================")
    print("BOT: " + first_choice.message.content)
    print("========================================================")
