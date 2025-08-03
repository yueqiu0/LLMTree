import os
from openai import OpenAI

# === PPIO 配置 ===
base_url = "https://api.ppinfra.com/v3/openai"
api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量读取你的API Key
if not api_key:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

model = "deepseek/deepseek-v3-0324"  # PPIO 控制台中的模型 ID

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# === 消息配置 ===
stream = False  # 如需逐步输出改为 True
prompt = "用中文简要解释什么是量子计算"

chat_completion_res = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    stream=stream,
    extra_body={},  # 一般为空，除非文档要求带额外参数
)

# === 输出结果 ===
if stream:
    for chunk in chat_completion_res:
        print(chunk.choices[0].delta.content or "", end="")
else:
    print(chat_completion_res.choices[0].message.content)
