from openai import OpenAI
 
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")   # any key; it is ignored
r = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "Guess the meaning of the word 'trendipitious'."}],
    max_tokens=256,
)
print(r)#.choices[0].message.content)