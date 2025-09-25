import time
from openai import OpenAI, AsyncOpenAI
from typing import Union, List, Optional

class ChatEngine:
    def __init__(self, engine_ip: str, model_id: str):
        engine_url = "http://" + engine_ip + ":8080/v1"
        self.client = OpenAI(
            base_url=engine_url,
            api_key="token"
        )
        self.client_async = AsyncOpenAI(
            base_url=engine_url,
            api_key="token"
        )
        pos = model_id.rfind('/')
        model_id = model_id[:pos]
        self.model_id = model_id.replace('.', '___')
    
    async def chat(self, prompt: Union[List, str], stream: Optional[bool] = False):
        if stream:
            chat = self.client.chat.completions.create(
                model=self.model_id,
                messages=prompt,
                max_tokens=1024,
                temperature=1,
                stream=stream
            )
            return chat
        else:
            assert isinstance(prompt, str)
            prompt = [{"role": "user", "content": prompt}]
            chat = await self.client_async.chat.completions.create(
                model=self.model_id,
                messages=prompt,
                max_tokens=1024,
                temperature=1,
                stream=stream
            )
            return chat.choices[0].message.content