import openai
import time
import os
import json
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OpenAI_API_key_self")

class Get:
    def __init__(self):
        self.prompt = ""
    
    def calc(self, query, temp=1, n=1, model='gpt-4o-mini'):
        '''
        Please implement this function to call your API model.

        **Input:**
        - `query`: string

        **Output:**
        - `[{{output string}}]`
        - `{'prompt': {{prompt cost, float}}, 'completion': {{completion cost, float}}}`

        '''
       # Use ChatCompletion (for GPT-3.5, GPT-4, etc.)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temp,
            n=n
        )

        # Retrieve token usage
        usage = response.usage
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]

        # Example cost rates for GPT-3.5-Turbo (approximate; adjust if needed)
        #   - Prompt tokens ≈ $0.0015 per 1K
        #   - Completion tokens ≈ $0.002 per 1K
        prompt_rate = 0.0015 / 1000
        completion_rate = 0.002 / 1000

        prompt_cost = prompt_tokens * prompt_rate
        completion_cost = completion_tokens * completion_rate

        # Extract each response
        outputs = []
        for choice in response.choices:
            outputs.append(choice.message["content"])

        # Return a list of lists for the outputs, and a dict for costs
        return [outputs], {'prompt': prompt_cost, 'completion': completion_cost}