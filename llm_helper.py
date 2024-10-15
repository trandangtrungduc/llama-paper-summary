from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
from PIL import Image
from io import BytesIO
    
def llama32(prompt_or_messages, temperature=0, model_size=90):
  model = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
  url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
  payload = {
    "model": model,
    "max_tokens": 4096,
    "temperature": temperature,
    "stop": ["<|eot_id|>","<|eom_id|>"],
    "messages": prompt_or_messages
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  return res['choices'][0]['message']['content']

def llama31(prompt_or_messages, temperature=0, model_size=8, raw=False, debug=False):
    model = f"meta-llama/Meta-Llama-3.1-{model_size}B-Instruct-Turbo"
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt
        }
    else:
        messages = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }

    if debug:
        print(payload)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()  
        res = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

    if 'error' in res:
        raise Exception(f"API Error: {res['error']}")

    if raw:
        return res

    if isinstance(prompt_or_messages, str):
        return res['choices'][0].get('text', '')
    else:
        return res['choices'][0].get('message', {}).get('content', '')
    
def generate_summary(title, model, inform, temperature):
    authors_prompt = [
    {
        "role": "user",
        "content": f"List the authors of the paper '{title}'. Only list authors in numerical order."
        }
    ]
    
    year_prompt = [
    {
        "role": "user",
        "content": f"What year was the '{title}' published?'. Only write the year of publication."
        }
    ]
    
    venue_prompt = [
    {
        "role": "user",
        "content": f"Which conference or journal was the '{title}' published? Only write the full name and abbreviation of the conference or journal."
        }
    ]
    
    github_prompt = [
    {
        "role": "user",
        "content": f"Find the Github code link for the '{title}' published? Only write the link."
        }
    ]
    
    if inform:
        information = ' and '.join(inform)
        summary_prompt = [
        {
            "role": "user",
            "content": f"Summarize parts {information} of the paper '{title}'. Present the results in numerical order."
            }
        ]
    else:
        summary_prompt = [
        {
            "role": "user",
            "content": f"Summarize the paper '{title}'."
            }
        ]
        
    if model == "LLama 3.1 - 8B":
        author_res = llama31(authors_prompt, temperature, 8)
        year_res = llama31(year_prompt, temperature, 8)
        venue_res = llama31(year_prompt, temperature, 8)
        summary_res = llama31(summary_prompt, temperature, 8)
        github_res = llama31(github_prompt, temperature, 8) 
    elif model == "LLama 3.1 - 70B":
        author_res = llama31(authors_prompt, temperature, 70)
        year_res = llama31(year_prompt, temperature, 70)
        venue_res = llama31(venue_prompt, temperature, 70)
        summary_res = llama31(summary_prompt, temperature, 70)
        github_res = llama31(github_prompt, temperature, 70) 
    elif model == "LLama 3.1 - 405B":
        author_res = llama31(authors_prompt, temperature, 405)
        year_res = llama31(year_prompt, temperature, 405)
        venue_res = llama31(venue_prompt, temperature, 405)
        summary_res = llama31(summary_prompt, temperature, 405)
        github_res = llama31(github_prompt, temperature, 405) 
    elif model == "LLama 3.2 - 90B":
        author_res = llama32(authors_prompt, temperature, 90)
        year_res = llama32(year_prompt, temperature, 90)
        venue_res = llama32(venue_prompt, temperature, 90)
        summary_res = llama32(summary_prompt, temperature, 90)
        github_res = llama32(github_prompt, temperature, 90) 
    else:
        return '', '', '', '', ''
    
    return summary_res, author_res, year_res, venue_res, github_res

def show_image(img_url):
    if img_url.startswith("http://") or img_url.startswith("https://"):
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_url)
    return img
        
def generate_description(img_prompt, img_url):
    prompt = [
        {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": img_prompt
                },
            {
                "type": "image_url",
                "image_url": {"url": img_url}
            }
        ]
        },
    ]
    return show_image(img_url), llama32(prompt, 0, 90)