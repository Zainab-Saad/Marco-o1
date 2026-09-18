import requests

URL = ''
MODEL = ''

def set_server(url, model):
    global URL, MODEL
    URL, MODEL = url, model

def get_response(user_question, history_text, end_tokens, n=1, max_tokens=2048,
                temperature=0.6, top_p=0.95, top_k=20, seed=0, **_):
    payload = {
        'model': MODEL,
        'prompt': user_question + history_text,
        'n': n,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'seed': seed,
        'stop': end_tokens if isinstance(end_tokens, list) else [end_tokens],
    }
    r = requests.post(URL, json=payload, timeout=3600)
    r.raise_for_status()
    return [c['text'] for c in r.json()['choices']]