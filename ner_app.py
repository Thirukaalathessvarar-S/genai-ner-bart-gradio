import os, requests, json, gradio as gr
from dotenv import load_dotenv, find_dotenv

os.environ['PORT2'] = '7861'
os.environ['HF_API_NER_BASE'] = 'https://api-inference.huggingface.co/models/dslim/bert-base-NER'

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ.get('HF_API_KEY')

def get_completion(inputs, parameters=None, ENDPOINT_URL=None):
    if ENDPOINT_URL is None:
        ENDPOINT_URL = os.environ.get('HF_API_NER_BASE')

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters:
        data.update({"parameters": parameters})

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=os.environ.get('HF_API_NER_BASE'))

    # Handle model-loading or error messages
    if isinstance(output, dict) and "error" in output:
        return {"text": input, "entities": [{"word": "Error", "entity": "ERROR", "score": 1.0}]}

    # Some responses are nested (list inside list)
    if isinstance(output, list) and isinstance(output[0], list):
        output = output[0]

    # Normalize keys (some models use 'entity_group' instead of 'entity')
    for token in output:
        if "entity_group" in token and "entity" not in token:
            token["entity"] = token["entity_group"]

    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}


print("Launching Gradio app...")
gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using Hugging Face's `dslim/bert-base-NER` model",
    allow_flagging="never",
    examples=[
        ["My name is Andrew, I'm building DeeplearningAI and I live in California"],
        ["My name is Poli, I live in Vienna and work at HuggingFace"]
    ]
)
demo.launch(share=True, server_port=int(os.environ['PORT2']))
