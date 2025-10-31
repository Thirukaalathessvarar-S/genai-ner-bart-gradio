## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:

### DESIGN STEPS:

#### STEP 1:
The project uses the Hugging Face Inference API with the pretrained model dslim/bert-base-NER.

#### STEP 2:
The raw model output returns multiple tokens for single entities (like “New” + “Delhi”).

#### STEP 3:
A Gradio interface is created for easy user interaction. The user inputs a text, and the system highlights recognized entities (like Person, Location, Organization, etc.).The gr.Interface() function defines input and output component.

Input: Textbox

Output: Highlighted text showing entity categories

### PROGRAM:
```
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

```

### OUTPUT:
![WhatsApp Image 2025-10-31 at 11 59 10_2552d9a2](https://github.com/user-attachments/assets/1394ab3c-0bdc-49b8-85c5-e8dbe7c4dea2)

![WhatsApp Image 2025-10-31 at 11 58 39_4ef6e995](https://github.com/user-attachments/assets/fb54425e-0955-4d8c-91fe-1b3b967e5f7f)


### RESULT:
The Gradio interface successfully identifies and highlights named entities such as persons, organizations, and locations from the given input text using the dslim/bert-base-NER model.
