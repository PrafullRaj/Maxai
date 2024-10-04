from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        input_text = data.get('text', 'Default ad content')

        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=50, do_sample=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"description": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return the error with a 500 status code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
