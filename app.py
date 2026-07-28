import io
import base64
import torch
from pdf2image import convert_from_bytes
from flask import Flask, jsonify, request, send_file
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# File type checking
def filetype_check(filename):
    allowed_filetypes = ['pdf', 'ppt', 'pptx']
    return filename.rsplit('.', 1)[1] in allowed_filetypes
        
# File Conversion
def pdf_to_image(pdf_file):
    try:
        png_data = convert_from_bytes(pdf_file.read())

        full_png = []
        for pages in png_data:
            img_io = io.BytesIO()
            pages.save(img_io, format='PNG')
            img_io.seek(0)

            full_png.append(base64.b64encode(img_io.getvalue()).decode('utf-8'))

        return full_png

        # jsonify({
        #    "filename": "page_1.png",
        #    "mimeType": "image/png",
        #    "data": full_png
        # })
    
    except Exception as e:
        return str(e), 400

# Load Donut processor and model (Using a generic parsing model, or fine-tuned if applicable)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# Set device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def donut_extract(img_array):
    result = []
    for image in img_array:
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = decoder_input_ids.to(device)

        # Generate output
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.eos_token_id]],
            return_dict_in_generate=True,
        )

        # Decode sequence
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = sequence.replace(task_prompt, "")

        # Extract structured data
        result.append(processor.token2json(sequence))

    return result

# Main File Checking Module
@app.route("/test_route", methods=["POST"])
def predict_document():
    if 'file' not in request.files:
        return '', 400

    uploaded_file = request.files['file']

    if (uploaded_file and filetype_check(uploaded_file.filename)):
        return donut_extract(pdf_to_image(uploaded_file))

    else:
        return uploaded_file.filename.rsplit('.', 1)[1], 400

 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)