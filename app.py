import io, re, json
import base64
import torch
from pdf2image import convert_from_bytes
from flask import Flask, jsonify, request, send_file
from PIL import Image
from pptxtoimages.tools import PPTXToImageConverter
from transformers import DonutProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# File type checking
def filetype_check(filename):
    allowed_filetypes = ['pdf', 'ppt', 'pptx']
    return filename.rsplit('.', 1)[1] in allowed_filetypes

# File Conversion
@app.route("/convert_image", methods=["POST"])
def convert_img():
    if 'file' not in request.files:
        return '', 400

    uploaded_file = request.files['file']

    if (uploaded_file and filetype_check(uploaded_file.filename)):
        return pdf_to_image(uploaded_file), 200

    else:
        return "Invalid File", 400
    
def pdf_to_image(pdf_file):
    try:
        img_data = convert_from_bytes(pdf_file.read())

        img_list = []
        for image in img_data:
            # Save Image as Base64 String
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_list.append(base64.b64encode(img_buffer.getvalue()))

        return jsonify({
            "data": img_list
            })
    
    except Exception as ex:
        return str(ex)

# Donut Text Extraction
# Load Donut model and processor once on startup
donut_model = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(donut_model)
model = VisionEncoderDecoderModel.from_pretrained(donut_model)
# Use CPU if GPU is unavailable
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route("/donut_extract", methods=["POST"])
def donut_extract(img_array):

    result = []
    for img in img_array:
        image = img.convert("RGB")

        task_prompt = "<s_synthdog>" 
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = processor(image, return_tensors="pt").pixel_values

        # 4. Generate structured text output
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.config.decoder.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # 6. Post-process to extract raw text elements
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task tag

        # Convert token structures directly to clean text/JSON representation
        result.append(processor.token2json(sequence))

    return result

 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)