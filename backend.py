#instalation
# Install Transformers for  DETR, Depth pipeline
!pip install transformers
# Install BitsAndBytes for 8-bit quantization
!pip install bitsandbytes
# Install OpenCV for image processing
!pip install opencv-python-headless
# support for cross origin
!pip install flask-cors
#ngrok install
!pip install pyngrok

# app.py
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import io
import numpy as np
import torch
import cv2
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import os
import base64
from flask_cors import CORS
from openai import OpenAI
from flask import Response
import threading
import queue
import time
#ngrok//////////////////////////////////////////

from pyngrok import ngrok
import os

# Terminate any existing ngrok processes
ngrok.kill()

# Set your authtoken 
ngrok.set_auth_token("NGROK_TOKEN")

# Open a tunnel to your Flask app's port (default is 5000)
port = int(os.environ.get("PORT", 5000))  # Get port from environment or default
tunnel = ngrok.connect(port)

# Print the public URL
print("Public URL:", tunnel.public_url)

os.environ["NGROK_URL"] = tunnel.public_url
#//////////////////////////////////
app = Flask(__name__)
CORS(app)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- OpenAI Client Setup for Qwen ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_REFERER_URL = os.environ.get("OPENROUTER_REFERER_URL", "http://localhost:5000")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_TITLE", "My Flask App")

if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable not set. OpenAI calls will fail.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

# --- Model Loading ---
def load_detr():
    print("Loading DETR...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)
    return processor, model, model.config.id2label

def load_depth_anything():
    print("Loading Depth Anything...")
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

detr_processor, detr_model, id2label = load_detr()
depth_estimator = load_depth_anything()

def analyze_scene(image, detection_sensitivity=0.3):
    """Performs object detection, depth estimation."""
    pil_image = Image.open(io.BytesIO(image))
    width, height = pil_image.size

    # --- DETR Object Detection ---
    inputs = detr_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detr_model(**inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
    results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=detection_sensitivity)[0]

    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]

    # --- Depth Estimation ---
    depth_output = depth_estimator(pil_image)
    depth_map = np.array(depth_output["depth"])
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
    depth_map_display = Image.fromarray(depth_map_color)

    # --- Draw Bounding Boxes ---
    image_with_boxes = np.array(pil_image).copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(id2label), 3), dtype="uint8")

    depth_info_lines = []

    for i in range(len(boxes)):
        score = scores[i].item()
        label_idx = labels[i].item()
        class_name = id2label[label_idx]

        if score >= detection_sensitivity:
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            center_depth = depth_map[cy, cx] if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1] else 0.0
            color = tuple(map(int, colors[label_idx]))

            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            label_str = f"{class_name} ({score:.2f})"
            (label_width, label_height), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_with_boxes, (x1, y1 - 20), (x1 + label_width, y1), color, -1)
            cv2.putText(image_with_boxes, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            depth_label = f"Depth: {center_depth:.2f}"
            cv2.putText(image_with_boxes, depth_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            depth_info_lines.append(f"- {class_name}: center_depth={center_depth:.2f}")

    image_with_boxes_display = Image.fromarray(image_with_boxes)

    # --- Convert PIL images to base64 ---
    buffered_boxes = io.BytesIO()
    image_with_boxes_display.save(buffered_boxes, format="JPEG")
    boxes_b64 = base64.b64encode(buffered_boxes.getvalue()).decode("utf-8")

    buffered_depth = io.BytesIO()
    depth_map_display.save(buffered_depth, format="JPEG")
    depth_b64 = base64.b64encode(buffered_depth.getvalue()).decode("utf-8")

    return boxes_b64, depth_b64, depth_info_lines

def format_description(raw_description):
    formatted_description = "Scene Description:\n\n"
    formatted_description += "----------------------------------------\n"
    formatted_description += raw_description
    return formatted_description

# --- System Prompt ---
SYSTEM_PROMPT = "You are an assistant developed by Team 5 designed to help blind people understand the environment around them."

def qwen_describe_image(image_file, user_input="Describe the image for a blind person in a structured way."):
    """Describes the image using Qwen2.5 VL via OpenRouter API."""
    try:
        pil_image = Image.open(io.BytesIO(image_file))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{img_base64}"

        instruction = "Provide a concise and detailed description of the scene in this image, aiming for approximately 10 lines or less. Focus on the most important aspects of the scene, including the main objects, the environment, and any significant activities or context. Describe it in a way that is informative and easy to understand for someone who cannot see. Be brief but comprehensive."

        user_text = f"{instruction}\n\nUser request: {user_input}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}, # System prompt added here
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": user_text
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": img_url
                  }
                }
              ]
            }
          ]

        completion = client.chat.completions.create(
          extra_headers={
            "HTTP-Referer": OPENROUTER_REFERER_URL,
            "X-Title": OPENROUTER_TITLE,
          },
          model="qwen/qwen2.5-vl-72b-instruct:free",
          messages=messages
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in qwen_describe_image: {e}")
        return f"Error describing image: {e}"

def qwen_chat_text(user_message, image_base64=None, depth_info_lines=None):
    """Chat with Qwen2.5 via OpenRouter API for text interactions."""
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] # System prompt added here
        content_list = []

        if image_base64:
            img_url = f"data:image/jpeg;base64,{image_base64}"
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": img_url
                }
            })

        if depth_info_lines:
            depth_info_text = "\n".join(depth_info_lines)
            user_message_with_depth = f"{user_message}\n\nObject depth information:\n{depth_info_text}"
        else:
            user_message_with_depth = user_message

        content_list.append({
            "type": "text",
            "text": user_message_with_depth
        })

        messages.append({"role": "user", "content": content_list})

        completion = client.chat.completions.create(
          extra_headers={
            "HTTP-Referer": OPENROUTER_REFERER_URL,
            "X-Title": OPENROUTER_TITLE,
          },
          model="google/gemini-2.0-flash-thinking-exp:free",
          messages=messages
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in qwen_chat_text: {e}")
        return f"Error in chat: {e}"

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handles chat messages."""
    print("Entering handle_chat function...") # Debug print
    try:
        user_message = request.form.get('message')
        image_b64_chat = request.form.get('image_b64')
        depth_info_lines_chat = request.form.get('depth_info_lines')

        depth_info_lines_chat_list = None
        if depth_info_lines_chat:
            depth_info_lines_chat_list = depth_info_lines_chat.splitlines()

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        print("About to call qwen_chat_text...") # Debug print
        chatbot_response = qwen_chat_text(user_message, image_base64=image_b64_chat, depth_info_lines=depth_info_lines_chat_list)
        print("qwen_chat_text call returned successfully...") # Debug print

        return jsonify({"response": chatbot_response})

    except Exception as e:
        print(f"Error in /chat route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    """Processes image and returns results."""
    try:
        image_file = request.files['image'].read()
        sensitivity = float(request.form.get('sensitivity', 0.3))

        boxes_b64, depth_b64, depth_info_lines = analyze_scene(image_file, sensitivity)
        raw_description = qwen_describe_image(image_file)
        formatted_description = format_description(raw_description)

        depth_info_lines_str = "\n".join(depth_info_lines) if depth_info_lines else ""

        return jsonify({
            'processed_image': boxes_b64,
            'depth_map': depth_b64,
            'description': formatted_description,
            'depth_info_lines': depth_info_lines_str
        })
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    ngrok_url = os.environ.get("NGROK_URL")
    if not ngrok_url:
        return "Ngrok URL not set. Please run ngrok separately and set the NGROK_URL environment variable."
    return render_template('index.html', ngrok_url=ngrok_url)

# --- Real-time processing variables ---
frame_queue = queue.Queue(maxsize=10)
processing_lock = threading.Lock()
stop_stream = threading.Event()

def process_frame(frame_data):
    """Process a single frame."""
    try:
        image = Image.open(io.BytesIO(frame_data))

        inputs = detr_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = detr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3)[0]

        depth_output = depth_estimator(image)
        depth_map = np.array(depth_output["depth"])

        image_array = np.array(image)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(id2label), 3), dtype="uint8")

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth_value = depth_map[cy, cx] if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1] else 0

            color = tuple(map(int, colors[label.item()]))
            cv2.rectangle(image_array, (x1, y1), (x2, y2), color, 2)
            label_text = f"{id2label[label.item()]}: {score.item():.2f}, Depth: {depth_value:.2f}"
            cv2.putText(image_array, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        processed_frame = cv2.imencode('.jpg', image_array)[1].tobytes()
        return processed_frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def generate_frames():
    """Generator function for streaming frames with frame skipping."""
    frame_count = 0  # Counter to track frames
    skip_rate = 2     # Process every 'skip_rate' frames (e.g., 2 means process every other frame)

    while not stop_stream.is_set():
        try:
            frame_data = frame_queue.get(timeout=1.0)
            frame_count += 1

            if frame_count % skip_rate == 0:  # Process only if frame_count is a multiple of skip_rate
                processed_frame = process_frame(frame_data)
                if processed_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            continue

@app.route('/stream-frame', methods=['POST'])
def receive_frame():
    """Endpoint to receive frames."""
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    frame_data = request.files['frame'].read()
    try:
        frame_queue.put_nowait(frame_data)
        return jsonify({'status': 'success'})
    except queue.Full:
        return jsonify({'status': 'queue full'}), 503

@app.route('/video-feed')
def video_feed():
    """Endpoint to stream video."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-stream')
def start_stream():
    """Endpoint to start streaming."""
    stop_stream.clear()
    return jsonify({'status': 'streaming started'})

@app.route('/stop-stream')
def stop_stream_route():
    """Endpoint to stop streaming."""
    stop_stream.set()
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    return jsonify({'status': 'streaming stopped'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
