
# SIH Syntax Squad: Real-time, Fully Local Speech-to-Text with Speaker Identification

✨ A high-performance engine for real-time speech transcription and speaker identification, featuring a ready-to-use backend server and a simple demonstration frontend.

Our system delivers ultra-low latency, streaming transcription directly in the browser, all processed locally on your server.

### Powered by Leading Research:

This project integrates state-of-the-art research to achieve its performance:

-   **SimulStreaming (SOTA 2025):** Ultra-low latency transcription using an AlignAtt policy.
-   **NLLB (2024):** Translation to more than 100 languages.
-   **WhisperStreaming (SOTA 2023):** Low latency transcription using a LocalAgreement policy.
-   **Streaming Sortformer (SOTA 2025):** Advanced real-time speaker diarization.
-   **Diart (SOTA 2021):** Real-time speaker diarization.
-   **Silero VAD (2024):** Enterprise-grade Voice Activity Detection.

### Why Our Approach is Better

Standard speech-to-text models are designed for complete audio files, not real-time streams. Feeding them small, live audio chunks results in poor context, clipped words, and inaccurate transcriptions. Our engine is built on state-of-the-art simultaneous speech research, enabling intelligent audio buffering and incremental processing for highly accurate, low-latency results.

## Architecture


The backend is designed for concurrency, supporting multiple users at once. Incoming audio is processed and segmented by a Voice Activity Detection (VAD) module to reduce computational overhead during silent periods. The core engine then uses parallel transcription and diarization models to generate a final, speaker-labeled transcript.

## Installation & Quick Start

First, clone the repository to your local machine:
```bash
git clone [your-repo-url]
cd [your-repo-name]
````

Then, install the package and its dependencies:

```bash
pip install -e .
```

### Quick Start

1.  **Start the transcription server:**
    *(Note: This assumes `realtime_server` is defined as an entry point in your `setup.py`)*

    ```bash
    realtime_server --model base --language en
    ```

2.  **Open your browser** and navigate to `http://localhost:8000`. Start speaking and watch your words appear in real-time\!

## Optional Dependencies

| Feature                               | Installation Command                                                 |
| ------------------------------------- | -------------------------------------------------------------------- |
| Speaker diarization with Sortformer   | `pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]` |
| Apple Silicon optimized backend       | `pip install mlx-whisper`                                            |
| NLLB Translation                      | `pip install huggingface_hub transformers`                           |
| [Not Recc] Diarization with Diart     | `pip install diart`                                                  |
| [Not Recc] Original Whisper backend   | `pip install whisper`                                                |
| [Not Recc] Improved timestamps backend| `pip install whisper-timestamped`                                    |
| OpenAI API backend                    | `pip install openai`                                                 |

*To use speaker diarization with Diart, you must accept user conditions on HuggingFace for the required models and log in via `huggingface-cli login`.*

## Usage Examples

### Command-line Interface

Start the server with various options:

```bash
# Use a large model and translate from French to Danish
realtime_server --model large-v3 --language fr --target-language da

# Enable diarization and listen on all network interfaces on port 80
realtime_server --host 0.0.0.0 --port 80 --model medium --diarization --language fr
```

### Python API Integration

You can integrate the engine directly into your own FastAPI application.

**Note:** Replace `your_package_name` with the actual name of your Python package.

```python
# main.py
from your_package_name import TranscriptionEngine, AudioProcessor
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import asyncio

transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    # Initialize the core engine
    transcription_engine = TranscriptionEngine(model="medium", diarization=True, lang="en")
    yield

app = FastAPI(lifespan=lifespan)

async def handle_results(websocket: WebSocket, generator):
    async for response in generator:
        await websocket.send_json(response)

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    await websocket.accept()

    # Create a new audio processor for each client
    audio_processor = AudioProcessor(transcription_engine=transcription_engine)
    results_generator = await audio_processor.create_tasks()

    # Run the result handler
    asyncio.create_task(handle_results(websocket, results_generator))

    try:
        while True:
            # Process incoming audio bytes from the client
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except Exception as e:
        print(f"Connection closed: {e}")
```

## Parameters & Configuration

A full list of runtime parameters is available by running `realtime_server --help`. Key parameters are listed below.

| Parameter                 | Description                                                                                                    | Default        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------- |
| `--model`                 | Model size (e.g., `tiny`, `base`, `small`, `medium`, `large-v3`).                                                | `small`        |
| `--language`              | Language code (`en`, `fr`, etc.). Set to `auto` for automatic detection.                                         | `auto`         |
| `--target-language`       | If set, activates translation using NLLB. Example: `fr`.                                                       | `None`         |
| `--diarization`           | Enable speaker identification.                                                                                 | `False`        |
| `--backend`               | Processing backend (`simulstreaming` or `faster-whisper`).                                                     | `simulstreaming` |
| `--host`                  | Server host address.                                                                                           | `localhost`    |
| `--port`                  | Server port.                                                                                                   | `8000`         |

## Deployment Guide

To deploy in a production environment:

1.  **Server Setup:** Use a production-grade ASGI server like Gunicorn with Uvicorn workers.

    ```bash
    pip install uvicorn gunicorn
    gunicorn -k uvicorn.workers.UvicornWorker -w 4 main:app
    ```

2.  **Nginx Reverse Proxy (Recommended):**

    ```nginx
    server {
       listen 80;
       server_name your-domain.com;

        location / {
            proxy_pass [http://127.0.0.1:8000](http://127.0.0.1:8000);
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
    }
    ```

3.  **HTTPS Support:** For secure deployments (`wss://`), configure SSL in your reverse proxy and ensure your frontend connects to the secure endpoint.

## Docker

Deploy the application easily using Docker with GPU or CPU support.

#### Prerequisites

  - Docker installed.
  - For GPU support: NVIDIA Docker runtime installed.

### Build and Run

#### With GPU Acceleration (Recommended)

```bash
# Build the Docker image
docker build -t syntax-squad-stt .

# Run the container
docker run --gpus all -p 8000:8000 --name stt_server syntax-squad-stt --model medium --language en
```

#### CPU Only

```bash
# Build the CPU-specific image
docker build -f Dockerfile.cpu -t syntax-squad-stt-cpu .

# Run the container
docker run -p 8000:8000 --name stt_server syntax-squad-stt-cpu --model medium --language en
```

## Use Cases

  - **Live Meeting Transcription:** Capture discussions in real-time with speaker labels.
  - **Accessibility Tools:** Help hearing-impaired users follow conversations.
  - **Content Creation:** Automatically transcribe podcasts, interviews, or videos.
  - **Customer Service:** Transcribe and analyze support calls with speaker identification.

<!-- end list -->

```
```
