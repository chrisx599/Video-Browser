# VideoBrowser: JIT Video Understanding Agent

VideoBrowser is an autonomous AI agent designed to research, watch, and analyze video content (primarily YouTube) to answer complex user queries. It implements the **Just-In-Time (JIT) Paradigm** for efficient video understanding, as described in **[Video-JIT: A Just-In-Time Paradigm for Video-Based Agentic Retrieval](https://arxiv.org/abs/2512.23044)**.

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the following installed:
*   **Python 3.10+**
*   **FFmpeg**: Required for video and audio processing.
    *   *Ubuntu/Debian*: `sudo apt install ffmpeg`
    *   *MacOS*: `brew install ffmpeg`
    *   *Windows*: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Video-Browser.git
cd Video-Browser

# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root to store your API keys:

```ini
# .env file

# Required: Core LLM Logic
OPENAI_API_KEY=sk-your_openai_api_key_here

# Optional: Enhanced Search & Transcripts
# OXYLABS_USERNAME=your_username (For robust YouTube scraping)
# OXYLABS_PASSWORD=your_password
# TAVILY_API_KEY=tvly-... (For web text search)
```

### 4. Configuration (`config.yaml`)

Control the agent's behavior by editing `config.yaml`. The defaults are optimized for general use.

**Key Settings:**

```yaml
llm:
  default:
    provider: "openai"
    model: "gpt-4o"  # Recommended model

watcher:
  num_frames: 16             # How many frames to "skim" initially
  video_downloader: "pytubefix" # Options: "pytubefix" (faster) or "ytdlp" (more robust)

selector:
  top_k: 3                   # Max videos to process per search round

transcript:
  provider: "whisper"        # Options: "whisper" (OpenAI API), "ytdlp" (subtitles), "oxylabs"
```

### 5. Running the Agent

You can run the agent using the graph builder script:

```bash
python videobrowser/graph/builder.py
```

*Note: By default, this runs a demo query defined in the script. Open `videobrowser/graph/builder.py` and modify the `inputs` dictionary to change the question.*

```python
# In videobrowser/graph/builder.py
inputs = {"user_query": "Your custom question here..."}
```

## 📚 Citation

If you use this codebase or the JIT paradigm in your research, please cite:

```bibtex
@article{VideoJIT2025,
  title={Video-JIT: A Just-In-Time Paradigm for Video-Based Agentic Retrieval},
  author={Zhengyang Liang and others},
  journal={arXiv preprint arXiv:2512.23044},
  year={2025},
  url={https://arxiv.org/abs/2512.23044}
}
```

## 📄 License

[MIT License](LICENSE)