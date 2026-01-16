# Video-Browser: Towards Agentic Open-web Video Browsing

VideoBrowser is an autonomous AI agent designed to research, watch, and analyze video content to answer complex user queries. It implements the **Pyramidal Perception** for efficient video understanding.

<p align="center">
    <a href="https://liang-zhengyang.github.io/video-browsecomp/" target="_blank">
        <img src="https://img.shields.io/badge/🏠-Project_Page-orange.svg" alt="Project Page">
    </a>
    <a href="https://arxiv.org/abs/2512.23044" target="_blank">
        <img src="https://img.shields.io/badge/📄-Paper-brightgreen.svg" alt="Paper">
    </a>
    <a href="https://huggingface.co/datasets/chr1ce/Video-Browsecomp" target="_blank">
        <img src="https://img.shields.io/badge/📊-Benchmark-yellow.svg" alt="Benchmark">
    </a>
</p>

<p align="center">
  <img src="assets/cover.jpeg" align="center" width="90%">
</p>

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Video-Browser.git
cd Video-Browser

# Recommended: Create a virtual environment
conda create -n videobrowser python=3.10
conda activate videobrowser

pip install -r requirements.txt
pip install -e .
```

### 3. Environment Setup

Create a `.env` file in the project root to store your API keys:
cp .env.example .env

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
python run_cli.py
```

*Note: By default, this runs a demo query defined in the script. Open `run_cli.py` and modify the `inputs` dictionary to change the question.*

### 6. Running the Web UI

To use the interactive Chat UI (powered by Chainlit):

```bash
chainlit run app.py
```
This will start a local server at `http://localhost:8000`. You can interact with the agent just like a chat application.



## 📚 Citation

If you use the videobrowser in your research, please cite:

```bibtex
@misc{liang2025videobrowsecompbenchmarkingagenticvideo,
      title={Video-BrowseComp: Benchmarking Agentic Video Research on Open Web}, 
      author={Zhengyang Liang and Yan Shu and Xiangrui Liu and Minghao Qin and Kaixin Liang and Paolo Rota and Nicu Sebe and Zheng Liu and Lizi Liao},
      year={2025},
      eprint={2512.23044},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.23044}, 
}
```

## 📄 License

[MIT License](LICENSE)

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=chrisx599/Video-Browser&type=Date)](https://star-history.com/#chrisx599/Video-Browser&Date)
