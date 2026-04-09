# VoxCPM Gradio Interface

A comprehensive Gradio web interface for **VoxCPM 1.5** - an innovative tokenizer-free Text-to-Speech system with context-aware speech generation and zero-shot voice cloning capabilities.

## 🌟 Features

- **🎵 Basic Generation**: Simple text-to-speech synthesis with default voice
- **🎤 Voice Cloning**: Zero-shot voice cloning using reference audio
- **⚙️ Advanced Settings**: Fine-tune generation parameters for optimal results
- **⚡ Streaming Mode**: Real-time speech generation for faster processing
- **🎛️ Full Control**: Adjust CFG value, inference timesteps, retry settings, and more
- **🖥️ User-Friendly Interface**: Clean, intuitive Gradio UI with three specialized tabs

## 📋 Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU recommended (RTX 4090 achieves RTF 0.17)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB for models

## 🚀 Installation

### 1. Install Dependencies

From the repository root:

```bash
pip install -r app/requirements.txt
```

Or install individually:

```bash
pip install voxcpm gradio soundfile numpy torch
```

### 2. Download Models (Optional)

Models will be downloaded automatically on first run, but you can pre-download:

```python
from huggingface_hub import snapshot_download
snapshot_download("openbmb/VoxCPM1.5")
```

## 🎯 Usage

### Starting the Application

```bash
cd app
python app.py
```

The interface will launch at `http://127.0.0.1:7860` (or another available port).

### Pinokio

This repository includes Pinokio launcher scripts in the project root (`install.js`, `start.js`, `update.js`, `reset.js`, `pinokio.js`, `link.js`, `torch.js`). App code lives under `app/`. Use **Install** then **Start** in Pinokio; the Web UI URL is captured automatically when Gradio prints the local URL.

**Verification:** After a successful install, run **Start** and confirm the sidebar shows **Open Web UI** once the terminal prints the Gradio URL. If the tab does not appear, check launcher logs under `logs/api/` (or `pinokio/logs/api/` if your project uses a `pinokio/` folder). From the repo root you can sanity-check launcher scripts with:

```bash
node --check install.js start.js pinokio.js update.js reset.js link.js torch.js
```

### Interface Tabs

#### 🎵 Tab 1: Basic Generation

Simple text-to-speech synthesis with minimal configuration.

**Steps:**
1. Enter your text in the text box
2. Enable/disable text normalization:
   - **ON**: For regular text with numbers and punctuation
   - **OFF**: For phoneme input with precise pronunciation control
3. Click "Generate Speech"

**Example:**
```
Text: "VoxCPM is an innovative end-to-end TTS model from ModelBest."
Text Normalization: OFF (for phoneme input) or ON (for regular text)
```

#### 🎤 Tab 2: Voice Cloning

Clone any voice using a reference audio sample.

**Steps:**
1. Enter the text you want to synthesize
2. Upload a reference audio file (the voice you want to clone)
3. Optionally provide the transcript of the reference audio
4. Configure enhancement:
   - **Enable**: Clean voice at 16kHz (recommended for noisy audio)
   - **Disable**: High-quality cloning up to 44.1kHz
5. Click "Clone Voice"

**Tips:**
- Use clear, high-quality reference audio for best results
- Provide the transcript for improved accuracy
- Reference audio should be 3-10 seconds long

#### ⚙️ Tab 3: Advanced Settings

Fine-tune all generation parameters for optimal results.

**Generation Parameters:**

- **CFG Value** (0.5-5.0, default: 2.0)
  - Higher values: Better adherence to prompt, may sound strained
  - Lower values: More relaxed, natural speech
  - Recommended: 2.5-3.5 for short sentences, 1.5-2.0 for long texts

- **Inference Timesteps** (5-50, default: 10)
  - Higher values: Better quality, slower generation
  - Lower values: Faster generation, draft quality
  - Recommended: 5-10 for drafts, 20-50 for high quality

- **Text Normalization**
  - Enable for regular text with numbers and abbreviations
  - Disable for phoneme input

- **Denoise Output**
  - Enable for cleaner audio (limits to 16kHz, may cause distortion)
  - Disable for natural output

- **Prompt Speech Enhancement**
  - Enable for clean voice cloning (16kHz limit)
  - Disable for high-quality cloning (up to 44.1kHz)

**Retry Settings:**

- **Enable Retry for Bad Cases**: Automatically retry if generation is unstable
- **Maximum Retry Times** (1-10, default: 3): Number of retry attempts
- **Retry Ratio Threshold** (3.0-10.0, default: 6.0): Length restriction for bad case detection

**Generation Modes:**
- **Generate**: Standard generation mode
- **Generate (Streaming)**: Real-time streaming generation for faster processing

## 👩‍🍳 Quick Tips

### Text Input
- **Regular Text**: Keep "Text Normalization" ON for natural text with numbers and punctuation
- **Phoneme Input**: Turn "Text Normalization" OFF for precise pronunciation control

### Voice Cloning
- **Clean Voice Cloning**: Enable "Prompt Speech Enhancement" (16kHz limit)
- **High-Quality Cloning**: Disable enhancement for up to 44.1kHz sampling
- Use clear, noise-free reference audio for best results

### Quality vs Speed
- **Short Sentences**: Increase CFG value (2.5-3.5) for better clarity
- **Long Texts**: Lower CFG value (1.5-2.0) for better stability
- **Fast Draft**: Use lower inference timesteps (5-10)
- **High Quality**: Use higher inference timesteps (20-50)

## 🔧 Technical Details

### Model Information
- **Model**: VoxCPM1.5
- **Sample Rate**: 44.1kHz
- **Token Rate**: 6.25Hz
- **Architecture**: Tokenizer-free TTS
- **Languages**: Chinese and English

### Windows Compatibility
The application includes special handling for Windows systems:
- Disables `torch.compile` to avoid Dynamo errors
- Uses `127.0.0.1` instead of `0.0.0.0` for server binding
- Automatically suppresses PyTorch compilation warnings

### API Access
The application exposes a REST API for programmatic access. When the app is running, replace the host and port with your actual URL (shown in the terminal or Pinokio **Open Web UI**).

- **Browser / docs:** `http://127.0.0.1:7860/docs` (interactive OpenAPI).
- **OpenAPI JSON:** `http://127.0.0.1:7860/openapi.json` (machine-readable routes and schemas).
- **Python:** Use the [Gradio Python client](https://www.gradio.app/guides/getting-started-with-the-python-client), for example `gradio_client.Client("http://127.0.0.1:7860")`, or call REST endpoints with `requests` using paths and bodies from `/openapi.json`.
- **JavaScript:** Use `fetch("http://127.0.0.1:7860/openapi.json")` to discover routes, then `fetch` with `POST` and JSON bodies matching those schemas (same origin if you embed a page on the Gradio port).
- **curl:** `curl -s http://127.0.0.1:7860/openapi.json` to inspect routes; use `curl -X POST -H "Content-Type: application/json" -d "{...}" http://127.0.0.1:7860/...` with payloads taken from the OpenAPI definitions.

## 📊 Performance

- **RTF (Real-Time Factor)**: 0.17 on RTX 4090
- **Streaming Mode**: Faster processing with chunked generation
- **Memory Usage**: Varies based on text length and settings

## ⚠️ Important Notes

- Output quality depends on prompt speech quality
- Model is trained on Chinese and English data
- Please use responsibly and mark AI-generated content appropriately
- For research and development purposes only
- The model may not perform well on languages other than Chinese and English

## 🐛 Troubleshooting

### Model Download Issues
If automatic download fails:
```python
from huggingface_hub import snapshot_download
snapshot_download("openbmb/VoxCPM1.5", cache_dir="./models")
```

### GPU Memory Issues
If you encounter out-of-memory errors:
- Reduce inference timesteps
- Process shorter text segments
- Use CPU mode (slower but works)

### Audio Quality Issues
- Ensure reference audio is clear and high-quality
- Try adjusting CFG value
- Increase inference timesteps
- Disable denoise if output sounds distorted

### Windows-Specific Issues
The application automatically handles Windows compatibility issues. If you still encounter problems:
- Ensure PyTorch is properly installed
- Check that CUDA is available (for GPU acceleration)
- Try running with CPU mode if GPU issues persist

## 📝 License

- **Application**: Apache-2.0
- **VoxCPM Model**: Apache-2.0
- **Gradio**: Apache-2.0

## 🙏 Credits

- **VoxCPM**: OpenBMB/ModelBest
- **Gradio**: Gradio Team
- **Model**: VoxCPM1.5 (44.1kHz, 6.25Hz token rate)

## 📚 Resources

- **VoxCPM GitHub**: https://github.com/OpenBMB/VoxCPM
- **Gradio Documentation**: https://gradio.app/docs/
- **Model on Hugging Face**: https://huggingface.co/openbmb/VoxCPM1.5

## 🤝 Support

For issues and questions:
- VoxCPM Issues: https://github.com/OpenBMB/VoxCPM/issues
- Gradio Issues: https://github.com/gradio-app/gradio/issues

---

**Note**: This is a research tool. Please use responsibly and always mark AI-generated content appropriately.