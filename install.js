module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r app/requirements.txt"
        ],
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          triton: true,
        }
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch LFM2.5-VL-450M (Gradio). The model downloads from Hugging Face on first run."
      }
    }
  ]
}
