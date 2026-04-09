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
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "hf download openbmb/VoxCPM2"
        ],
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python -c \"from modelscope import snapshot_download; snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base'); snapshot_download('iic/SenseVoiceSmall')\""
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch VoxCPM 2."
      }
    }
  ]
}
