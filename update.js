module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        triton: true,
      }
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      message: [
        "uv pip install -r app/requirements.txt"
      ],
    }
  }]
}
