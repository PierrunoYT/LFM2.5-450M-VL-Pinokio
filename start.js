module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {},
        message: [
          "python app/app.py"
        ],
        on: [{
          "event": "/http:\/\/\\S+/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
