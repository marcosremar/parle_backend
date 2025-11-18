job "skypilot" {
  datacenters = ["dc1"]
  type        = "service"

  group "skypilot" {
    count = 1

    network {
      port "http" {
        static = 8370
      }
    }

    task "skypilot" {
      driver = "raw_exec"

      config {
        command = "bash"
        args = ["-c", "cd /Users/marcos/Downloads/temp/ultravox-pipeline/V2/skypilot && pip3 install -r requirements.txt && python3 app_complete.py"]
      }

      env {
        PORT = "8370"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/skypilot"
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      # Health check
      service {
        name = "skypilot"
        port = "http"
        provider = "nomad"

        check {
          type     = "http"
          path     = "/health"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
  }
}
