job "neural-codec" {
  datacenters = ["dc1"]
  type        = "service"

  group "neural-codec" {
    count = 1

    network {
      port "http" {
        to = 8106
      }
    }

    task "neural-codec-service" {
      driver = "raw_exec"

      config {
        command = "/bin/sh"
        args = [
          "-c",
          "cd /Users/marcos/Downloads/temp/ultravox-pipeline/V2/neural_codec && PYTHONPATH=/Users/marcos/Downloads/temp/ultravox-pipeline/src python3 -m uvicorn app_complete:app --host 0.0.0.0 --port 8106"
        ]
      }

      env {
        PORT = "8106"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src"
      }

      resources {
        cpu    = 2000  # 2 CPU cores (neural codec needs more CPU)
        memory = 2048  # 2 GB (EnCodec model needs memory)
      }

      # Health check
      service {
        name = "neural-codec-service"
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

