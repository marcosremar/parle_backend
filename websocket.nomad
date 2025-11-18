job "websocket" {
  datacenters = ["dc1"]
  type        = "service"

  group "websocket" {
    count = 1

    network {
      port "http" {
        static = 8022
      }
    }

    task "websocket" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/websocket/app_complete.py"]
      }

      env {
        PORT = "8022"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/websocket"
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      # Health check
      service {
        name = "websocket"
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
