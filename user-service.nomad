job "user-service" {
  datacenters = ["dc1"]
  type        = "service"

  group "user" {
    count = 1

    network {
      port "http" {
        static = 8202
      }
    }

    task "user-service" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/user/app_complete.py"]
      }

      env {
        PORT = "8202"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/user"
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      # Health check
      service {
        name = "user-service"
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
