job "user-service" {
  datacenters = ["dc1"]
  type        = "service"

  group "user" {
    count = 1

    network {
      port "http" {
        static = 8200
      }
    }

    task "user-service" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["user/app_complete.py"]
      }

      env {
        PORT = "8200"
        PYTHONPATH = "core:user"
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
