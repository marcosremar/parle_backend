job "database-service" {
  datacenters = ["dc1"]
  type = "service"

  group "database" {
    count = 1

    network {
      port "http" {
        static = 8300
      }
    }

    task "database" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/database/app_complete.py"]
      }

      # Note: Artifacts removed for local testing - files should be available in workspace

      env {
        PORT = "8300"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/database"
        DATABASE_STORAGE_PATH = "/tmp/database/data"
      }

      resources {
        cpu    = 500
        memory = 512
      }
    }
  }
}
