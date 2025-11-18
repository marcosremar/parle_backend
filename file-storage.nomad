job "file-storage" {
  datacenters = ["dc1"]
  type = "service"

  group "file-storage" {
    count = 1

    network {
      port "http" {
        static = 8107
      }
    }

    task "file-storage" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/file_storage/app_complete.py"]
      }

      env {
        PORT = "8107"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/file_storage"
      }

      resources {
        cpu    = 500
        memory = 1024
      }
    }
  }
}

