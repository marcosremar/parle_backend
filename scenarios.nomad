job "scenarios" {
  datacenters = ["dc1"]
  type = "service"

  group "scenarios" {
    count = 1

    network {
      port "http" {
        static = 8700
      }
    }

    task "scenarios" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/scenarios/app_complete.py"]
      }

      env {
        PORT = "8700"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/scenarios"
      }

      resources {
        cpu    = 500
        memory = 1024
      }
    }
  }
}
