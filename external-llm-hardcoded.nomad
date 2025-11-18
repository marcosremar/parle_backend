job "external-llm-hardcoded" {
  datacenters = ["dc1"]
  type = "service"

  group "external-llm" {
    count = 1

    network {
      port "http" {
        static = 8110
      }
    }

    task "external-llm" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_llm/app_complete.py"]
      }

      env {
        PORT = "8110"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_llm"
        GROQ_API_KEY = "${GROQ_API_KEY}"  # Definir como variável de ambiente
      }

      resources {
        cpu    = 500
        memory = 1024
      }
    }
  }
}
