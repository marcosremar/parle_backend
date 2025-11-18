job "external-llm-file" {
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

      # Copiar arquivo de configuração com API key
      artifact {
        source = "file:///Users/marcos/Downloads/temp/ultravox-pipeline/config/api-keys.env"
        destination = "secrets/api-keys.env"
      }

      env {
        PORT = "8110"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_llm"
      }

      # Carregar variáveis do arquivo
      template {
        data = file("secrets/api-keys.env")
        destination = "local/env"
        env = true
      }

      resources {
        cpu    = 500
        memory = 1024
      }
    }
  }
}
