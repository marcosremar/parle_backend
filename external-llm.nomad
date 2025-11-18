job "external-llm" {
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

      # Note: Artifacts removed for local testing - files should be available in workspace

      # Template para ler secrets do Vault
      template {
        data = <<EOF
{{ with secret "secret/external-llm" }}
GROQ_API_KEY={{ .Data.data.groq_api_key }}
{{ end }}
EOF
        destination = "secrets/api-keys.env"
        env = true
      }

      env {
        PORT = "8110"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_llm"
      }

      resources {
        cpu    = 500
        memory = 1024
      }
    }
  }
}
