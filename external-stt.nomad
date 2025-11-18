job "external-stt" {
  datacenters = ["dc1"]
  type = "service"

  group "external-stt" {
    count = 1

    network {
      port "http" {
        static = 8099
      }
    }

    task "external-stt" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_stt/app_complete.py"]
      }

      # Template para obter API key do Vault
      template {
        data = <<EOF
{{ with secret "secret/data/groq" }}
GROQ_API_KEY={{ .Data.data.api_key }}
{{ end }}
EOF
        destination = "local/env"
        env = true
      }

      env {
        PORT = "8099"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_stt"
      }

      resources {
        cpu    = 500
        memory = 1024
      }
    }
  }
}

