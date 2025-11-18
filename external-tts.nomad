job "external-tts" {
  datacenters = ["dc1"]
  type = "service"

  group "external-tts" {
    count = 1

    network {
      port "http" {
        static = 8103
      }
    }

    task "external-tts" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_tts/app_complete.py"]
      }

      # Templates para obter API keys do Vault
      template {
        data = <<EOF
{{ with secret "secret/data/huggingface" }}
HF_API_KEY={{ .Data.data.api_key }}
{{ end }}
EOF
        destination = "local/hf.env"
        env = true
      }

      template {
        data = <<EOF
{{ with secret "secret/data/elevenlabs" }}
ELEVENLABS_API_KEY={{ .Data.data.api_key }}
{{ end }}
EOF
        destination = "local/eleven.env"
        env = true
      }

      env {
        PORT = "8103"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/external_tts"
      }

      resources {
        cpu    = 500
        memory = 1024
      }
    }
  }
}
