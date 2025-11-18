job "conversation-history" {
  datacenters = ["dc1"]
  type = "service"

  group "conversation-history" {
    count = 1

    network {
      port "http" {
        static = 8010
      }
    }

    task "conversation-history" {
      driver = "raw_exec"

      config {
        command = "python3"
        args = ["/Users/marcos/Downloads/temp/ultravox-pipeline/V2/conversation_history/app_complete.py"]
      }

      # Note: Artifacts removed for local testing - files should be available in workspace

      env {
        PORT = "8010"
        PYTHONPATH = "/Users/marcos/Downloads/temp/ultravox-pipeline/src:/Users/marcos/Downloads/temp/ultravox-pipeline/V2/conversation_history"
        CONVERSATION_HISTORY_STORAGE_PATH = "/tmp/conversation_history/data"
      }

      resources {
        cpu    = 500
        memory = 512
      }
    }
  }
}
