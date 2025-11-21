# Nomad & Infraestrutura Guidelines

Guia resumido de boas práticas e decisões de arquitetura para o projeto Parle Backend.

## 1. Drivers de Execução: `raw_exec` vs `docker`

### Desenvolvimento (macOS / Apple Silicon)
*   **Recomendado:** `raw_exec`
*   **Motivo:**
    *   Evita overhead da VM Docker no Mac (economiza ~2GB+ RAM).
    *   Acesso nativo a hardware (microfone/alto-falantes) para testes de áudio.
    *   Iteração rápida (sem build de imagem a cada alteração).
    *   Usa o mesmo `venv` local do desenvolvedor.

### Produção (Linux)
*   **Recomendado:** `docker`
*   **Motivo:**
    *   Isolamento total e segurança.
    *   Garante ambiente idêntico em qualquer servidor.
    *   No Linux, Docker tem overhead de performance desprezível (< 1%).

## 2. Gerenciamento de Recursos (Servidor 8GB RAM)

*   **Viabilidade:** É possível rodar todo o stack (16+ serviços) em 8GB de RAM no Linux.
*   **Estratégia de IA:**
    *   **LLM/STT/TTS Pesados:** Usar APIs externas (Groq, ElevenLabs) ou modelos leves.
    *   **Evitar:** Rodar LLMs locais grandes (Llama-3 8B+) junto com o stack completo.
*   **Monitoramento:** Manter serviços Python leves (~150MB cada) para sobrar memória.

## 3. Ambientes Virtuais (Venvs)

*   **Estratégia:** 1 Venv por serviço (Microserviços) ou 1 Venv Monolítico.
*   **Performance:** O uso de múltiplos venvs **NÃO** aumenta consumo de RAM/CPU em execução.
*   **Processos:** O número de processos Python é igual ao número de serviços, independente dos venvs.
*   **Recomendação:** Isolamento é preferível para evitar conflitos de dependências ("Dependency Hell"), ao custo apenas de espaço em disco.

## 4. Configurações Críticas do Nomad

Todo arquivo `.nomad` deve conter:

### A. Logs (Log Rotation)
Evita encher o disco do servidor.
```hcl
logs {
  max_files     = 10
  max_file_size = 10  # MB
}
```

### B. Resiliência (Restart Policy)
Evita loops infinitos de reinicialização em caso de bug.
```hcl
restart {
  attempts = 3
  interval = "2m"
  delay    = "15s"
  mode     = "fail"
}
```

### C. Updates Seguros (Rolling Updates)
Permite deploy sem downtime e rollback automático.
```hcl
update {
  max_parallel      = 1
  health_check      = "checks"
  min_healthy_time  = "10s"
  healthy_deadline  = "5m"
  auto_revert       = true
}
```

## 5. Service Discovery (Consul)

O Consul atua como a "Lista Telefônica" e o "Médico" do cluster.

*   **Papel:**
    *   **Lista Telefônica:** Sabe onde cada serviço está (IP e Porta).
    *   **Médico (Health Check):** Monitora se o serviço está saudável. Se cair, para de enviar tráfego.
*   **Integração com Nomad:**
    *   O Nomad registra os serviços no Consul automaticamente (`service { ... }`).
    *   O Nomad usa o Consul para preencher os templates de configuração (`template { ... }`).
*   **Implementação Prática:**
    *   **Nomad:** Usar templates `{{ range service "nome-do-servico" }}`.
    *   **Python:** Ler variável de ambiente `os.getenv("SERVICE_URL")`.
    *   **Portas:** Cada serviço deve ter uma porta única e exclusiva para evitar conflitos no host (`raw_exec`).

