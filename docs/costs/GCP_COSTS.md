# 💰 Custos do Google Cloud Platform

## 📊 Cálculo Básico

### Instância a $0.02/hora

- **Preço por hora**: $0.02
- **Horas por mês** (30 dias): 720 horas
- **Custo mensal**: **$14.40**

### Detalhamento

```
$0.02/hora × 24 horas/dia × 30 dias = $14.40/mês
```

## 🖥️ Opções de Máquinas no Google Cloud

### 1. Compute Engine (VMs)

#### e2-micro (Free Tier elegível)
- **CPU**: 0.25-1 vCPU compartilhado
- **Memória**: 1 GB
- **Preço**: **$0.0067/hora** (~$4.82/mês)
- **Região**: us-central1

#### e2-small
- **CPU**: 0.5-2 vCPU compartilhado
- **Memória**: 2 GB
- **Preço**: **$0.0134/hora** (~$9.65/mês)

#### e2-medium
- **CPU**: 1-2 vCPU compartilhado
- **Memória**: 4 GB
- **Preço**: **$0.0268/hora** (~$19.30/mês)

#### e2-standard-2
- **CPU**: 2 vCPU
- **Memória**: 8 GB
- **Preço**: **$0.0671/hora** (~$48.31/mês)

### 2. Cloud Run (Serverless)

#### Preços Cloud Run
- **CPU**: $0.00002400 por vCPU-segundo
- **Memória**: $0.00000250 por GiB-segundo
- **Requisições**: Primeiros 2 milhões grátis, depois $0.40 por milhão

#### Exemplo de Custo Cloud Run
Para uma aplicação que:
- Usa 1 vCPU e 512 MB de memória
- Processa 1000 requisições/dia
- Cada requisição demora 1 segundo

**Cálculo mensal**:
- CPU: 1000 req/dia × 1s × 30 dias × $0.00002400 = **$0.72**
- Memória: 1000 req/dia × 1s × 0.5 GiB × 30 dias × $0.00000250 = **$0.04**
- Requisições: Grátis (dentro do limite)

**Total**: **~$0.76/mês** ✅

### 3. Comparação: Compute Engine vs Cloud Run

| Recurso | Compute Engine | Cloud Run |
|---------|---------------|-----------|
| **Custo mínimo** | ~$4.82/mês (e2-micro) | ~$0.76/mês (uso baixo) |
| **Custo $0.02/hora** | e2-small (~$14.40/mês) | N/A (pago por uso) |
| **Uso 24/7** | Fixo | Variável |
| **Ideal para** | Aplicações sempre rodando | Aplicações com tráfego variável |

## 💡 Recomendações

### Para Aplicações com Tráfego Baixo/Intermittente
**Cloud Run** é mais econômico:
- Paga apenas quando há requisições
- Sem custo quando inativo
- Escala automaticamente

### Para Aplicações Sempre Rodando
**Compute Engine** pode ser melhor:
- Custo fixo previsível
- Mais controle sobre recursos
- Melhor para workloads constantes

### Exemplo Real: Parle Backend

Se o Parle Backend:
- Tem tráfego variável
- Pode ter períodos inativos
- Precisa escalar automaticamente

**Recomendação**: **Cloud Run**
- Custo estimado: **$5-15/mês** (dependendo do tráfego)
- Muito mais barato que manter VM 24/7

## 📈 Estimativa de Custos Mensais

### Cenário 1: Baixo Tráfego
- 1.000 requisições/dia
- 1 segundo por requisição
- **Cloud Run**: ~$1-2/mês

### Cenário 2: Médio Tráfego
- 10.000 requisições/dia
- 2 segundos por requisição
- **Cloud Run**: ~$10-15/mês
- **Compute Engine e2-small**: $14.40/mês (fixo)

### Cenário 3: Alto Tráfego
- 100.000 requisições/dia
- 1 segundo por requisição
- **Cloud Run**: ~$50-80/mês
- **Compute Engine e2-standard-2**: $48.31/mês (fixo)

## 🎯 Conclusão

**$0.02/hora = $14.40/mês**

Para a maioria dos casos, **Cloud Run é mais econômico** porque:
- ✅ Paga apenas pelo uso real
- ✅ Sem custo quando inativo
- ✅ Escala automaticamente
- ✅ Sem necessidade de gerenciar VMs

**Compute Engine** faz sentido apenas se:
- Você precisa de recursos garantidos 24/7
- Tem tráfego constante e previsível
- Precisa de controle total sobre a infraestrutura

## 📚 Recursos

- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Compute Engine Pricing](https://cloud.google.com/compute/pricing)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
