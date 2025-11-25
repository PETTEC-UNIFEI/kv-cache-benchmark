# KV Cache Benchmark - PETTEC

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0%2B-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Benchmark de diferentes estratégias de gerenciamento de KV Cache para modelos de linguagem (LLMs). Este repositório reúne experimentos, análises, notebook tutorial e materiais do artigo apresentado no **Simpósio Unifei 2025** pelos alunos do PETTEC.

## 📋 Sobre o Projeto

Este projeto apresenta uma análise comparativa detalhada de três estratégias de gerenciamento de KV Cache em modelos de linguagem:

- **Sem Cache**: Recalcula todos os estados intermediários a cada geração
- **Dynamic Cache**: Aloca memória de forma flexível conforme o histórico cresce
- **Static Cache**: Pré-aloca área fixa de memória para máxima velocidade

### 🎯 Objetivos

- Medir e comparar o desempenho de diferentes estratégias de cache
- Avaliar trade-offs entre velocidade, uso de memória e robustez
- Fornecer insights práticos para aplicações reais de LLMs
- Disponibilizar ferramenta de benchmark reproduzível

## 🔬 Métricas Avaliadas

O benchmark analisa as seguintes métricas:

| Métrica | Descrição | Importância |
|---------|-----------|-------------|
| **Tempo de Geração** | Tempo total para produzir resposta | Crítico para aplicações em tempo real |
| **Uso de Memória** | Quantidade de RAM/VRAM utilizada | Essencial para escalabilidade |
| **Throughput** | Tokens gerados por segundo | Importante para alto volume de requisições |
| **Taxa de Sucesso** | Proporção de respostas sem erro | Indica robustez do sistema |

## 🚀 Começando

### Pré-requisitos

- Python 3.8 ou superior
- GPU NVIDIA com suporte CUDA (recomendado) ou CPU
- 8GB+ de RAM (16GB+ recomendado)

### Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/kv-cache-benchmark.git
cd kv-cache-benchmark
```

2. Crie um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install numpy matplotlib psutil jupyter
```

### Uso Rápido

1. Abra o notebook tutorial:

```bash
jupyter notebook "Benchmark de Estratégias de KV Cache - PETTEC.ipynb"
```

2. Execute as células sequencialmente para:
   - Carregar o modelo (Llama 3.2-1B por padrão)
   - Executar benchmarks com diferentes estratégias
   - Visualizar resultados comparativos

### Exemplo de Código

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Inicializar modelo
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Criar instância do benchmark
benchmark = KVCacheBenchmark(model, tokenizer, device, model_name)

# Executar benchmark
results = benchmark.run_conversational_benchmark(
    cache_strategies=["none", "dynamic", "static"],
    scenario="insurance_claim_auto",
    num_turns=5,
    max_new_tokens=150
)

# Analisar resultados
benchmark.analyze_results(results)
```

## 📊 Cenários de Teste

O benchmark inclui cenários conversacionais realistas:

### Seguros (Insurance Support)
- `insurance_claim_auto`: Processo de sinistro automotivo
- `insurance_policy_update`: Atualização de apólice residencial
- `insurance_life_beneficiary`: Alteração de beneficiários

### Bancário (Banking Assistant)
- `banking_open_account`: Abertura de conta corrente
- `banking_loan_application`: Solicitação de empréstimo pessoal

### E-commerce (E-commerce Support)
- `ecommerce_support`: Suporte ao cliente de loja online

Cada cenário contém 15 turnos de conversação com contexto crescente, simulando interações reais.

## 🏗️ Estrutura do Projeto

```
kv-cache-benchmark/
├── Benchmark de Estratégias de KV Cache - PETTEC.ipynb  # Notebook principal
├── Benchmark de Estratégias de KV Cache - PETTEC.pdf    # Versão PDF do notebook
├── Simpósio Unifei 2025 Estratégias de Gerenciamento de KV Cache.pdf  # Artigo completo
├── Gerenciamento de KV-Cache - Simposio_2025.pdf       # Material adicional
├── README.md                                             # Este arquivo
└── LICENSE                                               # Licença MIT
```

## 📚 Documentação e Referências

### Recursos Oficiais
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Llama Model Cards](https://huggingface.co/meta-llama)

### Artigos Relacionados
- Consulte o artigo completo no arquivo `Simpósio Unifei 2025 Estratégias de Gerenciamento de KV Cache.pdf`
- Material adicional em `Gerenciamento de KV-Cache - Simposio_2025.pdf`


## 👥 Autores

**PETTEC** - Programa de Educação Tutorial em Tecnologia e Engenharia de Computação

- Universidade Federal de Itajubá (UNIFEI)
- Simpósio Unifei 2025

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

