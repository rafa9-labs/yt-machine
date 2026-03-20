# Ollama Setup Guide

## Installation

### Windows
1. Download Ollama from: https://ollama.ai/download
2. Run the installer
3. Ollama will start automatically as a service on `http://localhost:11434`

### Verify Installation
```powershell
ollama --version
```

## Pull DeepSeek Model

### Option 1: DeepSeek R1 (Recommended for reasoning)
```powershell
ollama pull deepseek-r1:latest
```

### Option 2: DeepSeek V3 (Faster, good for production)
```powershell
ollama pull deepseek-v3:latest
```

### Option 3: Fallback (Llama 3.2 - if DeepSeek unavailable)
```powershell
ollama pull llama3.2:latest
```

## Test the Model

```powershell
ollama run deepseek-r1:latest "Explain quantum computing in one sentence"
```

## Check Running Models

```powershell
ollama list
```

## API Endpoint

Once running, Ollama exposes a REST API at:
- **Base URL**: `http://localhost:11434`
- **Generate endpoint**: `http://localhost:11434/api/generate`
- **Chat endpoint**: `http://localhost:11434/api/chat`

## Memory Requirements

- **DeepSeek R1**: ~16GB RAM (8GB VRAM if using GPU)
- **DeepSeek V3**: ~12GB RAM
- **Llama 3.2**: ~8GB RAM

## Troubleshooting

### Ollama not responding
```powershell
# Restart Ollama service
Stop-Process -Name "ollama" -Force
ollama serve
```

### Check if Ollama is running
```powershell
curl http://localhost:11434/api/tags
```

### GPU Acceleration (NVIDIA)
Ollama automatically uses GPU if CUDA is available. Check with:
```powershell
nvidia-smi
```

## Next Steps

After installation:
1. Run `python test_llm.py` to verify the LLM interface works
2. Check `system_prompts.json` for prompt configurations
3. Proceed to Phase 2 (Redfish implementation)
