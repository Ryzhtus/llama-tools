
### LLM
```bash
docker build -t llm .
docker run --gpus=all -d --name llm_app -p 8001:8001 llm
```

### Retrieval

```bash
docker build -t retrieval .
docker run -d --name retrieval_app -p 8002:8002 retrieval
```

### App
```bash
docker build -t app .
docker run -d --name main_app -p 8501:8501 app
```