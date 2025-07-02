# ASESeminar-LangGraph

### 1. Clone repository into virutal environment

### 2. Install dependencies

Install LangGraph:

```
pip install -U langgraph
pip install -U requests
pip install -U typing_extensions
```

### 3. Set OpenAI-Key in asegraph.py

```
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
```

### 4. Run asegraph.py

- SWE-Bench-Lite API is expected to run at port 8081
- SWE-Bench-Lite Tester is expected to run at port 8082
