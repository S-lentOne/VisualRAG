import requests
import json

DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"

class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_URL, timeout: int = 120):
        self.model = model
        self.host = host
        self.timeout = timeout
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            available = [m["name"] for m in r.json().get("models", [])]
            if self.model not in available:
                print(f"[Ollama] Warning: '{self.model}' not found locally. Run: ollama pull {self.model}")
                print(f"[Ollama] Available: {available}")
            else:
                print(f"[Ollama] Connected. Using model: {self.model}")
        except requests.exceptions.ConnectionError:
            print("[Ollama] Not reachable. Make sure Ollama is running: ollama serve")

    def is_reachable(self) -> bool:
        try:
            requests.get(f"{self.host}/api/tags", timeout=5)
            return True
        except requests.exceptions.ConnectionError:
            return False

    def chat(self, messages: list, stream: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
            }
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def chat_stream(self, messages: list):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
            }
        }
        # (connect_timeout, read_timeout) — None means wait as long as it takes for tokens
        with requests.post(f"{self.host}/api/chat", json=payload, stream=True, timeout=(5, None)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    def switch_model(self, model_name: str):
        self.model = model_name
        print(f"[Ollama] Switched to model: {self.model}")