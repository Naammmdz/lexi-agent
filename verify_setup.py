"""Quick setup verification script"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

print("🔍 Verifying R2AI Setup...\n")

checks: dict[str, bool] = {
    "R2AIStage1DATA.json exists": Path("R2AIStage1DATA.json").exists(),
    "test_r2ai_pipeline.py exists": Path("test_r2ai_pipeline.py").exists(),
}

try:
    from config import Config

    checks["Config loads successfully"] = Config.USE_LOCAL_LLM is True
    print(f"   LLM backend: {getattr(Config, 'LLM_BACKEND', 'ollama')}")
    print(f"   LLM model: {Config.MODEL_GEN}")
    ollama_url = f"{Config.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(ollama_url, timeout=3) as resp:
            tags = json.loads(resp.read().decode("utf-8")).get("models", [])
        names = {m.get("name", "") for m in tags}
        model = Config.OLLAMA_MODEL
        checks["Ollama running"] = True
        checks[f"Ollama has {model}"] = any(
            n == model or n.startswith(f"{model}:") or model in n for n in names
        )
        if not checks[f"Ollama has {model}"]:
            print(f"   ⚠️  Run: ollama pull {model}")
    except Exception:
        checks["Ollama running"] = False
        checks[f"Ollama has {getattr(Config, 'OLLAMA_MODEL', 'qwen3:4b-instruct')}"] = False
        print("   ⚠️  Ollama not reachable — start with: ollama serve")
except Exception as e:
    checks["Config loads successfully"] = False
    print(f"   ❌ Config error: {e}")

checks["ollama CLI installed"] = shutil.which("ollama") is not None

try:
    with open("R2AIStage1DATA.json") as f:
        data = json.load(f)
    print(f"\n✅ Dataset: {len(data)} questions")
    print(f"   Sample Q1: {data[0]['question'][:60]}...")
except Exception as e:
    print(f"❌ Dataset error: {e}")

print("\n" + "=" * 60)
for check, result in checks.items():
    symbol = "✅" if result else "❌"
    print(f"{symbol} {check}")
print("=" * 60)

required = {
    "R2AIStage1DATA.json exists",
    "test_r2ai_pipeline.py exists",
    "Config loads successfully",
    "ollama CLI installed",
}
if required.issubset(checks) and all(checks[k] for k in required):
    print("\n✨ Core setup OK. Pull LLM if missing: ollama pull qwen3:4b-instruct")
else:
    print("\n⚠️  Some checks failed. Please fix before running pipeline.")
    sys.exit(1)
