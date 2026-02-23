import os
import json
import logging

# MIGRATED: google-generativeai -> google-genai
GENAI_IMPORT_ERROR = None
genai = None
try:
    from google import genai as google_genai
    genai = google_genai
    print("DEBUG: Successfully imported google.genai (NEW SDK)")
except ImportError as e:
    GENAI_IMPORT_ERROR = str(e)
    print(f"CRITICAL ERROR: Could not import google.genai: {e}")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealLLMClient:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

        self.gemini_configured = False
        self.gemini_client = None
        self.openai_client = None

        if self.gemini_key and genai:
            self.gemini_client = genai.Client(api_key=self.gemini_key)
            self.gemini_configured = True
            logger.info("✅ Gemini Client Configured (NEW SDK)")

        if self.openai_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_key)
            logger.info("✅ OpenAI Client Configured")

    def chat(self, system: str, user: str, provider: str = "gemini", api_keys: dict = {}) -> str:
        """Unified chat interface. Returns raw JSON string response."""
        last_error = "No Real AI Available"

        print(f"DEBUGGING LLM: Provider={provider}")
        print(f"DEBUGGING LLM: Keys Present: {list(api_keys.keys())}")
        if provider == "gemini":
            val = api_keys.get('gemini', '') or os.getenv('GEMINI_API_KEY', '') or ''
            print(f"DEBUGGING LLM: Gemini Key Length: {len(val)}")

        full_prompt = f"{system}\n\nUSER REQUEST:\n{user}\n\nOutput JSON only."

        client_gemini_key = api_keys.get("gemini")
        client_openai_key = api_keys.get("openai")

        if provider == "gemini":
            if not genai:
                last_error = f"Lib Import Fail: {GENAI_IMPORT_ERROR or 'Unknown'}"
                logger.warning(last_error)
            elif client_gemini_key:
                try:
                    return self._call_gemini(full_prompt, api_key=client_gemini_key)
                except Exception as e:
                    logger.error(f"Client Gemini Key Failed: {e}")
                    last_error = f"Gemini Error: {e}"
            elif self.gemini_configured:
                try:
                    return self._call_gemini(full_prompt, api_key=self.gemini_key)
                except Exception as e:
                    last_error = f"Server Gemini Key Error: {e}"
            else:
                last_error = "Client Gemini Key Missing"
                print("DEBUGGING LLM: Client Gemini Key is MISSING/EMPTY")

        elif provider == "openai":
            if client_openai_key and OpenAI:
                try:
                    temp_client = OpenAI(api_key=client_openai_key)
                    return self._call_openai(system, user, client=temp_client)
                except Exception as e:
                    logger.error(f"Client OpenAI Key Failed: {e}")
                    if "429" in str(e):
                        last_error = "OpenAI Quota Exceeded (Check Billing)"
                    else:
                        last_error = f"OpenAI Key Error: {str(e)[:50]}..."
            if self.openai_client:
                try:
                    return self._call_openai(system, user)
                except Exception as e:
                    last_error = f"Server OpenAI Key Error: {e}"

        msg = f"AI Error: {last_error}"
        logger.error(msg)
        raise Exception(msg)

    def _call_gemini(self, prompt: str, api_key: str = None) -> str:
        if not genai:
            raise Exception(f"GenAI Lib Missing: {GENAI_IMPORT_ERROR or 'Unknown'}")

        # Only try models that are known to work (Feb 2026)
        models_to_try = [
            'gemini-3-flash',       # Gemini 3 Flash - primary
            'gemini-2.0-flash',     # Stable fallback
            'gemini-1.5-flash',     # Reliable fallback
        ]

        client = genai.Client(api_key=api_key) if api_key else self.gemini_client
        if not client:
            raise Exception("No Gemini Client Available (No API Key)")

        print(f"DEBUGGING LLM: Using API key length={len(api_key) if api_key else 0}")

        last_model_error = "Unknown"
        for model_name in models_to_try:
            try:
                print(f"DEBUGGING LLM: Trying {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                print(f"DEBUGGING LLM: SUCCESS with {model_name}")
                return self._clean_json(response.text)
            except Exception as e:
                last_model_error = str(e)
                print(f"DEBUGGING LLM: Failed {model_name} -> {e}")
                logger.error(f"Gemini {model_name} Error: {e}")
                continue

        raise Exception(f"All Gemini Models Failed. Last error: {last_model_error}")

    def _call_openai(self, system: str, user: str, client=None) -> str:
        active_client = client or self.openai_client
        if not active_client:
            return None
        response = active_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _clean_json(self, text: str) -> str:
        return text.replace("```json", "").replace("```", "").strip()


def ask_ai(prompt, context=None):
    client = RealLLMClient()
    payload = {"prompt": prompt, "context": context or {}}
    return json.loads(client.chat("You are a helper.", json.dumps(payload)))
