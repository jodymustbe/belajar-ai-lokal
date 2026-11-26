import json
import os
import urllib.error
import urllib.request
from typing import Iterator

BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = "qwen3.5:2b-q4_K_M"
DEFAULT_NUM_CTX = 8192  # Batas aman 100% GPU untuk qwen3.5:2b di RTX 3050 4 GB
REQUEST_TIMEOUT = 60


def stream_ollama_native(
    messages: list[dict],
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool = False,
) -> Iterator[tuple[str, str]]:
    """Adapter Endpoint /api/chat (NDJSON) dengan pengaturan eksplisit `num_ctx`."""
    url = f"{BASE_URL}/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "think": think,
        "options": {
            "num_ctx": num_ctx,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

            chunk = json.loads(line)
            msg = chunk.get("message", {})

            thinking_tok = msg.get("thinking", "")
            if thinking_tok:
                yield ("thinking", thinking_tok)

            token = msg.get("content", "")
            if token:
                yield ("content", token)

            if chunk.get("done") is True:
                eval_count = chunk.get("eval_count", 0)
                eval_duration = chunk.get("eval_duration", 0)
                if eval_count and eval_duration:
                    tps = (eval_count / eval_duration) * 1_000_000_000
                    yield ("metrics", f"\n[Metrics: {eval_count} token | {tps:.2f} tok/s | ctx={num_ctx}]")
                break


def stream_openai_compat(
    messages: list[dict],
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool = False,
) -> Iterator[tuple[str, str]]:
    """Adapter Endpoint /v1/chat/completions (SSE).

    Catatan: Spesifikasi standar OpenAI tidak memiliki parameter `num_ctx` di payload JSON,
    sehingga endpoint ini mengikuti default server Ollama / `OLLAMA_CONTEXT_LENGTH` / Modelfile.
    """
    _ = num_ctx
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "reasoning_effort": "medium" if think else "none",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue

            data_str = line[len("data: "):]
            if data_str == "[DONE]":
                break

            chunk = json.loads(data_str)
            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            reasoning_tok = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_tok:
                yield ("thinking", reasoning_tok)

            token = delta.get("content")
            if token:
                yield ("content", token)


def main() -> None:
    mode = "ollama"
    num_ctx = DEFAULT_NUM_CTX
    think_enabled = False

    system_msg = {
        "role": "system",
        "content": "Kamu adalah asisten AI yang ringkas, akurat, dan menjawab dalam bahasa Indonesia.",
    }
    messages: list[dict] = [system_msg]

    print(f"=== CLI Chat Loop | Model: {MODEL} | Mode: {mode} | num_ctx: {num_ctx} | Think: {think_enabled} ===")
    print(
        "Perintah: /ctx <angka> (ubah context), /switch (ganti API), "
        "/think (toggle reasoning), /clear (reset memori), /history, /exit\n"
    )

    while True:
        try:
            user_input = input("Anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input.startswith("/ctx"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 256:
                num_ctx = int(parts[1])
                print(f"[Sistem: num_ctx diubah ke {num_ctx} (berlaku di mode /api/chat)]\n")
            else:
                print("[Sistem: Gunakan format `/ctx 4096` atau `/ctx 8192`]\n")
            continue
        if user_input == "/switch":
            mode = "openai" if mode == "ollama" else "ollama"
            endpoint = "/v1/chat/completions" if mode == "openai" else "/api/chat"
            note = " (memakai num_ctx default server/Modelfile)" if mode == "openai" else f" (num_ctx={num_ctx})"
            print(f"[Sistem: Beralih ke endpoint {endpoint}{note}]\n")
            continue
        if user_input == "/think":
            think_enabled = not think_enabled
            print(f"[Sistem: Mode thinking/reasoning {'AKTIF' if think_enabled else 'NONAKTIF'}]\n")
            continue
        if user_input == "/clear":
            messages = [system_msg]
            print("[Sistem: Riwayat percakapan dibersihkan; system prompt tetap terkunci di indeks 0]\n")
            continue
        if user_input == "/history":
            print(json.dumps(messages, indent=2, ensure_ascii=False) + "\n")
            continue

        messages.append({"role": "user", "content": user_input})
        stream_fn = stream_ollama_native if mode == "ollama" else stream_openai_compat

        print("AI: ", end="", flush=True)
        full_reply: list[str] = []
        in_thinking = False
        shown_wait_hint = False

        try:
            for kind, text in stream_fn(messages, num_ctx=num_ctx, think=think_enabled):
                if kind == "thinking":
                    if think_enabled:
                        if not in_thinking:
                            in_thinking = True
                            print("\n[Thinking]\n", end="", flush=True)
                        print(text, end="", flush=True)
                    elif not shown_wait_hint:
                        shown_wait_hint = True
                        print("(sedang berpikir...) ", end="", flush=True)
                elif kind == "content":
                    if in_thinking:
                        in_thinking = False
                        print("\n[/Thinking]\n", end="", flush=True)
                    print(text, end="", flush=True)
                    full_reply.append(text)
                elif kind == "metrics":
                    if in_thinking:
                        in_thinking = False
                        print("\n[/Thinking]", end="", flush=True)
                    print(text, end="", flush=True)
            print("\n")
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace").strip()
            print(f"\n[HTTP Error {err.code}: {detail or err.reason}]\n")
            messages.pop()
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as err:
            reason = getattr(err, "reason", str(err))
            print(f"\n[Error koneksi/stream ke Ollama: {reason}]\n")
            messages.pop()
            continue

        reply_text = "".join(full_reply).strip()
        if reply_text:
            messages.append({"role": "assistant", "content": reply_text})
        else:
            print("[Peringatan: Model tidak menghasilkan teks jawaban akhir. Coba aktifkan /think atau ulangi.]\n")
            messages.pop()


if __name__ == "__main__":
    main()