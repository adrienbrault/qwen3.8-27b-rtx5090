#!/usr/bin/env python3
"""killer 8x24K burst + vision 8x4x2048^2 burst + json_schema x4 against an OpenAI endpoint (reasoning-aware)."""
import base64, json, random, struct, sys, time, urllib.request, zlib, concurrent.futures as cf
url = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
def call(body, t=900):
    d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=t))
    m = d["choices"][0]["message"]; return (m.get("content") or "").strip(), (m.get("reasoning") or m.get("reasoning_content") or "").strip()
W = "alpha bravo charlie delta echo foxtrot golf hotel".split()
def killer(i):
    rng = random.Random(f"k-{time.time()}-{i}"); body = " ".join(rng.choice(W) for _ in range(int(24000 / 1.3)))
    try: c, r = call({"model": model, "max_tokens": 400, "temperature": 0.6, "messages": [{"role": "user", "content": body + "\n\nReply with the single word OK."}]}); return bool(c or r)
    except Exception as e: print("killer err", repr(e)[:100]); return False
with cf.ThreadPoolExecutor(8) as ex: print("KILLER %d/8" % sum(ex.map(killer, range(8))), flush=True)
Wd = H = 2048
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
raw = bytearray()
for y in range(H): raw.append(0); raw += bytes(v for x in range(Wd) for v in ((x * 255) // Wd, (y * 255) // H, ((x + y) * 255) // (Wd + H)))
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", Wd, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
def vision(j):
    try: c, r = call({"model": model, "max_tokens": 400, "temperature": 0.6, "messages": [{"role": "user", "content": [{"type": "text", "text": f"Stream {j}: describe the four images briefly."}] + [img] * 4}]}, 600); return bool(c or r)
    except Exception as e: print("vision err", repr(e)[:100]); return False
with cf.ThreadPoolExecutor(8) as ex: print("VISION %d/8" % sum(ex.map(vision, range(1, 9))), flush=True)
schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}, "landmarks": {"type": "array", "items": {"type": "string"}}}, "required": ["city", "population", "landmarks"], "additionalProperties": False}
ok = 0
for q in ("Paris", "Tokyo", "Lima", "Oslo"):
    try: c, _ = call({"model": model, "max_tokens": 400, "temperature": 0.6, "messages": [{"role": "user", "content": f"Give facts about {q} as JSON."}], "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "schema": schema, "strict": True}}}, 300); ok += set(json.loads(c)) == {"city", "population", "landmarks"}
    except Exception as e: print("so err", repr(e)[:100])
print("SO %d/4" % ok, flush=True)
