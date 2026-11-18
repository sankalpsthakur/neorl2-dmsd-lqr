#!/usr/bin/env python3
"""Static checks of the generated kernels (nothing is run, nothing is pushed). Exit 1 on any failure.
metadata (private, CPU only, internet, name, sources), syntax, undefined names (module-wide over-approximation), forbidden strings (secrets, pushes),
embedded package == current src/, CFG parses, final carries a frozen sha (or the placeholder)."""
import ast, base64, builtins, json, pathlib, re, sys, zlib, hashlib
root = pathlib.Path(__file__).resolve().parent.parent
fail = 0


def bad(msg):
    global fail
    fail = 1; print("FAIL ", msg)


def ok(msg):
    print("PASS ", msg)


def undefined_names(tree):
    bound = set(dir(builtins)) | {"CFG", "__file__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)): bound.add(n.name)
        if isinstance(n, ast.arg): bound.add(n.arg)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): bound.add(n.id)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: bound.add((a.asname or a.name).split(".")[0])
        if isinstance(n, ast.ExceptHandler) and n.name: bound.add(n.name)
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound})


for k in ("dryrun", "prep", "g0", "final"):
    kd = root / "kernels" / k
    meta = json.loads((kd / "kernel-metadata.json").read_text()); src = (kd / "script.py").read_text()
    name = f"bench-neorl2-dmsd-{k}"
    good = (meta["id"] == f"sankalpsthakur/{name}" and meta["title"] == name and str(meta["is_private"]).lower() == "true" and str(meta["enable_gpu"]).lower() == "false"
            and str(meta["enable_tpu"]).lower() == "false" and meta["code_file"] == "script.py" and meta["kernel_type"] == "script" and not meta["dataset_sources"] and not meta["competition_sources"] and not meta["model_sources"])
    (ok if good else bad)(f"{k}: metadata private, CPU only, named {name}, no dataset/competition/model sources")
    want_src = {"dryrun": [], "prep": [], "g0": ["sankalpsthakur/bench-neorl2-dmsd-prep"], "final": ["sankalpsthakur/bench-neorl2-dmsd-prep", "sankalpsthakur/bench-neorl2-dmsd-g0"]}[k]
    (ok if meta["kernel_sources"] == want_src else bad)(f"{k}: kernel_sources {meta['kernel_sources']}")
    try:
        tree = ast.parse(src); ok(f"{k}: syntax")
    except SyntaxError as e:
        bad(f"{k}: syntax {e}"); continue
    und = undefined_names(tree)
    (ok if not und else bad)(f"{k}: undefined names {und}")
    for pat in ("UserSecretsClient", "HF_TOKEN", "os.environ", "getpass", "kernels push", "subprocess", "api.kaggle", "Authorization", "login(", "upload_file", "push_to_hub", "huggingface_hub"):
        if pat in src: bad(f"{k}: forbidden string {pat!r}")
    m = re.search(r'_PKG = json.loads\(zlib.decompress\(base64.b64decode\("([^"]+)"\)\).decode\(\)\)', src)
    pkg = json.loads(zlib.decompress(base64.b64decode(m.group(1))).decode())
    cur = {p.name: p.read_text() for p in sorted((root / "src" / "nd").glob("*.py"))}
    cur["sources.sha256"] = (root / "protocol" / "sources.sha256").read_text(); cur["data_manifest.json"] = (root / "protocol" / "data_manifest.json").read_text()
    for pat in ("UserSecretsClient", "HF_TOKEN", "os.environ", "getpass", "kernels push", "subprocess", "Authorization", "upload_file", "push_to_hub", "huggingface_hub"):
        if any(pat in v for v in pkg.values()): bad(f"{k}: forbidden string {pat!r} in the embedded package")
    ok(f"{k}: no secret read, no push, no upload strings (script and embedded package)")
    (ok if pkg == cur else bad)(f"{k}: embedded package equals src/nd + protocol files")
    body = src.split("# ---------------------------------------------------------------- main.py\n", 1)[1]
    (ok if body == (kd / "main.py").read_text() else bad)(f"{k}: script body equals main.py")
    c = re.search(r"^CFG = json.loads\((.*)\)$", src, re.M); cfg = json.loads(eval(c.group(1))); ok(f"{k}: CFG {cfg}")
    (ok if int(cfg.get("wall_cap_s", 0)) in (900, 1800, 3600) and "arm_wall_cap" in body else bad)(f"{k}: in-kernel wall cap armed ({cfg.get('wall_cap_s')} s)")
    if k == "final":
        (ok if ("N = 1000; SEEDS = (7, 42, 210)" in body and 'CFG["n"]' not in body and "n" not in cfg) else bad)("final: n = 1000 and seeds (7, 42, 210) hard-coded in the script, not in CFG")
        s = cfg.get("frozen_sha256", ""); (ok if (len(s) == 64 or s == "SET_AFTER_PREP") else bad)(f"final: frozen_sha256 field is {s[:16]}...")
    print(f"      {k} script sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}")
sys.exit(fail)
