#!/usr/bin/env python3
"""Offline checks: python3 -m unittest discover -s scripts/tests -p 'test_build_v031.py'."""

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
ARCTIC_HASH = "36857e2fbe3ead34821a04be98dfab95086050c1e38199c8fdfe308234fdcf63"
SOURCE_HASH = "e20baf18e40f88446f745999379db04c8f88c173dc7afde5dc6b869c06299ad0"
VLLM_TAG = "vllm/vllm-openai:v0.31.0"
CUDA_TAG = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
VLLM_DIGEST = "sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
CUDA_DIGEST = "sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266"
VLLM_REF = "vllm/vllm-openai@" + VLLM_DIGEST
CUDA_REF = "nvidia/cuda@" + CUDA_DIGEST

FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os
from pathlib import Path
import shlex, shutil, subprocess, sys
args = sys.argv[1:]
state = Path(os.environ["FAKE_STATE"])
with (state / "docker.jsonl").open("a") as stream:
    stream.write(json.dumps({"args": args, "builder": os.environ.get("DOCKER_BUILDKIT")}) + "\n")
pins = {
    "vllm/vllm-openai:v0.31.0": "sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3",
    "nvidia/cuda:13.0.3-devel-ubuntu24.04": "sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266",
}
refs = {tag: tag.rsplit(":", 1)[0] + "@" + digest for tag, digest in pins.items()}
ids = dict(pins) if os.environ.get("FAKE_STORE", "containerd") == "containerd" else {
    tag: "sha256:" + str(i) * 64 for i, tag in enumerate(pins, 1)
}
if args[:2] == ["image", "inspect"]:
    target = args[-1]
    tag = next((tag for tag in pins if target in (tag, refs[tag])), None)
    if tag is None:
        assert target == "vllm-qwen38:v0310-r27c-A_S-L1a-ubplan", args
        print("{}")
        sys.exit(0)
    marker = state / ("pulled-vllm" if tag.startswith("vllm/") else "pulled-cuda")
    user_tag = target == tag and os.environ.get("FAKE_EXISTING_TAG_WRONG") == "1"
    if os.environ.get("FAKE_MISSING_BASE") == "1" and not marker.exists() and not user_tag:
        sys.exit(1)
    if "--format" in args:
        fmt = args[args.index("--format") + 1]
        if fmt == "{{.Id}}":
            print("sha256:" + "0" * 64 if user_tag else ids[tag])
        elif fmt == "{{.Os}}/{{.Architecture}}":
            print(os.environ.get("FAKE_PLATFORM", "linux/amd64"))
        elif fmt == "{{range .RepoDigests}}{{println .}}{{end}}":
            # Include another digest first, so membership must inspect the whole list.
            print(tag.rsplit(":", 1)[0] + "@sha256:" + "f" * 64)
            wrong = os.environ.get("FAKE_WRONG_REPODIGEST")
            if user_tag or wrong == "digest":
                print(tag.rsplit(":", 1)[0] + "@sha256:" + "0" * 64)
            elif wrong == "name":
                print("other/image@" + pins[tag])
            elif wrong == "suffix":
                print(refs[tag] + "0")
            else:
                print(refs[tag])
        else:
            raise AssertionError(args)
    else:
        print("{}")
elif args[0] == "pull":
    assert args == ["pull", "--platform", "linux/amd64", refs[next(tag for tag in pins if refs[tag] == args[-1])]], args
    (state / ("pulled-vllm" if args[-1].startswith("vllm/") else "pulled-cuda")).touch()
elif args[0] == "build":
    assert os.environ["DOCKER_BUILDKIT"] == "0"
    assert "--network=none" in args and "--pull=false" in args
    tag = args[args.index("-t") + 1]
    if "A-nvfp4kv-mtp" in tag:
        assert "VLLM_BASE=" + ids["vllm/vllm-openai:v0.31.0"] in args, args
        assert "CUDA_TOOLCHAIN=" + ids["nvidia/cuda:13.0.3-devel-ubuntu24.04"] in args, args
        shutil.copytree(args[-1], state / "context")
    if "A_S-BA4-L1a-ubplan" in tag:
        shutil.copytree(args[-1], state / "ba4-context")
    if os.environ.get("FAKE_BUILD_FAIL") and "A_S-splitkv" in tag:
        sys.exit(17)
    print("fake build " + tag)
elif args[0] == "run":
    image = args[args.index("--entrypoint") + 2]
    final = ("vllm-qwen38:v0310-r27c-A_S-L1a-ubplan", "vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan")
    if "generate" in args or "--runtime" in args:
        assert image in final, args
    else:
        assert image == ids["vllm/vllm-openai:v0.31.0"], args
    if "wheel" in args:
        # The clean-room base lacks CMake; the former direct wheel call must fail.
        print("error: [Errno 2] No such file or directory: 'cmake'", file=sys.stderr)
        sys.exit(1)
    elif "/inputs/arctic-build-requirements.txt" in args[-1]:
        assert "--rm" in args and "--network=bridge" in args, args
        assert args[args.index("--user") + 1] == "0", args
        assert "PIP_CONFIG_FILE=/dev/null" in args, args
        assert "ARCTIC_INFERENCE_PRECOMPILED_OPS=0" in args, args
        inputs = Path(next(a for a in args if a.endswith(":/inputs:ro"))[:-len(":/inputs:ro")])
        wheels = Path(next(a for a in args if a.endswith(":/wheels"))[:-len(":/wheels")])
        shutil.copyfile(inputs / "arctic-build-requirements.txt", state / "wheel-requirements.txt")
        tool_bin = state / "wheel-bin"
        tool_bin.mkdir()
        tool = tool_bin / "python3"
        tool.write_text("#!" + sys.executable + "\n" + os.environ["FAKE_WHEEL_PIP"])
        tool.chmod(0o755)
        # Run the actual container shell body with pip simulated at the process boundary.
        code = args[-1].replace("/inputs", shlex.quote(str(inputs))).replace("/wheels", shlex.quote(str(wheels)))
        env = dict(os.environ, PATH=str(tool_bin) + os.pathsep + os.environ["PATH"])
        sys.exit(subprocess.run(["bash", "-euo", "pipefail", "-c", code], env=env).returncode)
    elif "generate" in args:
        assert "--network=none" in args
        compile(sys.stdin.read(), "manifest-stdin", "exec")
        rows = Path(os.environ["FAKE_MANIFEST"]).read_text().splitlines()
        if image == final[1]:
            # The BA4 image: the layer's three patched vLLM files at their pinned outputs (FAKE_BA4_CHANGE keeps one stale).
            after = json.loads(Path(os.environ["FAKE_BA4_MANIFEST"]).read_text())["after"]
            stale = "vllm/envs.py" if os.environ.get("FAKE_BA4_CHANGE") else None
            rows = [after[row[66:]] + row[64:] if row[66:] in after and row[66:] != stale else row for row in rows]
        change = os.environ.get("FAKE_CONTENT_CHANGE")
        if change in (".py", ".so"):
            i = next(i for i, row in enumerate(rows) if row.endswith(change))
            rows[i] = "0" * 64 + rows[i][64:]
        elif change == "missing":
            rows.pop(next(i for i, row in enumerate(rows) if row.endswith(".so")))
        elif change == "extra":
            rows.append("0" * 64 + "  vllm-sm12x/extra.o")
        print("\n".join(rows))
    elif "--runtime" in args:
        compile(args[args.index("-c") + 1], "identity-code", "exec")
        if os.environ.get("FAKE_IDENTITY_FAIL"):
            sys.exit(18)
        print("IDENTITY OK fixture")
    elif "apt-get update" in args[-1]:
        assert "--network=none" not in args
        mount = next(a for a in args if a.endswith(":/debs"))
        (Path(mount[:-len(":/debs")]) / "fixture.deb").write_bytes(b"fixture deb")
    else:
        assert "--network=none" in args
        if os.environ.get("FAKE_MISSING_TOOLS"):
            sys.exit(1)
else:
    raise AssertionError(args)
'''

FAKE_WHEEL_PIP = r'''import json, os
from pathlib import Path
import sys
args = sys.argv[1:]
state = Path(os.environ["FAKE_STATE"])
with (state / "wheel-pip.jsonl").open("a") as stream:
    stream.write(json.dumps(args) + "\n")
assert args[:3] == ["-m", "pip", "--isolated"], args
if args[3] == "install":
    for flag in ("--require-hashes", "--only-binary=:all:", "--no-deps", "--no-cache-dir", "--force-reinstall"):
        assert flag in args, args
    assert args[args.index("--index-url") + 1] == "https://pypi.org/simple", args
    assert "--extra-index-url" not in args and "--trusted-host" not in args, args
    requirements = Path(args[args.index("-r") + 1]).read_text()
    for name in ("cmake", "ninja", "nanobind", "setuptools", "wheel", "protobuf", "grpcio-tools", "grpcio"):
        assert name + "==" in requirements, requirements
    assert "torch==" not in requirements
    if os.environ.get("FAKE_BUILD_TOOLS_FAIL"):
        sys.exit(19)
    (state / "wheel-tools-installed").touch()
elif args[3] == "wheel":
    for flag in ("--no-index", "--no-deps", "--no-build-isolation", "--no-cache-dir"):
        assert flag in args, args
    assert "--index-url" not in args, args
    if not (state / "wheel-tools-installed").exists():
        print("error: [Errno 2] No such file or directory: 'cmake'", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("FAKE_WHEEL_FAIL"):
        sys.exit(20)
    if not os.environ.get("FAKE_NO_WHEEL"):
        dest = Path(args[args.index("--wheel-dir") + 1])
        (dest / "arctic_inference-0.1.1-cp312-cp312-linux_x86_64.whl").write_bytes(b"fixture wheel")
else:
    raise AssertionError(args)
'''


FAKE_CURL = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import shutil, sys
args = sys.argv[1:]
state = Path(os.environ["FAKE_STATE"])
with (state / "curl.txt").open("a") as stream:
    stream.write(args[-1] + "\n")
dest = Path(args[args.index("--output") + 1])
src = state / ("source.tar.gz" if "codeload.github.com" in args[-1] else "sdist.tar.gz")
shutil.copyfile(src, dest)
if "files.pythonhosted.org" in args[-1] and os.environ.get("FAKE_CORRUPT_SDIST"):
    dest.write_bytes(b"corrupt sdist")
'''


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v031-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copytree(ROOT / "patches-v0310", self.repo / "patches-v0310")
        for name in ("build-v031-image.sh", "serve-v031-l1a-daily.sh", "serve-v031-ba4-daily.sh", "v031_manifest.py"):
            shutil.copy2(ROOT / "scripts" / name, self.repo / "scripts" / name)
        shutil.copy2(ROOT / "scripts/arctic-build-requirements.txt", self.repo / "scripts/arctic-build-requirements.txt")
        self.state = self.base / "state"
        self.state.mkdir()
        with tarfile.open(self.state / "source.tar.gz", "w:gz") as archive:
            for name in ("csrc/core/registration.h", "cmake/patches/pytorch_stable_string.patch"):
                body = b"fixture source\n"
                item = tarfile.TarInfo("vllm-" + COMMIT + "/" + name)
                item.size = len(body)
                archive.addfile(item, io.BytesIO(body))
        # Substitute synthetic archive pins only in the isolated fixture checkout.
        sdist = b"fixture sdist"
        (self.state / "sdist.tar.gz").write_bytes(sdist)
        script = self.repo / "scripts/build-v031-image.sh"
        text = script.read_text()
        self.assertEqual(text.count(ARCTIC_HASH), 1)
        self.assertEqual(text.count(SOURCE_HASH), 1)
        text = text.replace(ARCTIC_HASH, hashlib.sha256(sdist).hexdigest())
        script.write_text(text.replace(SOURCE_HASH, hashlib.sha256((self.state / "source.tar.gz").read_bytes()).hexdigest()))
        self.bin = self.base / "bin"
        self.bin.mkdir()
        for name, body in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL)):
            path = self.bin / name
            path.write_text(body)
            path.chmod(0o755)
        (self.base / "temp").mkdir()

    def run_build(self, expected=0, **extra):
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   DOCKER="docker", DRY_RUN="0", CHECK="0", NO_CACHE="0", OVERLAY_JOBS="4",
                   LOG_DIR=str(self.base / "logs"), TMPDIR=str(self.base / "temp"),
                   FAKE_STATE=str(self.state), FAKE_WHEEL_PIP=FAKE_WHEEL_PIP, FAKE_MANIFEST=str(self.repo / "patches-v0310/served-l1a-manifest.txt"),
                   FAKE_BA4_MANIFEST=str(self.repo / "patches-v0310/ba4/manifest.json"))
        env.pop("VLLM_SOURCE_SHA256", None)
        env.update(extra)
        result = subprocess.run(["bash", str(self.repo / "scripts/build-v031-image.sh")],
                                cwd=self.base, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, expected, result.stdout)
        self.assertEqual(list((self.base / "temp").iterdir()), [], "temporary build context leaked")
        return result.stdout

    def docker_calls(self):
        path = self.state / "docker.jsonl"
        return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []

    def assert_digest_references_only(self):
        for call in self.docker_calls():
            args = call["args"]
            self.assertIn(args[0], ("image", "pull", "build", "run"))
            if args[0] in ("image", "pull"):
                self.assertNotIn(args[-1], (VLLM_TAG, CUDA_TAG))

    def test_dry_run_uses_no_docker_or_network(self):
        output = self.run_build(DRY_RUN="1", CHECK="1", NO_CACHE="1", OVERLAY_JOBS="2")
        self.assertFalse(self.docker_calls())
        self.assertFalse((self.state / "curl.txt").exists())
        builds = [row for row in output.splitlines() if row.startswith("DOCKER_BUILDKIT=0 docker build ")]
        self.assertEqual(len(builds), 3)
        for row in builds:
            self.assertIn("--network=none --pull=false --no-cache", row)
        self.assertIn("OVERLAY_JOBS=2", builds[0])
        self.assertIn("v0310-r27c-A-nvfp4kv-mtp", builds[0])
        self.assertIn("v0310-r27c-A_S-splitkv", builds[1])
        self.assertIn("v0310-r27c-A_S-L1a-ubplan", builds[2])
        self.assertIn("--runtime nvidia", output)
        self.assertIn(VLLM_REF, output)
        self.assertIn(CUDA_REF, output)
        self.assertIn("local .Id", output)
        self.assertIn("--network=bridge", output)
        self.assertIn("--require-hashes", output)
        self.assertIn("--only-binary=:all:", output)
        self.assertIn("https://pypi.org/simple", output)
        self.assertIn("arctic-build-requirements.txt", output)
        self.assertIn("--no-build-isolation", output)
        self.assertIn("DRY_RUN DONE", output)

    def test_wheel_build_installs_tools_before_building_without_isolation(self):
        self.run_build()
        pip_calls = [json.loads(row) for row in (self.state / "wheel-pip.jsonl").read_text().splitlines()]
        self.assertEqual([args[3] for args in pip_calls], ["install", "wheel"])
        self.assertEqual((self.state / "wheel-requirements.txt").read_bytes(),
                         (self.repo / "scripts/arctic-build-requirements.txt").read_bytes())
        ctx = self.state / "context"
        self.assertFalse((ctx / "downloads").exists())
        self.assertEqual([p.name for p in (ctx / "assets/wheels").iterdir()],
                         ["arctic_inference-0.1.1-cp312-cp312-linux_x86_64.whl"])

    def test_failed_build_tool_install_stops_before_wheel_and_image_builds(self):
        output = self.run_build(expected=1, FAKE_BUILD_TOOLS_FAIL="1")
        self.assertIn("Arctic wheel preparation failed", output)
        self.assertFalse((self.state / "wheel-tools-installed").exists())
        pip_calls = [json.loads(row) for row in (self.state / "wheel-pip.jsonl").read_text().splitlines()]
        self.assertEqual([args[3] for args in pip_calls], ["install"])
        self.assertFalse(any(c["args"][0] == "build" for c in self.docker_calls()))

    def test_failed_wheel_build_stops_before_image_builds(self):
        output = self.run_build(expected=1, FAKE_WHEEL_FAIL="1")
        self.assertIn("Arctic wheel preparation failed", output)
        self.assertTrue((self.state / "wheel-tools-installed").exists())
        self.assertFalse(any(c["args"][0] == "build" for c in self.docker_calls()))

    def test_missing_wheel_stops_before_image_builds(self):
        output = self.run_build(expected=1, FAKE_NO_WHEEL="1")
        self.assertIn("pip wheel produced no Arctic wheel", output)
        self.assertFalse(any(c["args"][0] == "build" for c in self.docker_calls()))

    def test_complete_fake_build_and_staged_inputs(self):
        output = self.run_build(CHECK="1")
        self.assertIn("CONTENT PASS", output)
        self.assertIn("IDENTITY OK", output)
        calls = self.docker_calls()
        self.assert_digest_references_only()
        self.assertFalse(any(call["args"][0] == "pull" for call in calls))
        self.assertEqual(sum(call["args"][0] == "build" for call in calls), 3)
        self.assertFalse(any("apt-get update" in call["args"][-1] for call in calls))
        ctx = self.state / "context"
        self.assertEqual(list((ctx / "assets/debs").iterdir()), [])
        self.assertTrue((ctx / "vllm-v0.31.0/csrc/core/registration.h").is_file())
        self.assertTrue((ctx / "out/A/apply_layer.py").is_file())
        staged = (ctx / "out/A/patches-v0310/0136-opt-in-nvfp4-prefill-split-kv-v0310.diff").read_text()
        self.assertIn("lambda _kv_data_type: False", staged)
        self.assertNotIn("lambda *_args", staged)
        self.assertIn("@@ -164,6 +164,35 @@", staged)
        self.assertIn("@@ -903,6 +930,7 @@", staged)
        original = (self.repo / "patches-v0310/0136-opt-in-nvfp4-prefill-split-kv-v0310.diff").read_text()
        self.assertIn("lambda *_args, **_kwargs: False", original)
        for name in ("Dockerfile.A_S-splitkv", "Dockerfile.A_S-L1a-ubplan"):
            rendered = (ctx / name).read_text()
            self.assertNotIn("<<'PY'", rendered)
            # Parse and compile the rendered Python; its shell quoting must preserve the complete body.
            line = next(line for line in rendered.splitlines() if "python3 -c " in line and "exec(" in line)
            tokens = shlex.split(line[line.index("python3 -c "):])
            code = tokens[2]
            compile(code, name, "exec")
            body = ast.literal_eval(code[len("exec("):-1])
            original_body = (self.repo / "patches-v0310" / name).read_text().split("<<'PY'\n", 1)[1].rsplit("PY\n", 1)[0]
            self.assertEqual(body, original_body)
            self.assertIn("sha256sum -c -", rendered)
        self.assertTrue((self.base / "logs/content-report.txt").is_file())

    def test_absent_bases_are_pulled_once(self):
        self.run_build(FAKE_MISSING_BASE="1")
        pulls = [c["args"] for c in self.docker_calls() if c["args"][0] == "pull"]
        self.assertEqual(pulls, [["pull", "--platform", "linux/amd64", ref] for ref in (VLLM_REF, CUDA_REF)])
        self.assert_digest_references_only()

    def test_classic_store_uses_resolved_ids_for_builds_and_all_base_runs(self):
        output = self.run_build(FAKE_STORE="classic", FAKE_MISSING_TOOLS="1", CHECK="1")
        self.assertIn("CONTENT PASS", output)
        self.assertIn("IDENTITY OK", output)
        self.assert_digest_references_only()
        self.assertFalse(any(c["args"][0] == "pull" for c in self.docker_calls()))

    def test_classic_store_pulls_missing_digests_without_replacing_existing_tags(self):
        self.run_build(FAKE_STORE="classic", FAKE_MISSING_BASE="1", FAKE_EXISTING_TAG_WRONG="1")
        pulls = [c["args"] for c in self.docker_calls() if c["args"][0] == "pull"]
        self.assertEqual(pulls, [["pull", "--platform", "linux/amd64", ref] for ref in (VLLM_REF, CUDA_REF)])
        self.assert_digest_references_only()

    def test_existing_other_tags_are_ignored_when_pinned_digests_are_local(self):
        self.run_build(FAKE_STORE="classic", FAKE_EXISTING_TAG_WRONG="1")
        self.assertFalse(any(c["args"][0] == "pull" for c in self.docker_calls()))
        self.assert_digest_references_only()

    def test_wrong_repo_digest_is_refused_after_digest_pull(self):
        for wrong in ("digest", "name", "suffix"):
            with self.subTest(wrong=wrong):
                (self.state / "docker.jsonl").unlink(missing_ok=True)
                output = self.run_build(expected=1, FAKE_WRONG_REPODIGEST=wrong)
                self.assertIn("RepoDigests mismatch", output)
                pulls = [c["args"] for c in self.docker_calls() if c["args"][0] == "pull"]
                self.assertEqual(pulls, [["pull", "--platform", "linux/amd64", VLLM_REF]])
                self.assert_digest_references_only()
                self.assertFalse((self.state / "curl.txt").exists())
                self.assertFalse(any(c["args"][0] in ("build", "run") for c in self.docker_calls()))

    def test_wrong_platform_is_refused(self):
        self.assertIn("linux/amd64 is required", self.run_build(expected=1, FAKE_PLATFORM="linux/arm64"))

    def test_empty_source_hash_uses_verified_default(self):
        output = self.run_build(VLLM_SOURCE_SHA256="")
        self.assertIn("CONTENT PASS", output)
        self.assertIn(hashlib.sha256((self.state / "source.tar.gz").read_bytes()).hexdigest(), output)

    def test_bad_source_hash_is_refused(self):
        self.assertIn("vLLM source tarball SHA256 mismatch", self.run_build(expected=1, VLLM_SOURCE_SHA256="0" * 64))

    def test_bad_sdist_hash_is_refused_before_wheel(self):
        output = self.run_build(expected=1, FAKE_CORRUPT_SDIST="1")
        self.assertIn("Arctic sdist SHA256 mismatch", output)
        self.assertFalse(any("wheel" in c["args"] for c in self.docker_calls()))

    def test_debs_are_prepared_only_when_tools_are_missing(self):
        self.run_build(FAKE_MISSING_TOOLS="1")
        self.assertTrue((self.state / "context/assets/debs/fixture.deb").is_file())

    def test_required_content_mismatch_fails(self):
        output = self.run_build(expected=1, FAKE_CONTENT_CHANGE=".py")
        self.assertIn("CONTENT FAIL", output)
        self.assertIn("SUMMARY:", output)

    def test_compiled_difference_is_reported_and_allowed(self):
        output = self.run_build(FAKE_CONTENT_CHANGE=".so")
        self.assertIn("MISMATCH [shared objects]", output)
        self.assertIn("CONTENT PASS", output)

    def test_missing_compiled_file_fails(self):
        self.assertIn("MISSING", self.run_build(expected=1, FAKE_CONTENT_CHANGE="missing"))

    def test_extra_compiled_file_fails(self):
        self.assertIn("EXTRA", self.run_build(expected=1, FAKE_CONTENT_CHANGE="extra"))

    def test_failed_build_stops_chain(self):
        self.assertIn("build failed", self.run_build(expected=1, FAKE_BUILD_FAIL="1"))
        self.assertEqual(sum(c["args"][0] == "build" for c in self.docker_calls()), 2)

    def test_failed_identity_check_fails(self):
        output = self.run_build(expected=1, CHECK="1", FAKE_IDENTITY_FAIL="1")
        self.assertIn("identity check failed", output)

    def test_default_build_stops_at_l1a(self):
        self.run_build()
        tags = [c["args"][c["args"].index("-t") + 1] for c in self.docker_calls() if c["args"][0] == "build"]
        self.assertNotIn("vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan", tags)
        self.assertFalse((self.base / "logs/built-ba4-manifest.txt").exists())

    def test_ba4_builds_on_checked_l1a_with_its_own_context(self):
        output = self.run_build(BA4="1", CHECK="1")
        builds = [c["args"] for c in self.docker_calls() if c["args"][0] == "build"]
        self.assertEqual([a[a.index("-t") + 1] for a in builds][2:],
                         ["vllm-qwen38:v0310-r27c-A_S-L1a-ubplan", "vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan"])
        self.assertTrue(builds[3][-1].endswith("/ba4"), builds[3])
        self.assertTrue(builds[3][builds[3].index("-f") + 1].endswith("/ba4/Dockerfile"))
        runs = [c["args"] for c in self.docker_calls() if c["args"][0] == "run" and "generate" in c["args"]]
        self.assertEqual([a[a.index("--entrypoint") + 2] for a in runs],
                         ["vllm-qwen38:v0310-r27c-A_S-L1a-ubplan", "vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan"])
        identity = [c["args"] for c in self.docker_calls() if c["args"][0] == "run" and "--runtime" in c["args"]]
        self.assertEqual(identity[0][identity[0].index("--entrypoint") + 2], "vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan")
        self.assertIn("vllm_ba4", identity[0][-1])
        self.assertEqual(output.count("CONTENT PASS"), 2)
        self.assertIn("BA4 layer's pinned outputs", output)
        ctx = self.state / "ba4-context"
        source = ROOT / "patches-v0310/ba4"
        staged = sorted(p.relative_to(ctx).as_posix() for p in ctx.rglob("*") if p.is_file())
        self.assertEqual(staged, sorted(p.relative_to(source).as_posix() for p in source.rglob("*")
                                        if p.is_file() and "__pycache__" not in p.parts))
        for rel in staged:
            self.assertEqual((ctx / rel).read_bytes(), (source / rel).read_bytes(), rel)
        expected = (self.base / "logs/expected-ba4-manifest.txt").read_text()
        layer = json.loads((source / "manifest.json").read_text())
        for path, digest in layer["after"].items():
            if path.startswith("vllm/"):
                self.assertIn(digest + "  " + path + "\n", expected)

    def test_ba4_inventory_mismatch_fails(self):
        output = self.run_build(expected=1, BA4="1", FAKE_BA4_CHANGE="1")
        self.assertIn("CONTENT FAIL", output)
        self.assertIn("vllm/envs.py", output)

    def test_ba4_not_built_when_l1a_inventory_fails(self):
        output = self.run_build(expected=1, BA4="1", FAKE_CONTENT_CHANGE=".py")
        self.assertIn("the BA4 layer is not built", output)
        tags = [c["args"][c["args"].index("-t") + 1] for c in self.docker_calls() if c["args"][0] == "build"]
        self.assertNotIn("vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan", tags)

    def test_ba4_dry_run_plans_the_fourth_build(self):
        output = self.run_build(DRY_RUN="1", BA4="1")
        self.assertFalse(self.docker_calls())
        builds = [row for row in output.splitlines() if row.startswith("DOCKER_BUILDKIT=0 docker build ")]
        self.assertEqual(len(builds), 4)
        self.assertIn("v0310-r27c-A_S-BA4-L1a-ubplan", builds[3])
        self.assertIn("Derived BA4 inventory checked", output)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v031-manifest-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def compare(self, expected, actual):
        a, b = self.base / "expected.txt", self.base / "actual.txt"
        a.write_text(expected)
        b.write_text(actual)
        return subprocess.run([sys.executable, str(ROOT / "scripts/v031_manifest.py"), "compare", str(a), str(b)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_each_required_source_suffix_fails_on_hash_change(self):
        for suffix in (".py", ".json", ".cu", ".cuh"):
            with self.subTest(suffix=suffix):
                name = "vllm/a" + suffix
                result = self.compare("a" * 64 + "  " + name + "\n", "b" * 64 + "  " + name + "\n")
                self.assertEqual(result.returncode, 1, result.stdout)

    def test_other_overlay_byte_differences_are_informational(self):
        for name in ("vllm/a.so", "vllm-sm12x/build/a.o", "vllm-sm12x/build/.ninja_log", "vllm-sm12x/a.cpp"):
            with self.subTest(name=name):
                result = self.compare("a" * 64 + "  " + name + "\n", "b" * 64 + "  " + name + "\n")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("MISMATCH", result.stdout)

    def test_order_does_not_affect_comparison(self):
        rows = ["a" * 64 + "  vllm/a.py\n", "b" * 64 + "  vllm/b.py\n"]
        self.assertEqual(self.compare("".join(rows), "".join(reversed(rows))).returncode, 0)

    def test_malformed_duplicate_empty_and_invalid_paths_fail(self):
        valid = "a" * 64 + "  vllm/a.py\n"
        for bad in ("", "bad hash\n", valid + valid, "a" * 64 + "  vllm/../a.py\n",
                    "a" * 64 + "  /vllm/a.py\n", "a" * 64 + "  vllm/a.txt\n"):
            with self.subTest(bad=bad):
                self.assertEqual(self.compare(valid, bad).returncode, 2)

    def test_generator_selects_exact_package_suffixes_and_all_overlay_files(self):
        site, overlay = self.base / "site", self.base / "overlay"
        for name in ("flashinfer/a.py", "flashinfer/a.so", "vllm/a.cu", "vllm/sub/a.cuh", "vllm/a.json",
                     "vllm/skip.pyc", "vllm/skip.txt"):
            path = site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        overlay.mkdir()
        (overlay / ".ninja_log").write_bytes(b"log")
        (overlay / "a.o").write_bytes(b"object")
        (site / "vllm/link.py").symlink_to(site / "flashinfer/a.py")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/v031_manifest.py"), "generate",
                                 "--site-root", str(site), "--overlay-root", str(overlay)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        rows = result.stdout.splitlines()
        self.assertEqual(len(rows), 7)
        self.assertFalse(any("skip" in row or "link.py" in row for row in rows))
        for row in rows:
            digest, name = row.split("  ", 1)
            path = overlay / name.split("/", 1)[1] if name.startswith("vllm-sm12x/") else site / name
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
