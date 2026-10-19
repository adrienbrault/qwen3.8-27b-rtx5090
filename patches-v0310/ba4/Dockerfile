FROM vllm-qwen38:v0310-r27c-A_S-L1a-ubplan
# Build context: out/L1a_BA4. No compiler, downloads, entrypoint, model or
# target FP8/NVFP4 selector changes. Flags remain OFF by default.
COPY L1a_BA4.diff manifest.json install.py make_layer.py /opt/l1a-ba4/
COPY fixtures /opt/l1a-ba4/fixtures
COPY tests /opt/l1a-ba4/tests
RUN CUDA_VISIBLE_DEVICES="" python3 /opt/l1a-ba4/install.py
RUN CUDA_VISIBLE_DEVICES="" python3 -c "import torch; assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__" && CUDA_VISIBLE_DEVICES="" python3 -m unittest discover -s /opt/l1a-ba4/tests -p 'test_cpu.py' -v
# test_cpu also runs test_compile.ServingCompileTests: real vLLM Parameters
# through patched apply/postload, fullgraph and evaluate_guards=False,
# live storage rebinding, metadata mutations, and original fallback lifetime.
# It also runs test_logging.WorkerLoggingTests against the pinned real vLLM
# logger/formatter and WorkerProc prefix setup, before/after runtime import.
# test_symbolic also executes the complete real vLLM guard-disabled wrapper
# with backed symbolic M, AOT/Inductor, M4 dispatch, and fresh F.linear fallbacks.
# j41r also runs test_piecewise: complete real piecewise cold/warm paths,
# prepared-device refusal before allocation, and real GDN generated-program
# equality for general M (matching Python development headers required).
# j41s keeps this equality in the inherited image environment, including all
# embedded C++ strings. Buffer-edge assertions replace Torch-specific kernel
# names; test_piecewise repeats the gate with Inductor node naming as well.
# j41t pins the inherited serving Torch release and includes test_profile:
# upstream producers, dynamic dim 0, M8192 first call, real wrapper.aot_compile.
# j41v includes test_splitter: real split_graph and attention splitting ops,
# first M8192, tensor-only offset metadata, piecewise AOT/Inductor compilation.
# j41x includes test_cache_safety: real Torch AOT content keys and GraphPickler
# for Dynamo/M4/general graphs; separate-process shared-disk cold/warm tests,
# both rank orders, with a real obsolete nonce entry containing a wrong program.
# Only registered metadata OpOverloads reach serialization; compiled ranges
# remove their dead probes explicitly, preserving attention side effects.
# j41y documents trace-time integer specialization; test_compile retains the
# real/fake metadata and cached-integer probe on aot_eager/inductor, alongside
# aligned offset-8 reuse and offset-1 refusal before run_out or USED logging.
# j42a includes test_v3_admission: profile-qualified admission, unchanged v2,
# source binding and weight/code refusals before allocation or selected work.
