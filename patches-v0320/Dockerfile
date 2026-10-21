ARG BASE_IMAGE=vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9
FROM ${BASE_IMAGE} AS native
USER root
COPY apply.py manifest.json series /opt/v032/
COPY patches/ /opt/v032/patches/
COPY pcie_ipc_ar21/ /opt/v032/pcie_ipc_ar21/
RUN python3 -c "import importlib.metadata as m; assert m.version('vllm') == '0.32.0'; assert m.version('torch') == '2.13.0+cu130'; assert m.version('flashinfer-python') == '0.7.0.post1'"
RUN SITE=$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)') && \
    python3 /opt/v032/apply.py --root "$SITE" --receipt /opt/v032/core-receipt.json && \
    CUDA_VISIBLE_DEVICES="" MAX_JOBS=2 python3 -m pcie_ipc_ar21.build \
        --build-root /opt/pcie-build --output "$SITE/pcie_ipc_ar21/_C.so"
RUN python3 -c 'import torch,pcie_ipc_ar21.build; pcie_ipc_ar21.build.load_native(); print("PCIe extension loaded without a GPU")'

FROM native AS ba4
ARG CONTROL_IMAGE_ID
RUN SITE=$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)') && \
    python3 /opt/v032/apply.py --root "$SITE" --ba4 --control-image "$CONTROL_IMAGE_ID" \
        --receipt /opt/v032/ba4-receipt.json
ENV VLLM_SM12X_BA_M4=0
