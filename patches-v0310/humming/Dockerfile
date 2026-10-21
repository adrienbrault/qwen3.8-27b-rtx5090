FROM vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan
COPY patches/0001-deviceinfo-compile.patch /opt/humming-screen/0001-deviceinfo-compile.patch
COPY test_deviceinfo.py /opt/humming-screen/test_deviceinfo.py
RUN echo '70bb4187ae1ffc769e77290f6a370917d2201eb184115815452d887617c16ee4  /usr/local/lib/python3.12/dist-packages/humming/device.py' | sha256sum -c - \
 && cd /usr/local/lib/python3.12/dist-packages \
 && patch -p1 < /opt/humming-screen/0001-deviceinfo-compile.patch \
 && echo '76157b39ef2cdfa12eb3146023d8a7753d7a68d3a5eb8ac14f6f46db3a7aabb1  humming/device.py' | sha256sum -c - \
 && python3 /opt/humming-screen/test_deviceinfo.py
LABEL org.opencontainers.image.description="Humming experiment: upstream DeviceInfo fixes #148/#149 only; GEMM kernels unchanged"
