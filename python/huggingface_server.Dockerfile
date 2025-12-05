ARG CUDA_VERSION=12.8.1
ARG VENV_PATH=prod_venv
ARG PYTHON_VERSION=3.12
ARG WORKSPACE_DIR=/kserve-workspace

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-rockylinux9 AS base

ARG WORKSPACE_DIR
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

RUN dnf update -y --nobest \
    && dnf install -y --allowerasing git curl sudo gcc gcc-c++ kernel-devel cmake make automake \
    && dnf install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-devel \
    && alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 -m pip install --upgrade pip \
    && python3 --version && python3 -m pip --version \
    && dnf clean all && rm -rf /var/cache/dnf

# Install uv and ensure it's in PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}

WORKDIR ${WORKSPACE_DIR}

#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build

ARG WORKSPACE_DIR
ARG VLLM_VERSION=0.12.0
ARG LMCACHE_VERSION=0.3.0
ARG BITSANDBYTES_VERSION=0.46.1
ARG FLASHINFER_VERSION=0.2.6.post1
# Need a separate CUDA arch list for flashinfer because '7.0' is not supported by flashinfer
ARG FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

WORKDIR ${WORKSPACE_DIR}

ARG VENV_PATH
RUN python3 -m venv ${VENV_PATH}
# Activate virtual env by setting VIRTUAL_ENV
ENV VIRTUAL_ENV=${WORKSPACE_DIR}/${VENV_PATH}
ENV PATH="${WORKSPACE_DIR}/${VENV_PATH}/bin:$PATH"

# From this point, all Python packages will be installed in the virtual environment and copied to the final image

COPY kserve/pyproject.toml kserve/uv.lock kserve/
RUN --mount=type=cache,target=/root/.cache/uv cd kserve && uv sync --active --no-cache
COPY kserve kserve
RUN --mount=type=cache,target=/root/.cache/uv cd kserve && uv sync --active --no-cache

COPY storage/pyproject.toml storage/uv.lock storage/
RUN --mount=type=cache,target=/root/.cache/uv cd storage && uv sync --active --no-cache
COPY storage storage
RUN --mount=type=cache,target=/root/.cache/uv cd storage && uv pip install . --no-cache

COPY huggingfaceserver/pyproject.toml huggingfaceserver/uv.lock huggingfaceserver/health_check.py huggingfaceserver/
RUN --mount=type=cache,target=/root/.cache/uv cd huggingfaceserver && uv sync --active --no-cache
COPY huggingfaceserver huggingfaceserver
RUN --mount=type=cache,target=/root/.cache/uv cd huggingfaceserver && uv sync --active --no-cache

# Install vllm
# https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html, https://docs.vllm.ai/en/latest/models/extensions/tensorizer.html
# https://docs.vllm.ai/en/latest/models/extensions/fastsafetensor.html
RUN --mount=type=cache,target=/root/.cache/pip pip install vllm[runai,tensorizer,fastsafetensors]==${VLLM_VERSION}

# Install lmcache
#RUN --mount=type=cache,target=/root/.cache/pip pip install lmcache==${LMCACHE_VERSION}

# Install bits and bytes from source (so it uses TORCH_CUDA_ARCH_LIST)
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/ && cmake -DCOMPUTE_BACKEND=cuda -S . && make && pip install .

####################################
# Fix CVE
RUN pip install "setuptools>=78.1.1"
RUN pip install "urllib3>=2.5.0" "requests>=2.32.4" "starlette>=0.49.1" "aiohttp>=3.12.14"
RUN pip install "h11>=0.16.0" "pillow>=11.3.0"
RUN pip uninstall -y ray

# Use Bash with `-o pipefail` so we can leverage Bash-specific features (like `[[ … ]]` for glob tests)
# and ensure that failures in any part of a piped command cause the build to fail immediately.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install flashinfer
#RUN --mount=type=cache,target=/root/.cache/pip \
#  # FlashInfer already has a wheel for PyTorch 2.7.0 and CUDA 12.8.
#  if [[ "$CUDA_VERSION" == 12.8* ]]; then \
#    pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-${FLASHINFER_VERSION}%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl; \
#  else \
#    export TORCH_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST}" && \
#    git clone --branch v${FLASHINFER_VERSION} --recursive https://github.com/flashinfer-ai/flashinfer.git && \
#    cd flashinfer && \
#    python3 -m flashinfer.aot && \
#    pip install --no-build-isolation . && \
#    cd .. && rm -rf flashinfer; \
#  fi

# Generate third-party licenses
#COPY pyproject.toml pyproject.toml
#COPY third_party/pip-licenses.py pip-licenses.py
#RUN mkdir -p third_party/library && python3 pip-licenses.py

#################### WHEEL BUILD IMAGE ####################

#################### PROD IMAGE ####################
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-rockylinux9 AS prod

ARG WORKSPACE_DIR
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR ${WORKSPACE_DIR}

# Fix repository configuration
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/rocky*.repo \
    && sed -i 's|#baseurl=http://dl.rockylinux.org/$contentdir|baseurl=http://download.rockylinux.org/pub/rocky|g' /etc/yum.repos.d/rocky*.repo \
    && dnf clean all

# Install Python and other dependencies
# NOTE: no pip.  Update for CVEs in this layer
RUN dnf update -y \
    && dnf install -y --allowerasing git curl sudo gcc \
    && dnf install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-devel \
    && alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && python3 --version \
    && dnf install -y --allowerasing sqlite-libs expat shadow-utils \
    && dnf clean all && rm -rf /var/cache/dnf


ARG VENV_PATH
# Activate virtual env by setting VIRTUAL_ENV
ENV VIRTUAL_ENV=${WORKSPACE_DIR}/${VENV_PATH}
ENV PATH="${WORKSPACE_DIR}/${VENV_PATH}/bin:$PATH"

# Create non-root user
RUN useradd kserve -m -u 1000 -d /home/kserve

#COPY --from=build --chown=kserve:kserve ${WORKSPACE_DIR}/third_party third_party
COPY --from=build --chown=kserve:kserve ${WORKSPACE_DIR}/$VENV_PATH $VENV_PATH
COPY --from=build ${WORKSPACE_DIR}/kserve kserve
COPY --from=build ${WORKSPACE_DIR}/storage storage
COPY --from=build ${WORKSPACE_DIR}/huggingfaceserver huggingfaceserver

# Set a writable Hugging Face home folder to avoid permission issue. See https://github.com/kserve/kserve/issues/3562
ENV HF_HOME="/tmp/huggingface"
# https://huggingface.co/docs/safetensors/en/speed#gpu-benchmark
ENV SAFETENSORS_FAST_GPU="1"
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubdisabletelemetry
ENV HF_HUB_DISABLE_TELEMETRY="1"
# NCCL Lib path for vLLM. https://github.com/vllm-project/vllm/blob/ec784b2526219cd96159a52074ab8cd4e684410a/vllm/utils.py#L598-L602
ENV VLLM_NCCL_SO_PATH="/lib/x86_64-linux-gnu/libnccl.so.2"
# https://github.com/vllm-project/vllm/issues/6152
# Set the multiprocess method to spawn to avoid issues with cuda initialization for `mp` executor backend.
ENV VLLM_WORKER_MULTIPROC_METHOD="spawn"

# Download harmony artifacts
ENV TIKTOKEN_ENCODINGS_BASE="/kserve-workspace/encodings"
RUN mkdir $TIKTOKEN_ENCODINGS_BASE
RUN cd $TIKTOKEN_ENCODINGS_BASE && curl -O https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
RUN cd $TIKTOKEN_ENCODINGS_BASE && curl -O https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken


USER 1000
ENV PYTHONPATH=${WORKSPACE_DIR}/huggingfaceserver

ENTRYPOINT ["python3", "-m", "huggingfaceserver"]
#################### PROD IMAGE ####################



