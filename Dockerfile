   FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
   COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

   RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential git cmake autoconf automake libtool \
       ffmpeg libgl1 libglib2.0-0 python3-setuptools && rm -rf /var/lib/apt/lists/*

   # --- system layer: the part that actually hurts ---
   RUN git clone https://github.com/usnistgov/SCTK /opt/SCTK && cd /opt/SCTK && \
       make config && make all && make install && make doc
   ENV PATH="/opt/SCTK/bin:${PATH}"

   # --- python layer ---
   # Put uv's interpreter outside /root. By default it lands in
   # /root/.local/share/uv/python/, and /root is 0700 — so .venv/bin/python is a
   # symlink no non-root user can follow, and `docker run --user $(id -u)` fails
   # with "python: command not found". Running as root instead would leave every
   # file the container writes into the mounted repo owned by root.
   ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
   # Keep the venv OUTSIDE /workspace. The source is bind-mounted over
   # /workspace at run time, which would shadow a venv living there — and
   # working around that with an anonymous volume makes Docker create
   # /workspace/.venv on the host AS ROOT in any checkout that lacks one,
   # which then blocks `uv` from creating its own. Putting it at /opt/venv
   # removes the collision entirely.
   ENV UV_PROJECT_ENVIRONMENT=/opt/venv
   WORKDIR /workspace
   COPY pyproject.toml uv.lock .python-version ./
   RUN uv sync --frozen --no-install-project && \
       chmod -R a+rX /opt/uv-python /opt/venv

   # `uv pip` discovers ./.venv relative to the working directory, which no
   # longer exists now that the venv lives at /opt/venv. VIRTUAL_ENV points it
   # at the right place — both for the build steps below and for anyone running
   # `uv pip install` interactively inside the container.
   ENV VIRTUAL_ENV=/opt/venv

   # ctcdecode: setup.py imports torch, so build isolation must be off
   RUN uv pip install --upgrade pip wheel "setuptools<81"
   RUN git clone --recursive https://github.com/parlance/ctcdecode /tmp/ctcdecode && \
       uv pip install /tmp/ctcdecode --no-build-isolation && rm -rf /tmp/ctcdecode

   # --- MMPose / MMDetection: the RTMPose front end ---
   # Versions pinned to the set verified working against torch 2.0.0+cu118;
   # OpenMMLab's own instructions float them, and a float here is a rebuild
   # that silently changes the thing being measured.
   #
   # chumpy is DELIBERATELY EXCLUDED, which is why mmpose and mmdet install
   # with --no-deps and their real dependencies are listed explicitly. chumpy
   # is an mmpose runtime dependency used only for SMPL / 3D body models, which
   # RTMPose never touches, and it is unusable here on two counts: its setup.py
   # does `import pip` (invisible under build isolation), and its __init__ does
   # `from numpy import bool, int, float, ...` — aliases removed in NumPy 1.24.
   # Fixing it would mean pinning numpy below 1.24 for a package the pipeline
   # does not use.
   RUN uv pip install openmim && \
       /opt/venv/bin/mim install "mmengine==0.10.7" && \
       /opt/venv/bin/mim install "mmcv==2.1.0" && \
       uv pip install \
           json_tricks munkres xtcocotools scipy matplotlib pillow \
           terminaltables pycocotools shapely six && \
       uv pip install "mmpose==1.3.2" --no-deps && \
       uv pip install "mmdet==3.2.0" --no-deps && \
       uv pip install --upgrade "setuptools<81" "tqdm>=4.70.0"

   # --- TensorRT / ONNX: "ONNX -> engine, on each device" ---
   # TensorRT 10.3 because it is what JetPack 6.x ships on the Orin: an ONNX
   # graph that parses here parses there, which is an early opset check. The cu11 builds match the image's CUDA 11.8.
   #
   # The concrete -libs / -bindings wheels rather than the `tensorrt-cu11`
   # meta-package, which is an sdist whose build shells out to pip and fails.
   # The bindings wheel imports as `tensorrt_bindings`; harness/engines.py
   # accepts either name. pip rather than uv for these three lines, because
   # pip is what was verified to resolve them against this image.
   #
   # onnx 1.16 with protobuf 3.20: nothing else here uses protobuf, and newer
   # onnx drags in a far newer protobuf for nothing this project needs.
   # onnxruntime-gpu comes from Microsoft's CUDA 11 feed (PyPI's builds are
   # CUDA 12) and is the parity reference for exports, not a measured runtime.
   RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps \
           "tensorrt-cu11-libs==10.3.0" "tensorrt-cu11-bindings==10.3.0" && \
       /opt/venv/bin/python -m pip install --no-cache-dir "onnx==1.16.2" "protobuf==3.20.3" && \
       /opt/venv/bin/python -m pip install --no-cache-dir "onnxruntime-gpu==1.20.1" \
           --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ && \
       /opt/venv/bin/python -c "import tensorrt_bindings, onnx, onnxruntime, numpy; assert numpy.__version__.startswith('1.')"

   ENV PATH="/opt/venv/bin:${PATH}"

   # The project itself is deliberately NOT installed (--no-install-project
   # above): the source is bind-mounted at run time, so code edits on the host
   # are live inside the container with no rebuild. It just needs to be on the
   # path.
   ENV PYTHONPATH=/workspace/src

   # Keep .pyc files out of the mounted repo — they would otherwise appear on
   # the host owned by whichever uid the container ran as.
   ENV PYTHONDONTWRITEBYTECODE=1

   # The container runs as the host user, who has no home directory inside it,
   # so matplotlib (pulled in by mmengine) warns and rebuilds its font cache on
   # every single run. Point it somewhere writable.
   ENV MPLCONFIGDIR=/tmp/matplotlib