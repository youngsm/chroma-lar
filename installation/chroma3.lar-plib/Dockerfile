FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ARG CHROMA_LAR_REF=main

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/python -m pip install --no-cache-dir uv==0.11.28
WORKDIR /workspace/
ADD https://api.github.com/repos/youngsm/chroma-lar/git/refs/heads/${CHROMA_LAR_REF} /opt/chroma_lar_version.json
RUN git clone --branch ${CHROMA_LAR_REF} https://github.com/youngsm/chroma-lar
# chroma-lar, TriChroma (Chroma + its Triton backend) and photonlib; no PyCUDA.
# Run from the checkout so uv reads its [tool.uv] settings (photonlib's build dependency).
WORKDIR /workspace/chroma-lar/
RUN uv pip install --no-cache --python /opt/conda/bin/python -e .
WORKDIR /workspace/
# analysis extras (their setup.py files import skbuild, installed above with photonlib)
RUN uv pip install --no-cache --python /opt/conda/bin/python --no-build-isolation scikit-learn gdown \
    git+https://github.com/CIDeR-ML/siren-lartpc.git \
    git+https://github.com/CIDeR-ML/sirentv.git
ENV PYRAT_ROOT=/workspace/chroma-lar
ENV PATH=/workspace/chroma-lar:$PATH
# keep a mounted home's ~/.local installs from shadowing the image's packages
ENV PYTHONNOUSERSITE=1

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
