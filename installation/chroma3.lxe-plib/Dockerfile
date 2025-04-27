FROM youngsm/chroma3:nvidia.base
MAINTAINER Sam Young <youngsam@stanford.edu>

WORKDIR /opt/
ADD https://api.github.com/repos/BenLand100/chroma/git/refs/heads/master /opt/chroma_version.json
RUN git clone https://github.com/BenLand100/chroma
WORKDIR /opt/chroma/
RUN sed -i 's/VIRTUAL_ENV/CONDA_PREFIX/g' setup.py #use anaconda env instead

# quick fix for EventViewer
RUN sed -i '741s/self\.i = len(self\.l-1)/self.i = len(self.l)-1/' chroma/camera.py
# quick fix for overflow error on big exponents
RUN pip install -e .
# below are extras for analyses
RUN pip install -U trimesh numpy scipy scikit-learn h5py fire scikit-build
RUN pip install git+https://github.com/CIDeR-ML/photonlib.git
RUN pip install git+https://github.com/CIDeR-ML/siren-lartpc.git
RUN pip install gdown
RUN pip install rich

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
