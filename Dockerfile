
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL="C"
ENV TZ="UTC"
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        gosu \
        htop \
        nvtop \
        wget \
        g++ \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CONDAHOME="/opt/conda"
ENV PATH="${CONDAHOME}/bin:${PATH}"
RUN wget -q -P /tmp https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

COPY environment.yml /opt/conda/environment.yml
RUN conda install -y python=3.12 \
    && conda env update -n base -f /opt/conda/environment.yml \
    && conda clean -y --all --force-pkgs-dirs

RUN useradd -m -s /bin/bash user
COPY --chmod=755 <<EOF /usr/local/bin/entrypoint.sh
#!/bin/bash
groupmod -g \${GID:-9001} -o user &>/dev/null
usermod -u \${UID:-9001} -d /home/user -m -o user &>/dev/null
exec gosu user "\$@"
EOF
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]