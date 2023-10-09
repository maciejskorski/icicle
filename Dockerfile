# docker build -t icicle:latest .

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# install package dependencies
RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev gawk \
    && rm -rf /var/lib/apt/lists/*

# install cmake
RUN wget https://cmake.org/files/v3.23/cmake-3.23.0-linux-x86_64.tar.gz -O cmake.tar.gz && \
    mkdir -p /opt/cmake && \
    tar -xf cmake.tar.gz --directory opt/cmake && \
    rm cmake.tar.gz && \
    ln -s /opt/cmake/cmake-3.23*/bin/* /usr/local/bin


CMD ["bash"]
