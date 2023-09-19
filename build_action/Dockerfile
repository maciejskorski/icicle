# docker build -t icicle:latest .

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# install package dependencies
RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev

# install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.5/cmake-3.26.5-linux-x86_64.tar.gz -o cmake.tar.gz && \
    tar xf cmake-3.26.5-linux-x86_64.tar.gz && \
    mv cmake-3.26.5-linux-x86_64 /opt/ && \
    ln -s /opt/cmake-3.26.5-linux-x86_64/bin/* /usr/local/bin

CMD ["bash"]
