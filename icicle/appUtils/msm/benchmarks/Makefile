all: msm

msm:
	nvcc -arch=sm_70 -std=c++17 -c wrappers.cu -o wrappers.o
	nvcc -arch=sm_70 -std=c++17 -lnvidia-ml wrappers.o ./benchmark.cu -o benchmark

