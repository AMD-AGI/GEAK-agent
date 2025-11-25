HIP_PATH ?= /opt/rocm
HIPCC = $(HIP_PATH)/bin/hipcc
CXXFLAGS = -O2 -std=c++17

TARGET = test_gemm
SRC = gemm_naive.hip

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: clean
