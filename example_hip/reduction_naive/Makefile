HIP_PATH ?= /opt/rocm
HIPCC = $(HIP_PATH)/bin/hipcc
CXXFLAGS = -O2 -std=c++17

TARGET = test_reduction
SRC = reduction_naive.hip

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: clean
