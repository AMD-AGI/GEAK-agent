HIP_PATH ?= /opt/rocm
HIPCC = $(HIP_PATH)/bin/hipcc
CXXFLAGS = -O3 -std=c++17 -ffast-math

TARGET = test_silu
SRC = silu.hip

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: clean
