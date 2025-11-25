HIPCC = hipcc
TARGET = test_transpose
SRC = transpose_naive.hip
CFLAGS = -O3 -std=c++17

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

