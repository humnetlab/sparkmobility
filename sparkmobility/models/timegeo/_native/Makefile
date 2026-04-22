# Build the TimeGeo parameter-estimation binary (module_2_3_1).
#
# Usage:
#   make            # build module_2_3_1
#   make clean      # remove binary
#
# Override include / lib paths on the command line if your Arrow / H3
# installs live elsewhere, e.g.:
#   make INCLUDES="-I/opt/arrow/include -I/usr/local/include"

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -fopenmp
INCLUDES ?= -I$(HOME)/local/include -I$(HOME)/apache-arrow-20.0.0/cpp/src
LIBS     ?= -larrow -lparquet -lh3 -lsnappy

TARGET = module_2_3_1
SRC    = module_2_3_1.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
