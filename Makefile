SRC=$(wildcard *.cpp)
OBJ=$(patsubst %.cpp, %.o, $(SRC))
OBJ_DEBUG=$(patsubst %.cpp, %.do, $(SRC))

EXE=$(shell basename $(CURDIR))
EXE_DEBUG=$(EXE)_debug

DEBUG_FLAG=-c -Ddebug_1 -Ddebug_2
HIP_PATH=/opt/rocm/hip
CXX=$(HIP_PATH)/bin/hipcc

LD_FLAG=-L$(HIP_PATH)/lib -lhip_hcc

.PHONY: all clean run

all:$(EXE)

release:$(EXE)

debug:$(EXE_DEBUG)

$(EXE):$(OBJ)
	$(CXX) $(OBJ) -o $@

$(EXE_DEBUG):$(OBJ_DEBUG)
	$(CXX) $(OBJ) -o $@

%.o:%.cpp
	$(CXX) -c $< -o $@

%.do:%.cpp
	$(CXX) $(DEBUG_FLAG) $< -o $(patsubst %.cpp, %.o, $<)

clean:
	rm *.do *.o $(EXE) $(EXE_DEBUG)
