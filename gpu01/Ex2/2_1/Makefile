# Makefile to compile current excersise
#
# ./bin/nullKernelAsync: ./src/nullKernelAsync.cu
# 	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

NVCC=nvcc

CUDA_ROOT=/opt/cuda-12.0
LOCAL_INC_PATH=./inc

LIB=-L$(CUDA_ROOT)/lib64
INC=-I$(CUDA_ROOT)/include -I$(LOCAL_INC_PATH)
NVCC_FLAGS=-O2 --compiler-options "-O2 -Wall -Wextra" --default-stream per-thread
OBJ_FLAGS := $(NVCC_FLAGS) -c

# path macros
BIN_PATH := bin
OBJ_PATH := obj
SRC_PATH := src

# compile macros
TARGET_NAME := nullKernelAsync

TARGET := $(BIN_PATH)/$(TARGET_NAME)

# src files & obj files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c*)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

# clean files list
DISTCLEAN_LIST := $(OBJ)
CLEAN_LIST := $(TARGET) $(DISTCLEAN_LIST)

# default rule
default: makedir all

# non-phony targets
$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(INC) $(LIB)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(NVCC) $(OBJ_FLAGS) -o $@ $< $(INC)

# phony rules
.PHONY: makedir
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(DISTCLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)
