# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -pthread -g -O0 -w
CXXFLAGS += -DIMP=$(IMP)

# Executable and source files
TEST_TARGET = test_linear
PROFILE_TARGET =
APP_TARGET = chat
TARGET = $(TEST_TARGET) $(PROFILE_TARGET) $(APP_TARGET)

BUILDDIR := build/transformer
PROFILEDIR := build_profile/transformer
LIB_DIR = ../kernels
LIB_SRC = $(wildcard $(LIB_DIR)/*.cc)
INCLUDE_DIRS = -I$(LIB_DIR) -I./include -I./include/nn_modules -I./json/single_include/
LIB =

ifeq ($(shell uname -m),x86_64)
	# For Intel machines with AVX
	CXXFLAGS += -mavx2 -mfma -ffast-math -fpermissive -DQM_x86
	LIB_SRC += $(wildcard $(LIB_DIR)/avx/*.cc)
else ifeq ($(shell uname -p),arm)
	CXX = /opt/homebrew/opt/llvm/bin/clang++
	LIB += -L/opt/homebrew/opt/boost/lib
	CXXFLAGS += -march=native -DQM_ARM -fPIC -march=armv8.2-a+dotprod
	INCLUDE_DIRS += -I/opt/homebrew/opt/boost/include
	LIB_SRC += $(wildcard $(LIB_DIR)/neon/*.cc)
else
	@echo "Device unsupported!.
	LIB_REF_SRC = $(wildcard $(LIB_DIR)/ref/*.cc)
	LIB_SRC += $(LIB_REF_SRC)
endif
LIB_REF_SRC = $(wildcard $(LIB_DIR)/starter_code/*.cc)
LIB_SRC += $(LIB_REF_SRC)

SRC_DIR = src
SRC = $(wildcard src/*.cc)
SRC += $(wildcard src/nn_modules/*.cc)
OPS =  $(wildcard src/ops/*.cc)
SRC += $(OPS)
SRC += $(LIB_SRC)

# Default target
all: $(TARGET)

# Phony targets
.PHONY: all clean

# Metal lib
library.air: $(LIB_DIR)/metal/kernel/op.metal
	xcrun -sdk macosx metal -ffast-math -fno-fast-math $(LIB_ACC_INC) -c $< -o library.air
default.metallib: library.air
	xcrun -sdk macosx metallib library.air -o default.metallib

OBJS = $(addprefix $(BUILDDIR)/,$(SRC:.cc=.o))
PROFILE_OBJS = $(addprefix $(PROFILEDIR)/,$(SRC:.cc=.o))

# Pattern rules
$(BUILDDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(PROFILEDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -DPROFILER -c $< -o $@

# Linking
# Rule for TEST_TARGET
$(TEST_TARGET): %: tests/%.cc $(OBJS)
	@ $(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $^ $(LIB) $(LDFLAGS)

# Rule for APP_TARGET
$(APP_TARGET): %: application/%.cc $(OBJS)
	@ $(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $@ $^ $(LIB) $(LDFLAGS)

# Clean up
clean:
	@ rm -f $(TARGET)
	@ rm -rf *.dSYM
	@ rm -rf build/kernels
	@ rm -rf $(BUILDDIR)
	@ rm -rf $(PROFILEDIR)
