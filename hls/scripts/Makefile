
REAL_ROOT_DIR := $(realpath $(ROOT_DIR))
ROOT_BINDIR := $(REAL_ROOT_DIR)/cmake-build-debug/bin
ROOT_LIBDIR := $(REAL_ROOT_DIR)/cmake-build-debug/lib
FILENAME := $(FN)

test:
	$(info $$var is [${REAL_ROOT_DIR}])

# promote allocs
$(FILENAME).promoted:
	$(ROOT_BINDIR)/torch-mlir-opt $(FILENAME) \
 		-torch-hls-promote-allocs \
 		-o $(FILENAME).promoted

# unroll reduction loops
$(FILENAME).dirty: $(FILENAME).promoted
	$(ROOT_BINDIR)/torch-mlir-opt $(FILENAME).promoted \
		-dirty-pass="unrollparfor=0 csts=1 unrollloops=1" \
		-o $(FILENAME).dirty

# lower partially so you can unroll parloops
$(FILENAME).dirty.llvm: $(FILENAME).dirty
	$(ROOT_BINDIR)/mlir-opt $(FILENAME).dirty \
	  -convert-memref-to-llvm \
	  -loop-invariant-code-motion \
	  -o $(FILENAME).dirty.llvm

# unroll parallel loops
$(FILENAME).dirty.llvm.unrollparfor: $(FILENAME).dirty.llvm
	$(ROOT_BINDIR)/torch-mlir-opt $(FILENAME).dirty.llvm \
		-dirty-pass="unrollparfor=1 csts=1 unrollloops=0" \
		-o $(FILENAME).dirty.llvm.unrollparfor

# lower to llvm dialect
$(FILENAME).dirty.llvm.unrollparfor.llvm: $(FILENAME).dirty.llvm.unrollparfor
	$(ROOT_BINDIR)/mlir-opt $(FILENAME).dirty.llvm.unrollparfor \
	  -convert-memref-to-llvm \
	  -convert-math-to-llvm \
	  -convert-arith-to-llvm \
	  -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
	  -reconcile-unrealized-casts \
	  -o $(FILENAME).dirty.llvm.unrollparfor.llvm

# hoist constants and constant propagate
$(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse: $(FILENAME).dirty.llvm.unrollparfor.llvm
	python read_large_file.py $(FILENAME).dirty.llvm.unrollparfor.llvm

# translate to llvm ir
$(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse.ll: $(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse
	$(ROOT_BINDIR)/mlir-translate \
		$(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse \
		-mlir-to-llvmir \
		-o $(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse.ll

# llvm ir opts
$(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse.ll.vitis: $(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse.ll
	${ROOT_BINDIR}/opt $(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse.ll \
		-debug \
		-S \
		-enable-new-pm=0 \
		-load "${ROOT_LIBDIR}/VhlsLLVMRewriter.so" \
		-strip-attr \
		-time-passes \
		-stats \
		-mem2arr \
		-strip-debug \
		-xlnmath \
		-xlnname \
		-o $(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse.ll.vitis


.PHONY : all
all : $(FILENAME).dirty.llvm.unrollparfor.llvm.cse.cse.ll.vitis

.DEFAULT_GOAL := all

#  -instcombine -instcombine-max-iterations=10
