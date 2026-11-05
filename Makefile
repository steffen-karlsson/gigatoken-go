# gigatoken-go: build the Rust shim, then link it into the Go binary.
#
# Profiles: make PROFILE=debug ... (faster builds, for iterating on the
# shim); the default is release, which matches upstream's fat-LTO profile.

PROFILE ?= release
VERSION ?= dev
TARGET  ?= $(shell rustc -vV | sed -n 's/^host: //p')

ifeq ($(PROFILE),release)
CARGO_PROFILE_FLAGS := --release
LIB_SRC := rust/target/release/libgigatoken_shim.a
else
CARGO_PROFILE_FLAGS :=
LIB_SRC := rust/target/debug/libgigatoken_shim.a
endif

SHIM_LIB     := rust/lib/libgigatoken_shim.a
GPT2_URL     ?= https://huggingface.co/gpt2/resolve/main/tokenizer.json
TEST_JSON    := testdata/gpt2/tokenizer.json
MUSL_TARGET   := x86_64-unknown-linux-musl

.PHONY: all rust build test testdata demo lint fmt-check clippy gofmt-check vet \
        clean distclean static-linux package

all: build

# The shim .a is copied to a stable path (rust/lib) that the cgo LDFLAGS
# in tokenizer/tokenizer.go points at, so the Go side never cares which
# target/profile produced it. gen-pystub.sh derives libpystub.a from the
# actual archive (see its header comment) and must run on the platform
# that produced the library.
rust:
	cd rust && cargo build $(CARGO_PROFILE_FLAGS)
	@mkdir -p rust/lib
	@cp $(LIB_SRC) $(SHIM_LIB)
	@./rust/gen-pystub.sh $(SHIM_LIB) rust/lib
	@echo "shim -> $(SHIM_LIB)"

build: rust
	go build ./...

test: rust testdata
	go test -v ./...

$(TEST_JSON):
	@mkdir -p testdata/gpt2
	curl -fsSL -o $@ $(GPT2_URL)

testdata: $(TEST_JSON)

demo: rust testdata
	go run ./cmd/gigatoken-demo $(TEST_JSON) "Hello, world!" "🚀 all in one binary"

# --- Linting gates (CI runs `make lint` before any packaging) -----------

lint: fmt-check clippy gofmt-check vet

fmt-check:
	cd rust && cargo fmt --check

clippy:
	cd rust && cargo clippy --all-targets -- -D warnings

gofmt-check:
	@out=$$(gofmt -l tokenizer cmd); if [ -n "$$out" ]; then \
		echo "the following files need gofmt:"; echo "$$out"; exit 1; fi
	@echo "gofmt OK"

vet:
	go vet ./...

# --- Packaging / release ------------------------------------------------

# Fully static Linux build (x86_64 musl): self-contained binary, friendly
# to distroless and scratch images. Run on a Linux host or CI runner with
# `musl-tools` installed. The CC_* exports point the C build scripts of
# ring/zstd/lz4/brotli (dep-tree natives) at the musl cross compiler.
static-linux: export CC_x86_64_unknown_linux_musl = musl-gcc
static-linux: export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER = musl-gcc
static-linux:
	rustup target add $(MUSL_TARGET)
	cd rust && cargo build --release --target $(MUSL_TARGET)
	@mkdir -p rust/lib
	@cp rust/target/$(MUSL_TARGET)/release/libgigatoken_shim.a $(SHIM_LIB)
	CC=musl-gcc ./rust/gen-pystub.sh $(SHIM_LIB) rust/lib
	CGO_ENABLED=1 CC=musl-gcc go build -trimpath -ldflags="-s -w" \
		-o gigatoken-demo ./cmd/gigatoken-demo
	@file gigatoken-demo || true

# Package the static binary plus the linkable shim libraries (for
# consuming the tokenizer package from another Go module — see README).
# Run after `make static-linux` (or `make rust` for a native build).
package:
	@mkdir -p dist
	tar czf dist/gigatoken-demo-$(VERSION)-$(TARGET).tar.gz gigatoken-demo
	tar -C rust/lib -czf dist/gigatoken-shim-$(VERSION)-$(TARGET).tar.gz \
		libgigatoken_shim.a libpystub.a pystub.c
	@echo "packaged -> dist/"

clean:
	cd rust && cargo clean
	rm -rf rust/lib dist
	go clean

distclean: clean
	rm -rf testdata
