# Gigatoken Go Wrapper

Run [gigatoken](https://github.com/marcelroed/gigatoken)'s Rust BPE
tokenizer **inside a single Go binary** — one process, one pod, no Python.
The Rust side is compiled to a static library and linked in via cgo, so
your Go-based Kubernetes deployment gets native-speed tokenization with
no sidecars, no wheels, no image bloat.

```
gigatoken-go/
├── go.mod
├── tokenizer/
│   └── tokenizer.go        # cgo wrapper — the Go API
├── rust/
│   ├── Cargo.toml          # shim crate (staticlib), git-pinned gigatoken
│   ├── rust-toolchain.toml # nightly (upstream needs portable_simd)
│   └── src/lib.rs          # extern "C" FFI surface
├── cmd/gigatoken-demo/     # smoke-test binary
└── Makefile                # cargo build + go build
```

## Build

Requires: Go, [rustup](https://rustup.rs) (nightly is auto-selected via
`rust/rust-toolchain.toml`), a C compiler, and `make`.

```sh
make test        # builds the shim, downloads GPT-2's tokenizer.json, runs tests
make demo        # "Hello, world!" -> [15496 11 995 0]
```

The Makefile copies `libgigatoken_shim.a` to `rust/lib/`, the stable
path the cgo `LDFLAGS` points at. `make PROFILE=debug test` skips fat LTO
for fast shim iteration.

## Use

```go
tok, err := tokenizer.Load("/models/gpt2/tokenizer.json")
defer tok.Close()

ids := tok.Encode("Hello, world!")     // []uint32{15496, 11, 995, 0}
txt, _ := tok.Decode(ids)              // []byte("Hello, world!")

batch, err := tok.EncodeBatch(docs)    // rayon-parallel
for i := range batch.NumDocs() {
    docIDs := batch.IDs(i)
}
// Already hold your corpus in one buffer? Zero-copy:
// tok.EncodeBatchFlat(flatBytes, byteOffsets)
```

- `Load` reads a HuggingFace **ByteLevel-BPE** `tokenizer.json` (GPT-2,
  RoBERTa, ...). SentencePiece/byte_fallback and tiktoken loaders exist
  upstream but aren't wired into the shim yet.
- `Tokenizer` is safe for concurrent use; `Close` must be called exactly once.
- `EncodeBatch` returns flat ids + cumulative offsets (gigatoken's ragged
  format); order is preserved.
- Rayon's thread pool is process-global: set `RAYON_NUM_THREADS` before
  the first batch encode to bound it.

## CI / releases

`.github/workflows/build.yml` runs on every PR and merge to `main`, all
through the Makefile targets:

1. **`make lint`** — `cargo fmt --check`, `cargo clippy -D warnings`,
   `gofmt` check, `go vet`
2. **`make test`** — builds the shim, downloads GPT-2's tokenizer.json,
   runs the Go suite
3. **`make static-linux` + `make package`** — only on merges to `main`,
   only after the gates pass: a fully static x86_64-musl binary plus a
   tarball of the linkable shim libraries. A native arm64 macOS job builds
   the same shim for `aarch64-apple-darwin` so macOS consumers can link
   without a Rust toolchain
4. Conventional-commit semver tag + GitHub release with the packages

## Consuming from another Go module

The release ships per-platform `gigatoken-shim-<version>-<target>.tar.gz`
(`x86_64-unknown-linux-musl`, `x86_64-unknown-linux-gnu`,
`aarch64-apple-darwin`) containing
`libgigatoken_shim.a`, `libpystub.a`, `pystub.c` and — for the musl
target — `libunwind.a` plus a companion `gigatoken-zig-<version>-<target>
.tar.gz` with the `zig cc`/`zig c++` wrappers. Another Go application
uses the tokenizer package like this:

```sh
# 1. Fetch the shim libraries for your platform
curl -fsSL -o shim.tar.gz \
  https://github.com/steffen-karlsson/gigatoken-go/releases/latest/download/gigatoken-shim-<version>-<target>.tar.gz
mkdir -p shim-lib && tar -C shim-lib -xzf shim.tar.gz

# 2. Build with the libraries on the linker path
export CGO_LDFLAGS="-L$PWD/shim-lib"
go build ./...
```

```go
import "github.com/steffen-karlsson/gigatoken-go/tokenizer"

tok, err := tokenizer.Load("/models/gpt2/tokenizer.json")
```

`CGO_LDFLAGS` appends to the package's built-in `-L` search path, so the
consumer only needs the directory holding the `.a` files — no Rust
toolchain, no Python, nothing else. (For this import path to resolve,
the module must be named after its GitHub URL — see `go.mod`.)

### Fully static (musl) consumers

On Linux, pick the archive by what the binary must do: `x86_64-unknown-linux-gnu`
for anything that runs on the build host — tests, dev builds, anything
linking against the system glibc — and `x86_64-unknown-linux-musl` for
release binaries. Test binaries in particular force an external link
against the host C library, so they cannot link through a musl cross
toolchain; CI should fetch the gnu archive for `go test` and reserve
the musl flow below for release builds.

Binaries destined for distroless/static or scratch images must link
musl. The musl shim tarball ships everything except the zig toolchain
itself (https://ziglang.org — one download, no install):

```sh
# 1. Fetch the shim + zig wrapper tarballs for your version
V=1.1.0
T=x86_64-unknown-linux-musl
curl -fsSL -o shim.tar.gz \
  https://github.com/steffen-karlsson/gigatoken-go/releases/download/v$V/gigatoken-shim-$V-$T.tar.gz
curl -fsSL -o zig.tar.gz \
  https://github.com/steffen-karlsson/gigatoken-go/releases/download/v$V/gigatoken-zig-$V-$T.tar.gz
mkdir -p shim-lib && tar -C shim-lib -xzf shim.tar.gz
mkdir -p shim-zig && tar -C shim-zig -xzf zig.tar.gz

# 2. Build fully static with zig as the musl C toolchain
export PATH="$PWD/shim-zig:$PATH"
CGO_ENABLED=1 CC=shim-zig/musl-cc \
  CGO_LDFLAGS="-L$PWD/shim-lib -lunwind -lc++" \
  go build -o my-static-binary ./cmd/my-app
```

The wrapper expects `zig` on `PATH`; `-lunwind` links Rust's musl
self-contained unwinder shipped in the tarball and `-lc++` resolves to
zig's bundled musl libc++, both required by the shim archive. The result
runs in distroless/static and scratch images.

## Design notes

- **Why a shim crate?** gigatoken is pip-distributed and its Python
  bindings (pyo3) compile in unconditionally. A static library must keep
  every public symbol of its dependency tree, so the pyo3 objects survive
  inside `libgigatoken_shim.a` even though nothing reachable from Go
  ever executes them. `rust/gen-pystub.sh` (run by `make rust`)
  regenerates `libpystub.a` — ~109 one-byte inert C variables satisfying
  the Python C-API symbol references of that dead code — so the link
  succeeds with no Python anywhere: none at build time, none at run
  time, none in the image. If a future gigatoken (or a fork) feature-gates
  pyo3, the stub regenerates as an empty archive and can be deleted.
- **Decode** goes through a copy of the public `vocab_entries()` table:
  upstream's `Tokenizer::decode` takes `&[TokenId]`, and `TokenId` lives
  in a `pub(crate)` module, so it's not nameable from outside the crate.
  Output is byte-identical (decode is just vocab-byte concatenation).
- **Single encode** takes a mutex and uses the tokenizer's persistent
  pretoken cache — the same path as gigatoken's own Python `encode`.
  **Batch encode** hands the tokenizer's cheap `fork()` (the vocab/merge
  tables are `Arc`-backed) to gigatoken's `WorkerPool` and fans out over
  rayon. Across the FFI boundary, batches travel as one contiguous byte
  buffer plus offsets — no pointer arrays — which keeps cgo's
  pointer-passing rules happy and lets corpus-holding callers pass
  buffers zero-copy via `EncodeBatchFlat`.
- All FFI entry points are `catch_unwind`-guarded: a Rust panic surfaces
  as a Go error via `ggt_last_error`, never as an unwind across the C
  boundary (which would abort the process).
- The gigatoken dep is pinned to a commit (`rev` in `rust/Cargo.toml`);
  the Rust API has no stability contract yet, so bump it deliberately.

## Static Linux builds

`make static-linux` (on a Linux host or CI runner) cross-builds the shim
for `x86_64-unknown-linux-musl` and compiles Go with the musl cross
compiler, producing a fully static, distroless-friendly binary. The
simdutf dependency needs a C++ cross compiler, which apt's `musl-tools`
doesn't ship — CI therefore uses zig for the whole toolchain:

```sh
make static-linux \
  MUSL_CC="zig cc -target x86_64-linux-musl" \
  MUSL_CXX="zig c++ -target x86_64-linux-musl"
```
Fat-LTO release builds are slow; cache `rust/target` (or just
`rust/lib/libgigatoken_shim.a`) in CI.
