// Package tokenizer embeds the gigatoken Rust tokenizer behind a small C
// API, so a single Go binary can run BPE tokenization with no Python in
// the image.
//
// The Rust side must be built first (see the Makefile): it produces
// rust/lib/libgigatoken_shim.a, which this package links against.
package tokenizer

/*
#cgo LDFLAGS: -L${SRCDIR}/../rust/lib -lgigatoken_shim -lpystub -lm -lpthread
#include <stdint.h>
#include <stdlib.h>

typedef struct GigaTokenizer GigaTokenizer;

typedef struct {
	uint32_t *ids;
	size_t    n_ids;
	int64_t  *offsets; // n_docs + 1 cumulative token offsets
	size_t    n_offsets;
} GgtBatch;

typedef struct {
	uint8_t *data;
	size_t   len;
} GgtBytes;

GigaTokenizer *ggt_load_hf(const char *path);
void           ggt_free(GigaTokenizer *t);
size_t         ggt_vocab_size(const GigaTokenizer *t);
size_t         ggt_encode(const GigaTokenizer *t, const uint8_t *bytes, size_t len,
                          uint32_t *out, size_t cap);
GgtBatch      *ggt_encode_batch(const GigaTokenizer *t, const uint8_t *flat,
                                size_t flat_len, const size_t *offs, size_t n_docs);
void           ggt_free_batch(GgtBatch *b);
GgtBytes      *ggt_decode(const GigaTokenizer *t, const uint32_t *ids, size_t n);
void           ggt_free_bytes(GgtBytes *b);
const char    *ggt_last_error(void);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Tokenizer is a loaded gigatoken BPE tokenizer. It is safe for
// concurrent use by multiple goroutines: single-doc encode serializes
// on an internal lock (with a persistent pretoken cache), batch encode
// fans out over gigatoken's rayon worker pool, and the vocab tables are
// shared behind atomically refcounted pointers on the Rust side.
//
// Close must be called exactly once, when the tokenizer is no longer
// needed; not calling it leaks the vocab and merge tables.
type Tokenizer struct {
	ptr *C.GigaTokenizer
}

// Load reads a HuggingFace ByteLevel-BPE tokenizer.json (GPT-2, RoBERTa,
// ...) from path. For SentencePiece/byte_fallback tokenizers or
// tiktoken rank files, the shim would need to grow the corresponding
// loader; upstream supports both.
func Load(path string) (*Tokenizer, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	ptr := C.ggt_load_hf(cpath)
	if ptr == nil {
		return nil, errFromShim()
	}
	return &Tokenizer{ptr: ptr}, nil
}

// Close releases the tokenizer's memory. The Tokenizer must not be used
// afterwards. Calling Close twice is safe (the second is a no-op).
func (t *Tokenizer) Close() {
	if t != nil && t.ptr != nil {
		C.ggt_free(t.ptr)
		t.ptr = nil
	}
}

// VocabSize returns the tokenizer's vocab size.
func (t *Tokenizer) VocabSize() int {
	return int(C.ggt_vocab_size(t.ptr))
}

// Encode tokenizes one document. It never allocates more than twice and
// retries at most once, since the second call hits the Rust-side pretoken
// cache.
func (t *Tokenizer) Encode(s string) []uint32 {
	// Avoid copying the string: encode treats input as raw bytes.
	return t.EncodeBytes(unsafe.Slice(unsafe.StringData(s), len(s)))
}

// EncodeBytes tokenizes one byte document.
func (t *Tokenizer) EncodeBytes(b []byte) []uint32 {
	// Token count never exceeds ~input length; +16 headroom for added
	// tokens; the retry loop below covers the pathological cases anyway.
	out := make([]uint32, len(b)+16)
	n := t.encodeInto(b, out)
	if n > len(out) {
		out = make([]uint32, n)
		n = t.encodeInto(b, out)
		if n > len(out) {
			// Can't happen: the retry used the size the shim reported.
			panic("tokenizer: encode size changed between calls")
		}
	}
	return out[:n:n]
}

func (t *Tokenizer) encodeInto(in []byte, out []uint32) int {
	var p *C.uint8_t
	if len(in) > 0 {
		p = (*C.uint8_t)(unsafe.Pointer(&in[0]))
	}
	return int(C.ggt_encode(t.ptr, p, C.size_t(len(in)),
		(*C.uint32_t)(unsafe.Pointer(&out[0])), C.size_t(len(out))))
}

// BatchResult is the outcome of EncodeBatch: flat ids plus cumulative
// offsets (len = NumDocs + 1), mirroring gigatoken's ragged batch format.
// IDs returns per-document views into the flat id slice.
type BatchResult struct {
	ids     []uint32
	offsets []int64
}

// NumDocs returns the number of documents in the batch.
func (b *BatchResult) NumDocs() int { return len(b.offsets) - 1 }

// IDs returns the token ids of document i.
func (b *BatchResult) IDs(i int) []uint32 {
	return b.ids[b.offsets[i]:b.offsets[i+1]:b.offsets[i+1]]
}

// FlatIDs returns all ids, concatenated in document order.
func (b *BatchResult) FlatIDs() []uint32 { return b.ids }

// Offsets returns the cumulative token offsets, len(NumDocs()+1).
func (b *BatchResult) Offsets() []int64 { return b.offsets }

// EncodeBatch tokenizes documents in parallel on gigatoken's rayon
// pool. Doc order is preserved in the flat id output. The rayon pool is
// process-global and sized once (honor RAYON_NUM_THREADS before the
// first call if you want to bound it).
func (t *Tokenizer) EncodeBatch(docs [][]byte) (*BatchResult, error) {
	if len(docs) == 0 {
		return &BatchResult{offsets: []int64{0}}, nil
	}
	total := 0
	for _, d := range docs {
		total += len(d)
	}
	flat := make([]byte, 0, total)
	offs := make([]int64, len(docs)+1)
	for i, d := range docs {
		flat = append(flat, d...)
		offs[i+1] = int64(len(flat))
	}
	return t.EncodeBatchFlat(flat, offs)
}

// EncodeBatchFlat is the zero-copy variant of EncodeBatch for callers
// that already hold their documents contiguously: document i is
// flat[offs[i]:offs[i+1]], with offs[0] == 0, offs[len-1] == len(flat),
// monotonically non-decreasing.
func (t *Tokenizer) EncodeBatchFlat(flat []byte, offs []int64) (*BatchResult, error) {
	if len(offs) == 0 {
		return nil, fmt.Errorf("gigatoken: offs needs at least one entry")
	}
	n := len(offs) - 1
	if offs[0] != 0 || offs[n] != int64(len(flat)) {
		return nil, fmt.Errorf("gigatoken: offs[0] must be 0 and offs[len-1] must equal len(flat)")
	}
	for i := 0; i < n; i++ {
		if offs[i+1] < offs[i] {
			return nil, fmt.Errorf("gigatoken: offs not monotonic at doc %d", i)
		}
	}
	cOffs := make([]C.size_t, len(offs))
	for i, o := range offs {
		cOffs[i] = C.size_t(o)
	}
	var flatPtr *C.uint8_t
	if len(flat) > 0 {
		flatPtr = (*C.uint8_t)(unsafe.Pointer(&flat[0]))
	}
	res := C.ggt_encode_batch(t.ptr, flatPtr, C.size_t(len(flat)),
		&cOffs[0], C.size_t(n))
	if res == nil {
		return nil, errFromShim()
	}
	defer C.ggt_free_batch(res)

	ids := unsafe.Slice((*uint32)(unsafe.Pointer(res.ids)), int(res.n_ids))
	offsets := unsafe.Slice((*int64)(unsafe.Pointer(res.offsets)), int(res.n_offsets))
	return &BatchResult{
		ids:     append([]uint32(nil), ids...),
		offsets: append([]int64(nil), offsets...),
	}, nil
}

// Decode turns token ids back into raw (UTF-8, byte-level) bytes.
// It errors on ids outside the vocab.
func (t *Tokenizer) Decode(ids []uint32) ([]byte, error) {
	var in *C.uint32_t
	if len(ids) > 0 {
		in = (*C.uint32_t)(unsafe.Pointer(&ids[0]))
	}
	res := C.ggt_decode(t.ptr, in, C.size_t(len(ids)))
	if res == nil {
		return nil, errFromShim()
	}
	defer C.ggt_free_bytes(res)
	return C.GoBytes(unsafe.Pointer(res.data), C.int(res.len)), nil
}

func errFromShim() error {
	msg := C.GoString(C.ggt_last_error())
	if msg == "" {
		msg = "unknown error"
	}
	return fmt.Errorf("gigatoken: %s", msg)
}
