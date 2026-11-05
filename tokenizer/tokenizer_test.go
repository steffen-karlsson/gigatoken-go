package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

// testTokenizerPath can point at any HF ByteLevel-BPE tokenizer.json;
// the Makefile downloads GPT-2's into testdata/.
func testTokenizerPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GGT_TEST_TOKENIZER")
	if path == "" {
		path = filepath.Join("..", "testdata", "gpt2", "tokenizer.json")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("tokenizer.json not found at %s — run `make testdata` to download GPT-2's", path)
	}
	return path
}

func loadTestTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := Load(testTokenizerPath(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(tok.Close)
	return tok
}

func TestVocabSize(t *testing.T) {
	tok := loadTestTokenizer(t)
	if got := tok.VocabSize(); got != 50257 {
		t.Fatalf("GPT-2 vocab size = %d, want 50257", got)
	}
}

func TestEncodeGPT2(t *testing.T) {
	tok := loadTestTokenizer(t)
	got := tok.Encode("Hello, world!")
	want := []uint32{15496, 11, 995, 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode = %v, want %v", got, want)
	}
}

func TestRoundTrip(t *testing.T) {
	tok := loadTestTokenizer(t)
	inputs := []string{
		"",
		"Hello, world!",
		"the quick brown fox jumps over the lazy dog",
		"héllo wörld — ünïcode ✓ 你好, 世界 🚀",
		"\n\t\r\x00\x01binary-ish\xff bytes",
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	for _, in := range inputs {
		ids := tok.Encode(in)
		back, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode(%q): %v", in, err)
		}
		if string(back) != in {
			t.Fatalf("round trip of %q gave %q (ids %v)", in, back, ids)
		}
	}
}

func TestDecodeRejectsOutOfRange(t *testing.T) {
	tok := loadTestTokenizer(t)
	if _, err := tok.Decode([]uint32{0, 999_999}); err == nil {
		t.Fatal("Decode of out-of-range id unexpectedly succeeded")
	}
}

func TestEncodeBatchMatchesSingle(t *testing.T) {
	tok := loadTestTokenizer(t)
	docs := [][]byte{
		[]byte("Hello, world!"),
		[]byte(""),
		[]byte("a second document, somewhat longer, with more words in it."),
		[]byte("unicode too: 你好, 世界"),
		[]byte("x"),
	}
	batch, err := tok.EncodeBatch(docs)
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	if batch.NumDocs() != len(docs) {
		t.Fatalf("NumDocs = %d, want %d", batch.NumDocs(), len(docs))
	}
	var singleTotal int
	for i, d := range docs {
		want := tok.EncodeBytes(d)
		got := batch.IDs(i)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("doc %d batch ids = %v, want %v", i, got, want)
		}
		singleTotal += len(want)
	}
	if len(batch.FlatIDs()) != singleTotal {
		t.Fatalf("flat ids len = %d, want %d", len(batch.FlatIDs()), singleTotal)
	}
	if last := batch.Offsets()[len(docs)]; last != int64(singleTotal) {
		t.Fatalf("last offset = %d, want %d", last, singleTotal)
	}
}

func TestEncodeBatchEmpty(t *testing.T) {
	tok := loadTestTokenizer(t)
	batch, err := tok.EncodeBatch(nil)
	if err != nil {
		t.Fatalf("EncodeBatch(nil): %v", err)
	}
	if batch.NumDocs() != 0 || len(batch.FlatIDs()) != 0 {
		t.Fatalf("empty batch = %d docs, %d ids", batch.NumDocs(), len(batch.FlatIDs()))
	}
}

func TestConcurrentEncode(t *testing.T) {
	tok := loadTestTokenizer(t)
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				in := "concurrent encode round trip ✓"
				ids := tok.Encode(in)
				back, err := tok.Decode(ids)
				if err != nil {
					errs <- err
					return
				}
				if string(back) != in {
					errs <- errRoundTrip
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

var errRoundTrip = &roundTripError{}

type roundTripError struct{}

func (*roundTripError) Error() string { return "concurrent round trip mismatch" }
