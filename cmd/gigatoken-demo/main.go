// Demo binary: load a tokenizer.json and encode each argument.
//
// Usage: go run ./cmd/gigatoken-demo /path/to/tokenizer.json "Hello, world!"
package main

import (
	"fmt"
	"os"

	"gigatoken-go/tokenizer"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "usage: %s <tokenizer.json> <text...>\n", os.Args[0])
		os.Exit(2)
	}
	tok, err := tokenizer.Load(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer tok.Close()
	fmt.Println("vocab size:", tok.VocabSize())
	for _, s := range os.Args[2:] {
		ids := tok.Encode(s)
		back, err := tok.Decode(ids)
		if err != nil {
			fmt.Fprintln(os.Stderr, "decode:", err)
			os.Exit(1)
		}
		fmt.Printf("%q -> %d tokens: %v\n  decoded: %q\n", s, len(ids), ids, back)
	}
}
