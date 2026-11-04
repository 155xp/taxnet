// go/scanner throughput on the files of a taxnet pack (single thread).
package main

import (
	"encoding/json"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"time"
)

func main() {
	base := os.Args[1]
	var idx struct{ Files [][3]any `json:"files"` }
	b, _ := os.ReadFile(base + ".idx.json")
	json.Unmarshal(b, &idx)
	src, _ := os.ReadFile(base + ".src")
	best := 1e9
	for rep := 0; rep < 3; rep++ {
		t := time.Now()
		ntok := 0
		for _, f := range idx.Files {
			off, n := int(f[1].(float64)), int(f[2].(float64))
			fset := token.NewFileSet()
			file := fset.AddFile("", -1, n)
			var s scanner.Scanner
			s.Init(file, src[off:off+n], nil, scanner.ScanComments)
			for {
				_, tok, _ := s.Scan()
				if tok == token.EOF {
					break
				}
				ntok++
			}
		}
		if d := time.Since(t).Seconds(); d < best {
			best = d
		}
	}
	fmt.Printf("go/scanner: %.1f M chars/s\n", float64(len(src))/best/1e6)
}
