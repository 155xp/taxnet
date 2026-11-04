// Go reference labeler: the standard library's go/scanner + the rules in ref/spec.py.
//
//	go run ./ref/go -spec spec -list files.txt -out labels.bin > status.txt
//
// For each path in the list it prints "OK" or "ERR <reason>" and, for OK files, appends
// one class byte per source byte to the output file.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

const (
	WS byte = iota
	KW
	TYPE
	BUILTIN
	FUNC
	IDENT
	NUM
	STR
	COM
	OP
	META
)

func readWords(dir, name string) map[string]bool {
	b, err := os.ReadFile(filepath.Join(dir, name+".txt"))
	if err != nil {
		panic(err)
	}
	m := map[string]bool{}
	for _, w := range strings.Fields(string(b)) {
		m[w] = true
	}
	return m
}

var types, builtins map[string]bool

type tok struct {
	off, end int
	t        token.Token
}

func label(src []byte) ([]byte, string) {
	if bytes.IndexByte(src, 0) >= 0 {
		return nil, "nul"
	}
	fset := token.NewFileSet()
	file := fset.AddFile("x.go", -1, len(src))
	var errmsg string
	var s scanner.Scanner
	s.Init(file, src, func(pos token.Position, msg string) {
		if errmsg == "" {
			errmsg = fmt.Sprintf("%d:%d: %s", pos.Line, pos.Column, msg)
		}
	}, scanner.ScanComments)
	var toks []tok
	for {
		pos, t, lit := s.Scan()
		if t == token.EOF {
			break
		}
		if t == token.SEMICOLON && lit == "\n" {
			continue // automatically inserted, covers no source bytes
		}
		off := file.Offset(pos)
		var end int
		switch {
		case t == token.COMMENT && src[off+1] == '/':
			end = off + bytes.IndexByte(src[off:], '\n')
			if end < off {
				end = len(src)
			}
		case t == token.COMMENT:
			end = off + 2 + bytes.Index(src[off+2:], []byte("*/")) + 2
		case t == token.STRING && src[off] == '`':
			end = off + 1 + bytes.IndexByte(src[off+1:], '`') + 1
		case lit != "":
			end = off + len(lit) // the scanner only rewrites CRs, handled above
		default:
			end = off + len(t.String())
		}
		toks = append(toks, tok{off, end, t})
	}
	if errmsg != "" {
		return nil, errmsg
	}
	lab := make([]byte, len(src))
	for i, k := range toks {
		var c byte
		switch {
		case k.t == token.COMMENT:
			c = COM
		case k.t == token.IDENT:
			name := string(src[k.off:k.end])
			afterDot := i > 0 && toks[i-1].t == token.PERIOD
			switch {
			case types[name] && !afterDot:
				c = TYPE
			case builtins[name] && !afterDot:
				c = BUILTIN
			case i+1 < len(toks) && toks[i+1].t == token.LPAREN &&
				len(bytes.Trim(src[k.end:toks[i+1].off], " \t")) == 0:
				c = FUNC
			default:
				c = IDENT
			}
		case k.t.IsKeyword():
			c = KW
		case k.t == token.INT || k.t == token.FLOAT || k.t == token.IMAG:
			c = NUM
		case k.t == token.CHAR || k.t == token.STRING:
			c = STR
		case k.t == token.ILLEGAL:
			return nil, "illegal token"
		default:
			c = OP
		}
		for j := k.off; j < k.end; j++ {
			lab[j] = c
		}
	}
	return lab, ""
}

func main() {
	spec := flag.String("spec", "spec", "spec directory")
	list := flag.String("list", "", "file with one source path per line")
	out := flag.String("out", "", "label output file")
	flag.Parse()
	types = readWords(*spec, "go_types")
	builtins = readWords(*spec, "go_builtins")
	lf, err := os.Open(*list)
	if err != nil {
		panic(err)
	}
	of, err := os.Create(*out)
	if err != nil {
		panic(err)
	}
	w := bufio.NewWriterSize(of, 1<<20)
	status := bufio.NewWriter(os.Stdout)
	sc := bufio.NewScanner(lf)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for sc.Scan() {
		src, err := os.ReadFile(sc.Text())
		if err != nil {
			panic(err)
		}
		lab, msg := label(src)
		if msg != "" {
			fmt.Fprintf(status, "ERR %s\n", strings.ReplaceAll(msg, "\n", " "))
			continue
		}
		w.Write(lab)
		fmt.Fprintln(status, "OK")
	}
	w.Flush()
	of.Close()
	status.Flush()
}
