# zindex

---

**zindex** is a small C program that can search within `.tar.zst` archives using an accompanying `.idx.json` index file. It uses a Boyer–Moore search on each decompressed chunk. The archives must be named as `batch_XXXXX.tar.zst` with corresponding `batch_XXXXX.tar.idx.json` index files in the same directory. 

This project arose from a prototype that originally used Aho–Corasick but was adapted to Boyer–Moore for simplicity and processing speed.

## Features

- **Multi-threaded**: Processes multiple `.tar.zst` + `.tar.idx.json` pairs in parallel.
- **Chunk-based**: Only decompresses small parts of the archive at a time, using the offsets from the `.idx.json`.
- **Simple ring buffer**: (Optional) to handle boundary matches across chunk boundaries.
- **Minimal dependencies**: Uses `cJSON` for JSON parsing and `zstd` for decompression.

## Requirements

- A C compiler (`gcc` or `clang`)  
- `zstd` library and headers  
- `pthread` (almost always available on Linux/macOS)  
- `cJSON` (you can either compile it directly from the provided `cJSON.c` + `cJSON.h`, or install it system-wide if you prefer)

### Installing dependencies on Linux

For example, on Ubuntu or Debian-based distros:

```bash
sudo apt-get update
sudo apt-get install -y libzstd-dev gcc make
```

### Installing dependencies on macOS (Homebrew)

```bash
brew install zstd
brew install make       # if you need GNU make
```

*(Optionally you can install cJSON or just compile with the included `cJSON.c`.)*

## Building

1. Clone this repository or download the files `main.c`, `cJSON.c`, `cJSON.h`, and the `Makefile`.
2. Run:

   ```bash
   make
   ```

   This will produce an executable named `zindex`.

If you are on macOS and want to statically link against Homebrew’s ZSTD (`/opt/homebrew/lib/libzstd.a`), you might edit the `Makefile` (see the comments in the file) or compile manually:

```bash
clang -O3 -o zindex main cJSON.c \
    -I/opt/homebrew/include /opt/homebrew/lib/libzstd.a -lpthread
```

## Usage

1. **Ensure your archives are present** in the current directory:
   - `batch_00000.tar.zst`  
   - `batch_00000.tar.idx.json`  
   - `batch_00001.tar.zst`  
   - `batch_00001.tar.idx.json`  
   - etc.

2. **Run the search**:

   ```bash
   ./zindex <search-pattern>
   ```

   For example, if you want to search for `"secretkey"`:

   ```bash
   ./zindex secretkey
   ```

3. **View output**: By default, the program prints JSON lines to stdout for each match. You can redirect to a file:

   ```bash
   ./zindex secretkey > results.json
   ```

## Example Output

If a match is found, it will emit lines like:

```json
{"file":"batch_00000.tar.zst", "offset":12345, "score":7, "preview":"...some surrounding text..."}
```

*(Note: The “score” is just a “closeness” metric used in the code. We removed any threshold so you see all matches by default.)*

## Troubleshooting

1. **No output**: 
   - Check that your file names actually match the `batch_[0-9][0-9][0-9][0-9][0-9].tar.zst` pattern.
   - Make sure you run the program from the directory containing the archives.
   - Remove or lower the “closeness” threshold (already done in this version) to ensure matches aren’t being filtered out.
2. **Library errors**: If `zstd` is not found, you might need to adjust the `-L` or `-I` flags in the `Makefile`.

## License

(This is placeholder text — choose a license for your project, e.g., MIT, Apache 2.0, etc.)

```

---

## Makefile

```makefile
# Simple Makefile for zindex on Linux.
# If you're on macOS with Homebrew and want static linking, 
# you can replace -lzstd with /opt/homebrew/lib/libzstd.a 
# and add -I/opt/homebrew/include as needed.

CC      = gcc         # or clang
CFLAGS  = -O3 -Wall
LIBS    = -lzstd -lpthread

# If you prefer a static link against Homebrew zstd on macOS, do:
# LIBS = /opt/homebrew/lib/libzstd.a -lpthread
# CFLAGS += -I/opt/homebrew/include

all: zindex

zindex: zindex.o cJSON.o
	$(CC) $(CFLAGS) -o zindex zindex.o cJSON.o $(LIBS)

zindex.o: zindex.c cJSON.h
	$(CC) $(CFLAGS) -c zindex.c

cJSON.o: cJSON.c cJSON.h
	$(CC) $(CFLAGS) -c cJSON.c

clean:
	rm -f *.o zindex

.PHONY: clean
```

### Notes on the Makefile

- By default, it compiles with dynamic linking against `zstd` (`-lzstd`).  
- If you want to statically link to a `.a` library from Homebrew, change the line:

  ```makefile
  LIBS    = -lzstd -lpthread
  ```
  to something like:
  ```makefile
  LIBS    = /opt/homebrew/lib/libzstd.a -lpthread
  CFLAGS += -I/opt/homebrew/include
  ```
- On many Linux systems, installing `libzstd-dev` means you can just do `-lzstd`.  

After building successfully, you’ll have a `zindex` executable in the same directory.
