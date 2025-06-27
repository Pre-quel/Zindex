/******************************************************************************
 # Zindex_o3

- Regex-free lazy pattern matching (NEON-accelerated BMH)
    (still faster than ugrep when stacked with normal grep as unix pipe)
- Bad-character shift table for maximizing skips.
- ARM NEON SIMD intrinsics (vld1q_u8, vceqq_u8) to compare 16-byte blocks in
parallel.
- Assumes patterns are ≤64 bytes. Memcmp for larger.
- The only reason this file is .cpp and not .c is because of simdjson >:(
    ... and I'm not that mad, becuase CJSON, ugh ...
    ... ok std::vector is convenient too sometimes ...
- Lockless ring buffer for worker-printer thread ops. (Hated every moment of
that.)
- Parallel subchunk processing because. Otpimization out of spite, at this
point.
- Data prefetched (__builtin_prefetch) for min. cache misses
- Way too burnt out to think abouts stacking regex on top of this abomination.
- Bloom filter is out. I did it all wrong, I think.
 ******************************************************************************/

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

extern "C" {
#include <zstd.h>
}

#include "simdjson.h"
using namespace simdjson;

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <sys/mman.h>
#endif
#endif

#define MAX_THREADS 8
#define STREAMING_WINDOW_SIZE (1024 * 1024)  
#define COMPRESSED_BUFFER_SIZE (256 * 1024)  
#define MAX_LINE_PREVIEW 4096
#define MAX_FILENAME_LEN 512
#define ASCII_SET_SIZE 256
#define MAX_PATTERN_SIZE 1024  


static pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

/*****************************************************************************
 * SIMD / NEON utilities
 *****************************************************************************/
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline bool neon_compare_aarch64(const char *text, const char *pat,
                                        size_t len) {
  if (len < 64)
    return memcmp(text, pat, len) == 0;

  const uint8x16_t *t = (const uint8x16_t *)__builtin_assume_aligned(text, 16);
  const uint8x16_t *p = (const uint8x16_t *)__builtin_assume_aligned(pat, 16);

  uint64x2_t cmp_accum = vdupq_n_u64(UINT64_MAX);

  for (; len >= 64; len -= 64, t += 4, p += 4) {
    uint8x16_t t0 = vld1q_u8((const uint8_t *)t);
    uint8x16_t t1 = vld1q_u8((const uint8_t *)(t + 1));
    uint8x16_t t2 = vld1q_u8((const uint8_t *)(t + 2));
    uint8x16_t t3 = vld1q_u8((const uint8_t *)(t + 3));

    uint8x16_t p0 = vld1q_u8((const uint8_t *)p);
    uint8x16_t p1 = vld1q_u8((const uint8_t *)(p + 1));
    uint8x16_t p2 = vld1q_u8((const uint8_t *)(p + 2));
    uint8x16_t p3 = vld1q_u8((const uint8_t *)(p + 3));

    uint8x16_t c0 = vceqq_u8(t0, p0);
    uint8x16_t c1 = vceqq_u8(t1, p1);
    uint8x16_t c2 = vceqq_u8(t2, p2);
    uint8x16_t c3 = vceqq_u8(t3, p3);

    uint8x16_t combined = vandq_u8(vandq_u8(c0, c1), vandq_u8(c2, c3));
    cmp_accum = vandq_u64(cmp_accum, vreinterpretq_u64_u8(combined));
  }

  return vgetq_lane_u64(cmp_accum, 0) == UINT64_MAX &&
         vgetq_lane_u64(cmp_accum, 1) == UINT64_MAX;
}
#endif

/*****************************************************************************
 * Chunk & Index Data
 *****************************************************************************/
typedef struct {
  int64_t chunk_id;
  int64_t compressed_offset;
  int64_t compressed_size;
  int64_t uncompressed_start;
  int64_t uncompressed_end;
} ChunkInfo;

typedef struct {
  ChunkInfo *chunks;
  size_t count;
} IndexInfo;

/*****************************************************************************
 * Thread-local context for streaming decompression
 *****************************************************************************/
typedef struct {
  ZSTD_DCtx *dctx;
  uint8_t *compressed_buf;    
  uint8_t *window_buf;        
  uint8_t *overlap_buf;       
  size_t overlap_size;        
} ThreadContext;

/*****************************************************************************
 * Boyer-Moore-Horspool
 *****************************************************************************/
static void build_badchar_table(const char *pattern, size_t patLen,
                                int badCharTable[ASCII_SET_SIZE]) {
  if (!pattern || patLen == 0 || !badCharTable) {
    return;
  }

  for (int i = 0; i < ASCII_SET_SIZE; i++) {
    badCharTable[i] = (int)patLen;
  }

  for (size_t i = 0; i < patLen - 1; i++) {
    unsigned char c = (unsigned char)pattern[i];
    badCharTable[c] = (int)(patLen - 1 - i);
  }
}

/*****************************************************************************
 * Extract and print matching line
 *****************************************************************************/
static void print_matching_line(const char *filename, const char *buffer, 
                               size_t buf_size, size_t match_pos,
                               int64_t base_offset) {
  
  size_t line_start = match_pos;
  while (line_start > 0 && buffer[line_start - 1] != '\n') {
    line_start--;
  }
  
  
  size_t line_end = match_pos;
  while (line_end < buf_size && buffer[line_end] != '\n' && buffer[line_end] != '\0') {
    line_end++;
  }
  
  
  size_t line_len = line_end - line_start;
  if (line_len == 0) return;
  
  
  pthread_mutex_lock(&print_mutex);
  
  
  printf("%s:%lld:", filename, (long long)(base_offset + match_pos));
  fwrite(buffer + line_start, 1, line_len, stdout);
  printf("\n");
  fflush(stdout);
  
  pthread_mutex_unlock(&print_mutex);
}

/*****************************************************************************
 * Streaming search with sliding window
 *****************************************************************************/
static void streaming_search(ThreadContext *ctx, const char *filename,
                           const char *buffer, size_t buffer_size,
                           const char *pattern, size_t pattern_len,
                           int64_t base_offset, int badCharTable[ASCII_SET_SIZE]) {
  size_t i = 0;
  
  while (i + pattern_len <= buffer_size) {
    
    ssize_t j = (ssize_t)pattern_len - 1;
    bool matched = true;
    
    while (j >= 0) {
      if (buffer[i + j] != pattern[j]) {
        matched = false;
        break;
      }
      j--;
    }
    
    if (matched) {
      print_matching_line(filename, buffer, buffer_size, i, base_offset);
      i++; 
    } else {
      
      unsigned char c = (unsigned char)buffer[i + pattern_len - 1];
      int shift = badCharTable[c];
      if (shift < 1) shift = 1;
      i += shift;
    }
  }
}

/*****************************************************************************
 * Streaming decompress and search a chunk
 *****************************************************************************/
static void streaming_decompress_and_search(ThreadContext *ctx,
                                          const char *zstFile,
                                          const ChunkInfo *chunk,
                                          const char *pattern,
                                          size_t pattern_len) {
  int fd = open(zstFile, O_RDONLY);
  if (fd < 0) return;
  
  
  ZSTD_DCtx_reset(ctx->dctx, ZSTD_reset_session_only);
  
  
  off_t read_offset = chunk->compressed_offset;
  size_t remaining_compressed = chunk->compressed_size;
  int64_t uncompressed_pos = chunk->uncompressed_start;
  
  ZSTD_inBuffer input = {ctx->compressed_buf, 0, 0};
  ZSTD_outBuffer output = {ctx->window_buf, STREAMING_WINDOW_SIZE, 0};
  
  
  int badCharTable[ASCII_SET_SIZE];
  build_badchar_table(pattern, pattern_len, badCharTable);
  
  
  size_t search_start = 0;
  if (ctx->overlap_size > 0) {
    
    memcpy(ctx->window_buf, ctx->overlap_buf, ctx->overlap_size);
    output.pos = ctx->overlap_size;
    search_start = 0;  
  }
  
  bool first_window = true;
  size_t last_newline_pos = 0;
  
  while (remaining_compressed > 0 || input.pos < input.size) {
    
    if (input.pos >= input.size && remaining_compressed > 0) {
      size_t to_read = (remaining_compressed < COMPRESSED_BUFFER_SIZE) 
                       ? remaining_compressed : COMPRESSED_BUFFER_SIZE;
      
      ssize_t bytes_read = pread(fd, ctx->compressed_buf, to_read, read_offset);
      if (bytes_read <= 0) break;
      
      input.size = bytes_read;
      input.pos = 0;
      read_offset += bytes_read;
      remaining_compressed -= bytes_read;
    }
    
    
    size_t ret = ZSTD_decompressStream(ctx->dctx, &output, &input);
    if (ZSTD_isError(ret)) break;
    
    
    if (output.pos >= STREAMING_WINDOW_SIZE - pattern_len || 
        (remaining_compressed == 0 && input.pos >= input.size)) {
      
      
      size_t search_end = output.pos;
      
      
      if (!first_window && ctx->overlap_size > 0) {
        search_start = ctx->overlap_size - pattern_len + 1;
        if (search_start > search_end) search_start = search_end;
      }
      
      
      if (search_end > search_start) {
        streaming_search(ctx, zstFile, 
                        (const char*)ctx->window_buf + search_start,
                        search_end - search_start,
                        pattern, pattern_len,
                        uncompressed_pos + search_start,
                        badCharTable);
      }
      
      
      last_newline_pos = output.pos;
      while (last_newline_pos > 0 && ctx->window_buf[last_newline_pos - 1] != '\n') {
        last_newline_pos--;
      }
      
      
      size_t overlap_start = (last_newline_pos > 0) ? last_newline_pos : output.pos - pattern_len + 1;
      if (overlap_start > output.pos) overlap_start = output.pos;
      
      ctx->overlap_size = output.pos - overlap_start;
      if (ctx->overlap_size > 0 && ctx->overlap_size <= MAX_LINE_PREVIEW) {
        memcpy(ctx->overlap_buf, ctx->window_buf + overlap_start, ctx->overlap_size);
      } else {
        ctx->overlap_size = 0;
      }
      
      
      uncompressed_pos += output.pos;
      output.pos = 0;
      first_window = false;
      search_start = 0;
    }
  }
  
  
  if (output.pos > search_start) {
    streaming_search(ctx, zstFile,
                    (const char*)ctx->window_buf + search_start,
                    output.pos - search_start,
                    pattern, pattern_len,
                    uncompressed_pos + search_start,
                    badCharTable);
  }
  
  close(fd);
}

/*****************************************************************************
 * Parse index
 *****************************************************************************/
static IndexInfo *parse_index_json(const char *idx_filename) {
  if (!idx_filename) return NULL;

  simdjson::dom::parser parser;
  try {
    simdjson::dom::element doc = parser.load(idx_filename);
    simdjson::dom::array chunksArr = doc["chunks"];

    size_t n = 0;
    for (auto elem : chunksArr) {
      n++;
    }

    if (n == 0) return NULL;

    IndexInfo *info = (IndexInfo *)calloc(1, sizeof(*info));
    if (!info) return NULL;

    info->count = n;
    info->chunks = (ChunkInfo *)calloc(n, sizeof(ChunkInfo));
    if (!info->chunks) {
      free(info);
      return NULL;
    }

    size_t i = 0;
    for (simdjson::dom::element chunk : chunksArr) {
      if (i >= n) break;

      ChunkInfo *C = &info->chunks[i];
      C->chunk_id = static_cast<int64_t>(chunk["chunk_id"]);
      C->compressed_offset = static_cast<int64_t>(chunk["compressed_offset"]);
      C->compressed_size = static_cast<int64_t>(chunk["compressed_size"]);
      C->uncompressed_start = static_cast<int64_t>(chunk["uncompressed_start"]);
      C->uncompressed_end = static_cast<int64_t>(chunk["uncompressed_end"]);
      i++;
    }
    return info;
  } catch (const simdjson::simdjson_error &e) {
    return NULL;
  }
}

static void free_index_info(IndexInfo *info) {
  if (!info) return;
  if (info->chunks) free(info->chunks);
  free(info);
}

/*****************************************************************************
 * Worker thread
 *****************************************************************************/
typedef struct {
  const char **zstFiles;
  const char **idxFiles;
  int fileCount;
  int start;
  int end;
  const char *pattern;
  size_t patLen;
  int thread_id;
} WorkerArg;

static void *worker_thread(void *arg) {
  WorkerArg *W = (WorkerArg *)arg;
  if (!W) return NULL;

  
  ThreadContext ctx = {0};
  ctx.dctx = ZSTD_createDCtx();
  ctx.compressed_buf = (uint8_t*)malloc(COMPRESSED_BUFFER_SIZE);
  ctx.window_buf = (uint8_t*)malloc(STREAMING_WINDOW_SIZE);
  ctx.overlap_buf = (uint8_t*)malloc(MAX_LINE_PREVIEW);
  
  if (!ctx.dctx || !ctx.compressed_buf || !ctx.window_buf || !ctx.overlap_buf) {
    
    if (ctx.dctx) ZSTD_freeDCtx(ctx.dctx);
    if (ctx.compressed_buf) free(ctx.compressed_buf);
    if (ctx.window_buf) free(ctx.window_buf);
    if (ctx.overlap_buf) free(ctx.overlap_buf);
    return NULL;
  }

  
  for (int i = W->start; i < W->end && i < W->fileCount; i++) {
    if (!W->zstFiles[i] || !W->idxFiles[i]) continue;

    IndexInfo *info = parse_index_json(W->idxFiles[i]);
    if (!info) continue;

    
    for (size_t j = 0; j < info->count; j++) {
      ctx.overlap_size = 0;  
      streaming_decompress_and_search(&ctx, W->zstFiles[i], 
                                    &info->chunks[j], 
                                    W->pattern, W->patLen);
    }

    free_index_info(info);
  }

  
  ZSTD_freeDCtx(ctx.dctx);
  free(ctx.compressed_buf);
  free(ctx.window_buf);
  free(ctx.overlap_buf);

  return NULL;
}

/*****************************************************************************
 * Main
 *****************************************************************************/
int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <pattern>\n", argv[0]);
    return 1;
  }

  const char *pattern = argv[1];
  size_t patLen = strlen(pattern);

  if (patLen == 0 || patLen > MAX_PATTERN_SIZE) {
    fprintf(stderr, "Error: Invalid search pattern\n");
    return 1;
  }

  
  DIR *d = opendir(".");
  if (!d) {
    fprintf(stderr, "Cannot open current directory: %s\n", strerror(errno));
    return 1;
  }

  const size_t initialCap = 64;
  char **zstFiles = (char **)malloc(initialCap * sizeof(*zstFiles));
  char **idxFiles = (char **)malloc(initialCap * sizeof(*idxFiles));

  if (!zstFiles || !idxFiles) {
    if (zstFiles) free(zstFiles);
    if (idxFiles) free(idxFiles);
    closedir(d);
    return 1;
  }

  size_t capacity = initialCap;
  size_t count = 0;

  struct dirent *de;
  while ((de = readdir(d)) != NULL) {
    if (fnmatch("batch_[0-9][0-9][0-9][0-9][0-9].tar.zst", de->d_name, 0) == 0) {
      char idxName[MAX_FILENAME_LEN];
      strncpy(idxName, de->d_name, sizeof(idxName) - 1);
      idxName[sizeof(idxName) - 1] = '\0';

      char *p = strstr(idxName, ".tar.zst");
      if (!p) continue;

      strcpy(p, ".tar.idx.json");

      struct stat st;
      if (stat(idxName, &st) == 0) {
        if (count >= capacity) {
          capacity *= 2;
          char **newZstFiles = (char **)realloc(zstFiles, capacity * sizeof(*zstFiles));
          char **newIdxFiles = (char **)realloc(idxFiles, capacity * sizeof(*idxFiles));

          if (!newZstFiles || !newIdxFiles) {
            for (size_t i = 0; i < count; i++) {
              free(zstFiles[i]);
              free(idxFiles[i]);
            }
            free(zstFiles);
            free(idxFiles);
            closedir(d);
            return 1;
          }

          zstFiles = newZstFiles;
          idxFiles = newIdxFiles;
        }

        zstFiles[count] = strdup(de->d_name);
        idxFiles[count] = strdup(idxName);

        if (!zstFiles[count] || !idxFiles[count]) {
          if (zstFiles[count]) free(zstFiles[count]);
          if (idxFiles[count]) free(idxFiles[count]);
          break;
        }

        count++;
      }
    }
  }
  closedir(d);

  if (count == 0) {
    fprintf(stderr, "No chunked .tar.zst/.idx.json pairs found.\n");
    free(zstFiles);
    free(idxFiles);
    return 0;
  }

  
  int threadCount = (count < MAX_THREADS) ? (int)count : MAX_THREADS;
  pthread_t *threads = (pthread_t *)malloc(threadCount * sizeof(pthread_t));
  WorkerArg *wargs = (WorkerArg *)calloc(threadCount, sizeof(WorkerArg));

  if (!threads || !wargs) {
    for (size_t i = 0; i < count; i++) {
      free(zstFiles[i]);
      free(idxFiles[i]);
    }
    free(zstFiles);
    free(idxFiles);
    if (threads) free(threads);
    if (wargs) free(wargs);
    return 1;
  }

  
  int filesPerThread = (int)count / threadCount;
  int remainder = (int)count % threadCount;
  int start = 0;

  for (int i = 0; i < threadCount; i++) {
    int load = filesPerThread + (i < remainder ? 1 : 0);
    wargs[i].zstFiles = (const char **)zstFiles;
    wargs[i].idxFiles = (const char **)idxFiles;
    wargs[i].fileCount = (int)count;
    wargs[i].start = start;
    wargs[i].end = start + load;
    wargs[i].pattern = pattern;
    wargs[i].patLen = patLen;
    wargs[i].thread_id = i;

    pthread_create(&threads[i], NULL, worker_thread, &wargs[i]);
    start += load;
  }

  
  for (int i = 0; i < threadCount; i++) {
    pthread_join(threads[i], NULL);
  }

  
  for (size_t i = 0; i < count; i++) {
    free(zstFiles[i]);
    free(idxFiles[i]);
  }
  free(zstFiles);
  free(idxFiles);
  free(threads);
  free(wargs);

  return 0;
}
