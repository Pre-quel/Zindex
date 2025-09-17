/*
    zindex.cpp — tiny zstd "grep-ish" searcher over chunked .tar.zst archives.

    This version just fixes the C++ goto issue and uses
    ZSTD_DCtx_reset() instead of ZSTD_resetDStream().
    Bloom filter functionality has been removed.
    Optimized to reuse buffers across chunks for better performance.
    Added -d flag to specify search directory.

    — Emma
*/

#define _FILE_OFFSET_BITS 64

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
#include <limits.h>

extern "C"
{
#include <zstd.h>
}

#include "simdjson.h"
using namespace simdjson;

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <sys/mman.h>
#endif
#endif

#define MAX_THREADS 32
#define MAX_QUEUE_SIZE 8192ULL
#define MAX_LINE_PREVIEW 2048
#define MAX_FILENAME_LEN 512
#define ASCII_SET_SIZE 256

#define ZSTD_IN_CHUNK (1u << 20) /* 1 MiB compressed feed */
#define ZSTD_OUT_WIN (1u << 20)  /* 1 MiB decompressed window */

#if defined(U_ICU_VERSION_MAJOR_NUM)
#include <unicode/utypes.h>
#include <unicode/uchar.h>
static inline unsigned char u_tolower_byte(unsigned char c)
{
  return (unsigned char)u_tolower((UChar32)c);
}
#else
static inline unsigned char u_tolower_byte(unsigned char c)
{
  return (unsigned char)tolower((int)c);
}
#endif

typedef struct
{
  const char *pat;
  size_t len;
  int bad[ASCII_SET_SIZE];
} PatCtx;

static PatCtx g_pat;

static void build_badchar_table(const char *pattern, size_t patLen, int bad[ASCII_SET_SIZE])
{
  if (!pattern || patLen == 0)
    return;
  for (int i = 0; i < ASCII_SET_SIZE; i++)
    bad[i] = (int)patLen;
  for (size_t i = 0; i + 1 < patLen; i++)
    bad[(unsigned char)pattern[i]] = (int)(patLen - 1 - i);
}

static inline bool bytes_equal(const char *a, const char *b, size_t len)
{
#if HAVE_NEON
  while (len >= 16)
  {
    uint8x16_t va = vld1q_u8((const uint8_t *)a);
    uint8x16_t vb = vld1q_u8((const uint8_t *)b);
    uint8x16_t cmp = vceqq_u8(va, vb);
    uint64x2_t lanes = vreinterpretq_u64_u8(cmp);
    if ((vgetq_lane_u64(lanes, 0) != UINT64_MAX) || (vgetq_lane_u64(lanes, 1) != UINT64_MAX))
      return false;
    a += 16;
    b += 16;
    len -= 16;
  }
  for (size_t i = 0; i < len; ++i)
    if ((unsigned char)a[i] != (unsigned char)b[i])
      return false;
  return true;
#else
  return memcmp(a, b, len) == 0;
#endif
}

static inline ssize_t BMH_find(const char *hay, size_t hayLen,
                               const char *pat, size_t patLen,
                               const int bad[256])
{
  if (!hay || !pat || patLen == 0 || hayLen < patLen)
    return -1;
  size_t i = 0;
  while (i <= hayLen - patLen)
  {
    bool match = false;
    if (patLen <= 16)
    {
      match = bytes_equal(hay + i, pat, patLen);
    }
    else
    {
      if (bytes_equal(hay + i + patLen - 16, pat + patLen - 16, 16))
      {
        size_t j = 0;
        while (j + 16 < patLen && hay[i + j] == pat[j])
          ++j;
        match = (j + 16 >= patLen);
      }
    }
    if (match)
      return (ssize_t)i;
    unsigned char last = (unsigned char)hay[i + patLen - 1];
    int shift = bad[last];
    i += (shift > 0) ? shift : 1;
  }
  return -1;
}

typedef struct
{
  char filename[MAX_FILENAME_LEN];
  char internalFilename[MAX_FILENAME_LEN];
  int64_t offset;
  int closenessScore;
  char preview[MAX_LINE_PREVIEW];
} SearchResult;

typedef struct PrintNode
{
  SearchResult item;
  struct PrintNode *next;
} PrintNode;

typedef struct
{
  PrintNode *head, *tail;
  size_t size, max_size;
  bool done;
  pthread_mutex_t lock;
  pthread_cond_t not_empty;
  pthread_cond_t not_full;
} PrintQueue;

static void printqueue_init(PrintQueue *q)
{
  memset(q, 0, sizeof(*q));
  q->max_size = MAX_QUEUE_SIZE;
  pthread_mutex_init(&q->lock, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
}
static void printqueue_destroy(PrintQueue *q)
{
  pthread_mutex_lock(&q->lock);
  for (PrintNode *cur = q->head; cur;)
  {
    PrintNode *n = cur->next;
    free(cur);
    cur = n;
  }
  q->head = q->tail = NULL;
  q->size = 0;
  pthread_mutex_unlock(&q->lock);
  pthread_mutex_destroy(&q->lock);
  pthread_cond_destroy(&q->not_empty);
  pthread_cond_destroy(&q->not_full);
}
static bool printqueue_push(PrintQueue *q, const SearchResult *res)
{
  PrintNode *node = (PrintNode *)malloc(sizeof(*node));
  if (!node)
    return false;
  memcpy(&node->item, res, sizeof(*res));
  node->item.filename[sizeof(node->item.filename) - 1] = '\0';
  node->item.preview[sizeof(node->item.preview) - 1] = '\0';
  node->next = NULL;
  pthread_mutex_lock(&q->lock);
  while (q->size >= q->max_size && !q->done)
    pthread_cond_wait(&q->not_full, &q->lock);
  if (q->done)
  {
    pthread_mutex_unlock(&q->lock);
    free(node);
    return false;
  }
  if (q->tail)
    q->tail->next = node;
  else
    q->head = node;
  q->tail = node;
  q->size++;
  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->lock);
  return true;
}
static bool printqueue_pop(PrintQueue *q, SearchResult *out)
{
  pthread_mutex_lock(&q->lock);
  while (q->size == 0 && !q->done)
    pthread_cond_wait(&q->not_empty, &q->lock);
  if (q->size == 0 && q->done)
  {
    pthread_mutex_unlock(&q->lock);
    return false;
  }
  PrintNode *node = q->head;
  q->head = node->next;
  if (!q->head)
    q->tail = NULL;
  q->size--;
  memcpy(out, &node->item, sizeof(*out));
  free(node);
  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->lock);
  return true;
}
static void printqueue_mark_done(PrintQueue *q)
{
  pthread_mutex_lock(&q->lock);
  q->done = true;
  pthread_cond_broadcast(&q->not_empty);
  pthread_cond_broadcast(&q->not_full);
  pthread_mutex_unlock(&q->lock);
}

static void *printer_thread(void *arg)
{
  PrintQueue *q = (PrintQueue *)arg;
  char line[MAX_LINE_PREVIEW + MAX_FILENAME_LEN + 128];
  SearchResult sr;
  setvbuf(stdout, NULL, _IOFBF, 1 << 20);
  while (printqueue_pop(q, &sr))
  {
    int n = snprintf(line, sizeof(line), "%s\n", sr.preview);
    if (n > 0)
      (void)!write(STDOUT_FILENO, line, (size_t)n);
  }
  return NULL;
}

static void make_preview_and_score(const char *text, size_t textLen,
                                   size_t matchPos, size_t /*patLen*/,
                                   const char * /*pattern*/, SearchResult *outRes)
{
  if (!text || !outRes || matchPos >= textLen)
  {
    if (outRes)
      outRes->preview[0] = '\0';
    return;
  }
  size_t start = matchPos, end = matchPos;
  while (start > 0 && text[start - 1] != '\n')
    --start;
  while (end < textLen && text[end] != '\n' && text[end] != '\0')
    ++end;
  size_t len = end > start ? (end - start) : 0;
  if (len >= MAX_LINE_PREVIEW - 1)
    len = MAX_LINE_PREVIEW - 1;
  memcpy(outRes->preview, text + start, len);
  outRes->preview[len] = '\0';
  outRes->internalFilename[0] = '\0';
  const char *exts[] = {".txt", ".csv", ".json", ".md", ".log", NULL};
  for (int i = 0; exts[i]; ++i)
  {
    char *pos = strstr(outRes->preview, exts[i]);
    if (!pos)
      continue;
    char *begin = pos;
    while (begin > outRes->preview)
    {
      char c = *(begin - 1);
      if (c == ' ' || c == '/' || c == '\\' || c == ':')
        break;
      --begin;
    }
    size_t fLen = (size_t)(pos - begin + strlen(exts[i]));
    if (fLen && fLen < MAX_FILENAME_LEN)
    {
      strncpy(outRes->internalFilename, begin, fLen);
      outRes->internalFilename[fLen] = '\0';
    }
    break;
  }
  outRes->closenessScore = 0;
}

typedef struct
{
  int64_t chunk_id;
  int64_t compressed_offset;
  int64_t compressed_size;
  int64_t uncompressed_start;
  int64_t uncompressed_end;
} ChunkInfo;

typedef struct
{
  ChunkInfo *chunks;
  size_t count;
} IndexInfo;

static IndexInfo *parse_index_json(const char *idx_filename)
{
  simdjson::dom::parser parser;
  try
  {
    dom::element doc = parser.load(idx_filename);
    dom::array arr = doc["chunks"];
    size_t n = 0;
    for (auto _ : arr)
      (void)_, ++n;
    if (!n)
      return NULL;
    IndexInfo *info = (IndexInfo *)calloc(1, sizeof(*info));
    info->count = n;
    info->chunks = (ChunkInfo *)calloc(n, sizeof(ChunkInfo));
    if (!info->chunks)
    {
      free(info);
      return NULL;
    }
    size_t i = 0;
    for (dom::element e : arr)
    {
      if (i >= n)
        break;
      ChunkInfo *C = &info->chunks[i++];
      C->chunk_id = (int64_t)e["chunk_id"];
      C->compressed_offset = (int64_t)e["compressed_offset"];
      C->compressed_size = (int64_t)e["compressed_size"];
      C->uncompressed_start = (int64_t)e["uncompressed_start"];
      C->uncompressed_end = (int64_t)e["uncompressed_end"];
    }
    return info;
  }
  catch (const simdjson_error &err)
  {
    fprintf(stderr, "Index parse error: %s\n", err.what());
    return NULL;
  }
}
static void free_index_info(IndexInfo *info)
{
  if (!info)
    return;
  if (info->chunks)
    free(info->chunks);
  free(info);
}

static void search_in_buffer(PrintQueue *pq,
                             const char *filename,
                             const char *text,
                             size_t textLen,
                             const PatCtx *ctx,
                             int64_t baseOff)
{
  if (!pq || !text || !ctx || !ctx->pat || ctx->len == 0 || textLen < ctx->len)
    return;
  const int *bad = ctx->bad;
  const size_t plen = ctx->len;
  const char *pat = ctx->pat;
  size_t offset = 0;
  size_t hits = 0;
  const size_t MAX_HITS = 1000;
  while (offset + plen <= textLen && hits < MAX_HITS)
  {
    ssize_t rel = BMH_find(text + offset, textLen - offset, pat, plen, bad);
    if (rel < 0)
      break;
    size_t pos = offset + (size_t)rel;
    SearchResult sr = {};
    strncpy(sr.filename, filename, sizeof(sr.filename) - 1);
    sr.offset = baseOff + (int64_t)pos;
    make_preview_and_score(text, textLen, pos, plen, pat, &sr);
    if (sr.preview[0] != '\0')
    {
      if (!printqueue_push(pq, &sr))
      {
        fprintf(stderr, "[warn] print queue saturated; dropping rest of this buffer\n");
        break;
      }
      ++hits;
    }
    offset = pos + 1;
  }
}

static void stream_decompress_and_search(PrintQueue *pq,
                                         int fd,
                                         const ChunkInfo *c,
                                         const PatCtx *ctx,
                                         const char *archive,
                                         char *buf,
                                         char *inbuf,
                                         ZSTD_DCtx *dctx)
{
  const size_t OVER = (ctx->len > 1) ? (ctx->len - 1) : 0;

  memset(buf, 0, OVER);
  char *dst = buf + OVER;

  int64_t absOff = c->uncompressed_start;
  off_t file_off = (off_t)c->compressed_offset;
  size_t remaining = (size_t)c->compressed_size;
  bool first = true;

  while (remaining > 0)
  {
    size_t to_read = remaining < ZSTD_IN_CHUNK ? remaining : ZSTD_IN_CHUNK;
    ssize_t r = pread(fd, inbuf, to_read, file_off);
    if (r <= 0)
    {
      fprintf(stderr, "pread failed on %s (chunk %lld): %s\n", archive, (long long)c->chunk_id, strerror(errno));
      break;
    }
    file_off += (off_t)r;
    remaining -= (size_t)r;

    ZSTD_inBuffer in = {inbuf, (size_t)r, 0};

    while (in.pos < in.size)
    {
      ZSTD_outBuffer out = {dst, ZSTD_OUT_WIN, 0};
      size_t ret = ZSTD_decompressStream(dctx, &out, &in);
      if (ZSTD_isError(ret))
      {
        fprintf(stderr, "zstd error on %s chunk %lld: %s\n", archive, (long long)c->chunk_id, ZSTD_getErrorName(ret));
        /* Reset session if available, then continue with next input slice */
#if defined(ZSTD_VERSION_MAJOR)
        (void)ZSTD_DCtx_reset(dctx, ZSTD_reset_session_only);
#endif
        break;
      }

      if (out.pos > 0)
      {
        char *search_ptr = first ? dst : buf;
        size_t search_len = first ? out.pos : OVER + out.pos;
        int64_t base = first ? absOff : absOff - (int64_t)OVER;

        search_in_buffer(pq, archive, search_ptr, search_len, ctx, base);

        absOff += (int64_t)out.pos;
        first = false;

        if (OVER > 0)
        {
          if (out.pos >= OVER)
          {
            memcpy(buf, dst + out.pos - OVER, OVER);
          }
          else
          {
            memmove(buf, buf + out.pos, OVER - out.pos);
            memcpy(buf + (OVER - out.pos), dst, out.pos);
          }
        }
      }
      /* ret==0 => end of frame; loop continues if more compressed input remains */
    }
  }
}

static void search_indexed_file(PrintQueue *pq,
                                const char *zstFile,
                                const char *idxFile,
                                const PatCtx *ctx,
                                char *buf,
                                char *inbuf,
                                ZSTD_DCtx *dctx)
{
  IndexInfo *info = parse_index_json(idxFile);
  if (!info)
    return;

  int fd = open(zstFile, O_RDONLY);
  if (fd < 0)
  {
    fprintf(stderr, "open failed: %s\n", zstFile);
    free_index_info(info);
    return;
  }

  for (size_t i = 0; i < info->count; ++i)
  {
    const ChunkInfo *C = &info->chunks[i];
    stream_decompress_and_search(pq, fd, C, ctx, zstFile, buf, inbuf, dctx);
    
    // Reset the decompression context between chunks for proper stream processing
#if defined(ZSTD_VERSION_MAJOR)
    ZSTD_DCtx_reset(dctx, ZSTD_reset_session_only);
#endif
  }

  close(fd);
  free_index_info(info);
}

typedef struct
{
  const char **zstFiles;
  const char **idxFiles;
  int fileCount;
  int start, end;
  PrintQueue *pqueue;
  pthread_t tid;
  int thread_id;
} WorkerArg;

static void *worker_thread(void *arg)
{
  WorkerArg *W = (WorkerArg *)arg;
  
  // Allocate reusable buffers once per thread
  const size_t OVER = (g_pat.len > 1) ? (g_pat.len - 1) : 0;
  char *buf = (char *)malloc(ZSTD_OUT_WIN + OVER);
  char *inbuf = (char *)malloc(ZSTD_IN_CHUNK);
  ZSTD_DCtx *dctx = ZSTD_createDCtx();
  
  if (!buf || !inbuf || !dctx)
  {
    fprintf(stderr, "Thread %d: Failed to allocate buffers\n", W->thread_id);
    if (buf) free(buf);
    if (inbuf) free(inbuf);
    if (dctx) ZSTD_freeDCtx(dctx);
    return NULL;
  }
  
  for (int i = W->start; i < W->end && i < W->fileCount; ++i)
  {
    if (!W->zstFiles[i] || !W->idxFiles[i])
      continue;
    search_indexed_file(W->pqueue, W->zstFiles[i], W->idxFiles[i], &g_pat, buf, inbuf, dctx);
  }
  
  // Free buffers once at thread completion
  ZSTD_freeDCtx(dctx);
  free(inbuf);
  free(buf);
  
  return NULL;
}

// Helper function to build full path
static char* build_path(const char *dir, const char *file)
{
  size_t dir_len = strlen(dir);
  size_t file_len = strlen(file);
  bool need_slash = (dir_len > 0 && dir[dir_len - 1] != '/');
  
  size_t total_len = dir_len + (need_slash ? 1 : 0) + file_len + 1;
  char *path = (char *)malloc(total_len);
  if (!path)
    return NULL;
  
  strcpy(path, dir);
  if (need_slash)
    strcat(path, "/");
  strcat(path, file);
  
  return path;
}

int main(int argc, char **argv)
{
  const char *search_dir = ".";
  const char *pattern = NULL;
  
  // Parse command line arguments
  int opt;
  while ((opt = getopt(argc, argv, "d:h")) != -1)
  {
    switch (opt)
    {
    case 'd':
      search_dir = optarg;
      break;
    case 'h':
      fprintf(stderr, "Usage: %s [-d directory] <pattern>\n", argv[0]);
      fprintf(stderr, "  -d directory  Directory to search (default: current directory)\n");
      fprintf(stderr, "  -h           Show this help message\n");
      return 0;
    default:
      fprintf(stderr, "Usage: %s [-d directory] <pattern>\n", argv[0]);
      return 1;
    }
  }
  
  // Get the pattern from remaining arguments
  if (optind >= argc)
  {
    fprintf(stderr, "Error: No search pattern specified\n");
    fprintf(stderr, "Usage: %s [-d directory] <pattern>\n", argv[0]);
    return 1;
  }
  pattern = argv[optind];
  
  const size_t patLen = strlen(pattern);
  if (!patLen)
  {
    fprintf(stderr, "Empty pattern.\n");
    return 1;
  }

  // Check if directory exists
  struct stat dir_stat;
  if (stat(search_dir, &dir_stat) != 0)
  {
    fprintf(stderr, "Error: Cannot access directory '%s': %s\n", search_dir, strerror(errno));
    return 1;
  }
  if (!S_ISDIR(dir_stat.st_mode))
  {
    fprintf(stderr, "Error: '%s' is not a directory\n", search_dir);
    return 1;
  }

  g_pat.pat = pattern;
  g_pat.len = patLen;
  build_badchar_table(g_pat.pat, g_pat.len, g_pat.bad);

  DIR *d = opendir(search_dir);
  if (!d)
  {
    fprintf(stderr, "opendir %s failed: %s\n", search_dir, strerror(errno));
    return 1;
  }

  size_t cap = 64, count = 0;
  char **zst = (char **)malloc(cap * sizeof(*zst));
  char **idx = (char **)malloc(cap * sizeof(*idx));
  if (!zst || !idx)
  {
    fprintf(stderr, "OOM\n");
    if (zst)
      free(zst);
    if (idx)
      free(idx);
    closedir(d);
    return 1;
  }

  struct dirent *de;
  while ((de = readdir(d)) != NULL)
  {
    if (fnmatch("batch_[0-9][0-9][0-9][0-9][0-9].tar.zst", de->d_name, 0) == 0)
    {
      // Build full paths for the .zst and .idx.json files
      char *zstPath = build_path(search_dir, de->d_name);
      if (!zstPath)
      {
        fprintf(stderr, "Failed to build path for %s\n", de->d_name);
        continue;
      }
      
      // Build the index filename
      char idxName[MAX_FILENAME_LEN];
      strncpy(idxName, de->d_name, sizeof(idxName) - 1);
      idxName[sizeof(idxName) - 1] = '\0';
      char *p = strstr(idxName, ".tar.zst");
      if (!p)
      {
        free(zstPath);
        continue;
      }
      strcpy(p, ".tar.idx.json");
      
      char *idxPath = build_path(search_dir, idxName);
      if (!idxPath)
      {
        fprintf(stderr, "Failed to build path for %s\n", idxName);
        free(zstPath);
        continue;
      }
      
      // Check if index file exists
      struct stat st;
      if (stat(idxPath, &st) == 0)
      {
        if (count >= cap)
        {
          cap *= 2;
          char **nz = (char **)realloc(zst, cap * sizeof(*nz));
          char **ni = (char **)realloc(idx, cap * sizeof(*ni));
          if (!nz || !ni)
          {
            fprintf(stderr, "realloc failed\n");
            free(zstPath);
            free(idxPath);
            for (size_t k = 0; k < count; k++)
            {
              free(zst[k]);
              free(idx[k]);
            }
            free(zst);
            free(idx);
            closedir(d);
            return 1;
          }
          zst = nz;
          idx = ni;
        }
        zst[count] = zstPath;
        idx[count] = idxPath;
        ++count;
      }
      else
      {
        free(zstPath);
        free(idxPath);
      }
    }
  }
  closedir(d);

  if (count == 0)
  {
    fprintf(stderr, "No batch_*.tar.zst + .idx.json pairs found in directory: %s\n", search_dir);
    free(zst);
    free(idx);
    return 0;
  }

  PrintQueue pq;
  printqueue_init(&pq);

  pthread_t printerTid;
  if (pthread_create(&printerTid, NULL, printer_thread, &pq) != 0)
  {
    fprintf(stderr, "printer thread failed\n");
    for (size_t i = 0; i < count; i++)
    {
      free(zst[i]);
      free(idx[i]);
    }
    free(zst);
    free(idx);
    printqueue_destroy(&pq);
    return 1;
  }

  int threads = (int)((count < MAX_THREADS) ? count : MAX_THREADS);
  WorkerArg *args = (WorkerArg *)calloc((size_t)threads, sizeof(*args));
  if (!args)
  {
    fprintf(stderr, "OOM (worker args)\n");
    printqueue_mark_done(&pq);
    pthread_join(printerTid, NULL);
    for (size_t i = 0; i < count; i++)
    {
      free(zst[i]);
      free(idx[i]);
    }
    free(zst);
    free(idx);
    printqueue_destroy(&pq);
    return 1;
  }

  int per = (int)count / threads, rem = (int)count % threads, start = 0;
  for (int t = 0; t < threads; ++t)
  {
    int load = per + (t < rem ? 1 : 0);
    args[t].zstFiles = (const char **)zst;
    args[t].idxFiles = (const char **)idx;
    args[t].fileCount = (int)count;
    args[t].start = start;
    args[t].end = start + load;
    args[t].pqueue = &pq;
    args[t].thread_id = t;
    start += load;
    if (pthread_create(&args[t].tid, NULL, worker_thread, &args[t]) != 0)
    {
      fprintf(stderr, "worker %d spawn failed\n", t);
      args[t].tid = 0;
    }
  }

  for (int t = 0; t < threads; ++t)
    if (args[t].tid)
      pthread_join(args[t].tid, NULL);

  printqueue_mark_done(&pq);
  pthread_join(printerTid, NULL);

  for (size_t i = 0; i < count; i++)
  {
    free(zst[i]);
    free(idx[i]);
  }
  free(zst);
  free(idx);
  free(args);
  printqueue_destroy(&pq);
  return 0;
}
