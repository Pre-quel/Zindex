#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <zstd.h>

#include "cJSON.h" // The cJSON library header

// -----------------------------------------------------------------------------
//  Const
// -----------------------------------------------------------------------------
#define MAX_CHUNK_UNCOMPRESSED (64ULL * 1024ULL * 1024ULL) // e.g. 64MB max
#define ROLLING_BUF_SIZE (64 * 1024)     // keep dat RAM low           
#define MAX_FILENAME_LEN 512
#define MAX_PREVIEW_LEN 1024
#define MAX_QUEUE_SIZE 20000
#define MAX_THREADS 8 // or set to whatever makes sense

// -----------------------------------------------------------------------------
//  Data structures
// -----------------------------------------------------------------------------
typedef struct
{
    int chunk_id;
    long long compressed_offset;
    long long compressed_size;
    long long uncompressed_start;
    long long uncompressed_end;
} ChunkInfo;

typedef struct
{
    ChunkInfo *chunks;
    size_t count;
} IndexInfo;

typedef struct
{
    char filename[MAX_FILENAME_LEN];
    off_t offset;
    int closenessScore;
    char preview[2048];
} SearchResult;

// A simple thread-safe queue for results
typedef struct PrintNode
{
    SearchResult item;
    struct PrintNode *next;
} PrintNode;

typedef struct
{
    PrintNode *head;
    PrintNode *tail;
    size_t size;
    bool done;
    pthread_mutex_t lock;
    pthread_cond_t cond;
} PrintQueue;

// -----------------------------------------------------------------------------
//  Base64 decode helper
// -----------------------------------------------------------------------------
static int b64Value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}
uint8_t* base64_decode(const char* b64, size_t* outLen)
{
    size_t len = strlen(b64);
    int pad = 0;
    if (len >= 2 && b64[len-1]=='=') { pad++; }
    if (len >= 2 && b64[len-2]=='=') { pad++; }

    size_t out_size = (len * 3) / 4 - pad;
    uint8_t* out = malloc(out_size);
    if(!out) return NULL;

    size_t j=0;
    unsigned val=0;
    int valb=-8;
    for(size_t i=0; i<len; i++){
        int c = b64Value((unsigned char)b64[i]);
        if(c<0 && b64[i] != '=') continue; // skip whitespace or invalid
        if(b64[i] == '=') c=0;  // pad
        val=(val<<6)+c;
        valb += 6;
        if(valb>=0){
            out[j++]=(uint8_t)((val>>valb)&0xFF);
            valb-=8;
        }
    }
    if(outLen) *outLen=j;
    return out;
}

// -----------------------------------------------------------------------------
//  Print Queue stuff
// -----------------------------------------------------------------------------
static void printqueue_init(PrintQueue *q)
{
    q->head = q->tail = NULL;
    q->size = 0;
    q->done = false;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->cond, NULL);
}

static void printqueue_destroy(PrintQueue *q)
{
    PrintNode *cur = q->head;
    while (cur)
    {
        PrintNode *tmp = cur->next;
        free(cur);
        cur = tmp;
    }
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->cond);
}

static void printqueue_push(PrintQueue *q, const SearchResult *res)
{
    pthread_mutex_lock(&q->lock);
    if (q->size >= MAX_QUEUE_SIZE)
    {
        pthread_mutex_unlock(&q->lock);
        return;
    }
    pthread_mutex_unlock(&q->lock);

    PrintNode *node = malloc(sizeof(*node));
    if (!node)
        return;
    node->item = *res;
    node->next = NULL;

    pthread_mutex_lock(&q->lock);
    if (q->tail)
    {
        q->tail->next = node;
        q->tail = node;
    }
    else
    {
        q->head = q->tail = node;
    }
    q->size++;
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->lock);
}

static bool printqueue_pop(PrintQueue *q, SearchResult *out)
{
    pthread_mutex_lock(&q->lock);
    while (q->size == 0 && !q->done)
    {
        pthread_cond_wait(&q->cond, &q->lock);
    }
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
    *out = node->item;
    free(node);
    pthread_mutex_unlock(&q->lock);
    return true;
}

static void printqueue_mark_done(PrintQueue *q)
{
    pthread_mutex_lock(&q->lock);
    q->done = true;
    pthread_cond_broadcast(&q->cond);
    pthread_mutex_unlock(&q->lock);
}

// -----------------------------------------------------------------------------
//  JSON Parsing for .idx.json (unchanged except for bloom stuff removed)
// -----------------------------------------------------------------------------
static IndexInfo *parse_index_json(const char *idx_filename)
{
    FILE *fp = fopen(idx_filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Cannot open idx file: %s\n", idx_filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buf = malloc(sz + 1);
    if (!buf)
    {
        fclose(fp);
        return NULL;
    }
    fread(buf, 1, sz, fp);
    buf[sz] = '\0';
    fclose(fp);

    cJSON *root = cJSON_Parse(buf);
    if (!root)
    {
        fprintf(stderr, "JSON parse error in %s\n", idx_filename);
        free(buf);
        return NULL;
    }
    cJSON *chunksArr = cJSON_GetObjectItem(root, "chunks");
    if (!chunksArr || !cJSON_IsArray(chunksArr))
    {
        fprintf(stderr, "No 'chunks' array in %s\n", idx_filename);
        cJSON_Delete(root);
        free(buf);
        return NULL;
    }
    int n = cJSON_GetArraySize(chunksArr);
    IndexInfo *info = calloc(1, sizeof(*info));
    info->count = n;
    info->chunks = calloc(n, sizeof(ChunkInfo));
    for (int i = 0; i < n; i++)
    {
        cJSON *elem = cJSON_GetArrayItem(chunksArr, i);
        if (!elem)
            continue;
        ChunkInfo *C = &info->chunks[i];
        C->chunk_id = cJSON_GetObjectItem(elem, "chunk_id")->valueint;
        C->compressed_offset =
            (long long)cJSON_GetObjectItem(elem, "compressed_offset")->valuedouble;
        C->compressed_size =
            (long long)cJSON_GetObjectItem(elem, "compressed_size")->valuedouble;
        C->uncompressed_start =
            (long long)cJSON_GetObjectItem(elem, "uncompressed_start")->valuedouble;
        C->uncompressed_end =
            (long long)cJSON_GetObjectItem(elem, "uncompressed_end")->valuedouble;
    }
    cJSON_Delete(root);
    free(buf);
    return info;
}

static void free_index_info(IndexInfo *info)
{
    if (!info)
        return;
    free(info->chunks);
    free(info);
}

// -----------------------------------------------------------------------------
//  Boyer–Moore (or Horspool) -style search function (case-insensitive).
//  // *** BM CHANGE ***
// -----------------------------------------------------------------------------
static void build_badchar_table(const char *pattern, size_t patLen, int badCharTable[256])
{
    // Initialize all occurrences as -1
    for (int i = 0; i < 256; i++)
        badCharTable[i] = (int)patLen;

    // Fill the actual last occurrence of each character
    for (size_t i = 0; i < patLen - 1; i++)
    {
        unsigned char c = (unsigned char)tolower((unsigned char)pattern[i]);
        badCharTable[c] = (int)(patLen - 1 - i);
    }
}

// Returns all matches of `pattern` in `text`, ignoring case. For each match
// found, calls "reporting" routine that will build a preview and push to queue.
static void boyer_moore_search(
    const char *text, size_t textLen,
    const char *pattern, size_t patLen,
    off_t baseOffset,
    const char *filename,
    PrintQueue *pqueue)
{
    if (patLen == 0 || textLen < patLen) return;

    // Build skip table
    int badCharTable[256];
    build_badchar_table(pattern, patLen, badCharTable);

    // Store a lowercased pattern
    char *patLower = (char *)malloc(patLen + 1);
    for (size_t i = 0; i < patLen; i++)
        patLower[i] = (char)tolower((unsigned char)pattern[i]);
    patLower[patLen] = '\0';

    size_t i = 0;  // index into text
    while (i <= textLen - patLen)
    {
        // Compare from the end of the pattern
        size_t j = patLen - 1;
        while (j < patLen) // j is unsigned, so j < patLen also checks j != -1
        {
            unsigned char t = (unsigned char)tolower((unsigned char)text[i + j]);
            if (t != (unsigned char)patLower[j])
                break;
            if (j == 0)
            {
                // Found a match at i
                {
                    off_t matchOffset = baseOffset + i;
                    // We'll extract a line preview
                    size_t lineStart = i;
                    while (lineStart > 0 && text[lineStart - 1] != '\n')
                    {
                        lineStart--;
                    }
                    size_t lineEnd = i + patLen - 1;
                    while (lineEnd < textLen && text[lineEnd] != '\n')
                    {
                        lineEnd++;
                    }
                    size_t lineLen = lineEnd - lineStart;
                    if (lineLen > 1023) lineLen = 1023;

                    char preview[1024];
                    memset(preview, 0, sizeof(preview));
                    memcpy(preview, text + lineStart, lineLen);
                    preview[lineLen] = '\0';

                    // Simple closeness measure
                    // (Orig. implemenetation w/ Aho-Corasick just aligned chars from pattern vs text)
                    // (...I guess I'll just stick to that lol)
                    int closeness = 0, curr = 0;
                    const char *haystackPtr = text + i;
                    const char *needlePtr   = pattern; // keep the original pattern for closeness
                    while (*haystackPtr && *needlePtr)
                    {
                        if (tolower(*haystackPtr) == tolower(*needlePtr))
                        {
                            curr++;
                            if (curr > closeness)
                                closeness = curr;
                        }
                        else
                        {
                            curr = 0;
                        }
                        haystackPtr++;
                        needlePtr++;
                    }

                    // Prepare the SearchResult
                    SearchResult sr;
                    strncpy(sr.filename, filename, sizeof(sr.filename) - 1);
                    sr.filename[sizeof(sr.filename) - 1] = '\0';
                    sr.offset = matchOffset;
                    sr.closenessScore = closeness;
                    strncpy(sr.preview, preview, sizeof(sr.preview) - 1);
                    sr.preview[sizeof(sr.preview) - 1] = '\0';
                    printqueue_push(pqueue, &sr);
                }

                // Move forward by 1 (or the entire pattern length).
                // For standard Boyer–Moore, TODO: shift less dumbly lol
                i++;
                goto skip_shift; 
            }
            j--;
        }

        {
            // Mismatch at text[i + j]
            unsigned char mismatchC = (unsigned char)tolower((unsigned char)text[i + (patLen - 1)]);
            int shift = badCharTable[mismatchC];
            if (shift < 1) shift = 1;
            i += shift;
        }

    skip_shift:;
    }

    free(patLower);
}

// -----------------------------------------------------------------------------
//  Keep a "ring buffer", but do BM search each time
//  we fill or flush the buffer.  // *** BM CHANGE after removing Aho-Corasick ***
// -----------------------------------------------------------------------------
typedef struct
{
    char buffer[ROLLING_BUF_SIZE];
    size_t writePos;
    off_t globalOffset;
    pthread_mutex_t lock;
} RingBufState;

static __thread RingBufState g_ring;

// Ring buffer -- why even? -- to check across chunk boundaries
static void ringbuf_init(void)
{
    pthread_mutex_init(&g_ring.lock, NULL);
    pthread_mutex_lock(&g_ring.lock);
    memset(g_ring.buffer, 0, sizeof(g_ring.buffer));
    g_ring.writePos = 0;
    g_ring.globalOffset = 0;
    pthread_mutex_unlock(&g_ring.lock);
}

// Feed function: once the buffer is full, we do a BM search inside it
static void ringbuf_feed(
    PrintQueue *pqueue,
    const char *filename,
    const char *pattern,
    size_t patLen,
    const char *src, size_t srcLen)
{
    pthread_mutex_lock(&g_ring.lock);
    size_t srcPos = 0;
    // Overlap to handle partial matches across boundaries if desired
    size_t overlap = (patLen > 0) ? (patLen - 1) : 0;
    if (overlap >= ROLLING_BUF_SIZE)
        overlap = 0;

    while (srcLen > 0)
    {
        if (g_ring.writePos >= ROLLING_BUF_SIZE)
        {
            // Search entire buffer
            boyer_moore_search(
                g_ring.buffer, g_ring.writePos,
                pattern, patLen,
                g_ring.globalOffset - g_ring.writePos, 
                filename, pqueue);

            // Keep overlap
            if (overlap > g_ring.writePos)
                overlap = g_ring.writePos;
            memmove(g_ring.buffer, g_ring.buffer + (g_ring.writePos - overlap), overlap);
            g_ring.writePos = overlap;
        }

        size_t space = ROLLING_BUF_SIZE - g_ring.writePos;
        if (space > srcLen)
            space = srcLen;

        memcpy(g_ring.buffer + g_ring.writePos, src + srcPos, space);
        g_ring.writePos += space;
        g_ring.globalOffset += space;
        srcLen -= space;
        srcPos += space;
    }
    pthread_mutex_unlock(&g_ring.lock);
}

// Once we are done with a chunk/file, flush any leftover in the ring buffer
static void ringbuf_flush(
    PrintQueue *pqueue,
    const char *filename,
    const char *pattern,
    size_t patLen)
{
    pthread_mutex_lock(&g_ring.lock);
    if (g_ring.writePos > 0)
    {
        boyer_moore_search(
            g_ring.buffer, g_ring.writePos,
            pattern, patLen,
            g_ring.globalOffset - g_ring.writePos,
            filename, pqueue);
    }
    g_ring.writePos = 0;
    pthread_mutex_unlock(&g_ring.lock);
}

// -----------------------------------------------------------------------------
//  Decompress a single chunk and perform searching via ringbuf (BM).
// -----------------------------------------------------------------------------
static void decompress_and_search_chunk(
    const char *zstFile,
    const ChunkInfo *c,
    PrintQueue *pqueue,
    const char *pattern,
    size_t patLen)
{
    FILE *fp = fopen(zstFile, "rb");
    if (!fp)
    {
        fprintf(stderr, "Cannot open zst file: %s\n", zstFile);
        return;
    }
    fseeko(fp, c->compressed_offset, SEEK_SET);

    void *cbuf = malloc((size_t)c->compressed_size);
    if (!cbuf)
    {
        fclose(fp);
        return;
    }
    size_t readBytes = fread(cbuf, 1, (size_t)c->compressed_size, fp);
    fclose(fp);
    if (readBytes != (size_t)c->compressed_size)
    {
        fprintf(stderr, "Read mismatch on chunk %d\n", c->chunk_id);
        free(cbuf);
        return;
    }

    ZSTD_DCtx *dctx = ZSTD_createDCtx();
    if (!dctx)
    {
        free(cbuf);
        return;
    }

    long long length = c->uncompressed_end - c->uncompressed_start + 1;
    if (length < 0)
        length = 0;
    if (length > (long long)MAX_CHUNK_UNCOMPRESSED)
        length = MAX_CHUNK_UNCOMPRESSED;

    void *dbuf = malloc((size_t)length);
    if (!dbuf)
    {
        free(cbuf);
        ZSTD_freeDCtx(dctx);
        return;
    }

    size_t dSize = ZSTD_decompressDCtx(
        dctx, dbuf, (size_t)length, cbuf, (size_t)c->compressed_size);
    if (!ZSTD_isError(dSize))
    {
        // feed ring buffer
        ringbuf_init();
        ringbuf_feed(pqueue, zstFile, pattern, patLen, (const char *)dbuf, dSize);
        ringbuf_flush(pqueue, zstFile, pattern, patLen);
    }
    else
    {
        fprintf(stderr, "Decompress error chunk %d: %s\n",
                c->chunk_id, ZSTD_getErrorName(dSize));
    }

    free(dbuf);
    free(cbuf);
    ZSTD_freeDCtx(dctx);
}

// -----------------------------------------------------------------------------
//  Search a single .tar.zst with .idx.json using Boyer–Moore
// -----------------------------------------------------------------------------
static void search_indexed_file(
    const char *zstFile,
    const char *idxFile,
    PrintQueue *pqueue,
    const char *pattern,
    size_t patLen)
{
    IndexInfo *info = parse_index_json(idxFile);
    if (!info)
    {
        fprintf(stderr, "Failed to parse index file: %s\n", idxFile);
        return;
    }

    // For each chunk
    for (size_t i = 0; i < info->count; i++)
    {
        decompress_and_search_chunk(zstFile, &info->chunks[i], pqueue, pattern, patLen);
    }

    free_index_info(info);
}

// -----------------------------------------------------------------------------
//  Threading
// -----------------------------------------------------------------------------
typedef struct
{
    PrintQueue *pqueue;
    const char **zstFiles;
    const char **idxFiles;
    int start;
    int end;
    const char *pattern;
    size_t patLen;
} WorkerArg;

static void *worker_thread(void *arg)
{
    WorkerArg *W = (WorkerArg *)arg;
    for (int i = W->start; i < W->end; i++)
    {
        search_indexed_file(
            W->zstFiles[i], W->idxFiles[i],
            W->pqueue,
            W->pattern,
            W->patLen);
    }
    return NULL;
}

static void *printer_thread(void *arg)
{
    PrintQueue *q = (PrintQueue *)arg;
    SearchResult sr;
    while (printqueue_pop(q, &sr))
    {

        printf("{\"file\":\"%s\", \"offset\":%lld, \"score\":%d, \"preview\":\"%s\"}\n",
               sr.filename, (long long)sr.offset,
               sr.closenessScore,
               sr.preview);
        fflush(stdout);
        
    }
    return NULL;
}

// -----------------------------------------------------------------------------
//  main()
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <pattern>\n", argv[0]);
        return 1;
    }
    const char *pattern = argv[1];
    size_t patLen = strlen(pattern);

    // Find .tar.zst + .tar.idx.json pairs in current directory
    DIR *d = opendir(".");
    if (!d)
    {
        fprintf(stderr, "Cannot open current dir.\n");
        return 1;
    }

    char **zstFiles = NULL;
    char **idxFiles = NULL;
    size_t capacity = 0, count = 0;

    struct dirent *de;
    while ((de = readdir(d)) != NULL)
    {
        if (fnmatch("batch_[0-9][0-9][0-9][0-9][0-9].tar.zst", de->d_name, 0) == 0)
        {
            char idxName[512];
            strncpy(idxName, de->d_name, sizeof(idxName));
            idxName[sizeof(idxName) - 1] = '\0';
            char *p = strstr(idxName, ".tar.zst");
            if (!p)
                continue;
            strcpy(p, ".tar.idx.json");

            struct stat stbuf;
            if (stat(idxName, &stbuf) == 0)
            {
                if (count >= capacity)
                {
                    size_t newcap = capacity ? capacity * 2 : 32;
                    zstFiles = realloc(zstFiles, newcap * sizeof(char *));
                    idxFiles = realloc(idxFiles, newcap * sizeof(char *));
                    capacity = newcap;
                }
                zstFiles[count] = strdup(de->d_name);
                idxFiles[count] = strdup(idxName);
                count++;
            }
        }
    }
    closedir(d);

    if (count == 0)
    {
        fprintf(stderr, "No chunked .tar.zst + .idx.json pairs found.\n");
        free(zstFiles);
        free(idxFiles);
        return 1;
    }

    // For printint what's been found, yay posix threads
    PrintQueue queue;
    printqueue_init(&queue);

    pthread_t printerTid;
    pthread_create(&printerTid, NULL, printer_thread, &queue);
    int threadCount = (count > MAX_THREADS ? MAX_THREADS : (int)count);
    pthread_t tids[threadCount];
    WorkerArg wargs[threadCount];

    int filesPerThread = count / threadCount;
    int remainder = count % threadCount;
    int start = 0;
    for (int i = 0; i < threadCount; i++)
    {
        int load = filesPerThread + (i < remainder ? 1 : 0);
        wargs[i].pqueue = &queue;
        wargs[i].zstFiles = (const char **)zstFiles;
        wargs[i].idxFiles = (const char **)idxFiles;
        wargs[i].start = start;
        wargs[i].end = start + load;
        wargs[i].pattern = pattern;
        wargs[i].patLen = patLen;
        pthread_create(&tids[i], NULL, worker_thread, &wargs[i]);
        start += load;
    }

    // Wait for workers to finish
    for (int i = 0; i < threadCount; i++)
    {
        pthread_join(tids[i], NULL);
    }

    // Signal printer we're done
    printqueue_mark_done(&queue);
    pthread_join(printerTid, NULL);

    // Free free free free free
    for (size_t i = 0; i < count; i++)
    {
        free(zstFiles[i]);
        free(idxFiles[i]);
    }
    free(zstFiles);
    free(idxFiles);

    printqueue_destroy(&queue);

    return 0;
}
