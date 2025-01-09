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

// For limiting chunk memory usage, etc.
#define MAX_CHUNK_UNCOMPRESSED (64ULL * 1024ULL * 1024ULL) // e.g. 64MB max
#define ROLLING_BUF_SIZE (64 * 1024)                       // Use a bigger ring buffer if you want
#define MAX_FILENAME_LEN 512
#define MAX_PREVIEW_LEN 1024
#define MAX_QUEUE_SIZE 20000

#define MAX_THREADS 8 // or 12, however many you want

// -----------------------------------------------------------------------------
//  Data structures from your existing code
// -----------------------------------------------------------------------------

// chunk-based index structures
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

// For your ring buffer approach
typedef struct
{
    char filename[MAX_FILENAME_LEN];
    off_t offset;
    int closenessScore;
    char preview[2048];
} SearchResult;

// PrintQueue for multi-threaded result printing
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
//  Aho-Corasick structures
// -----------------------------------------------------------------------------
#define AC_ALPHABET_SIZE 256

typedef struct ACNode
{
    struct ACNode *children[AC_ALPHABET_SIZE];
    struct ACNode *fail;
    int output;
    int depth;
} ACNode;

typedef struct
{
    ACNode **nodes;
    size_t count;
    size_t capacity;
    pthread_mutex_t lock;
} NodeTracker;

static NodeTracker g_tracker;
static pthread_mutex_t g_pattern_mutex = PTHREAD_MUTEX_INITIALIZER;
static size_t g_patternLen = 0;

static void set_pattern_len(size_t len)
{
    pthread_mutex_lock(&g_pattern_mutex);
    g_patternLen = len;
    pthread_mutex_unlock(&g_pattern_mutex);
}
static size_t get_pattern_len(void)
{
    pthread_mutex_lock(&g_pattern_mutex);
    size_t len = g_patternLen;
    pthread_mutex_unlock(&g_pattern_mutex);
    return len;
}

// ring buffer state
typedef struct
{
    char buffer[ROLLING_BUF_SIZE];
    size_t writePos;
    off_t globalOffset;
    ACNode *acState;
    pthread_mutex_t lock;
} RingBufState;

static __thread RingBufState g_ring;

// -----------------------------------------------------------------------------
//  JSON Parsing for .idx.json
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
//  Aho–Corasick creation
// -----------------------------------------------------------------------------
static void init_node_tracker(void)
{
    g_tracker.nodes = malloc(1024 * sizeof(ACNode *));
    g_tracker.capacity = 1024;
    g_tracker.count = 0;
    pthread_mutex_init(&g_tracker.lock, NULL);
}
static ACNode *create_node(void)
{
    ACNode *node = calloc(1, sizeof(*node));
    if (!node)
        return NULL;

    pthread_mutex_lock(&g_tracker.lock);
    if (g_tracker.count >= g_tracker.capacity)
    {
        size_t newcap = g_tracker.capacity * 2;
        ACNode **newarr = realloc(g_tracker.nodes, newcap * sizeof(ACNode *));
        if (!newarr)
        {
            pthread_mutex_unlock(&g_tracker.lock);
            free(node);
            return NULL;
        }
        g_tracker.nodes = newarr;
        g_tracker.capacity = newcap;
    }
    g_tracker.nodes[g_tracker.count++] = node;
    pthread_mutex_unlock(&g_tracker.lock);
    return node;
}
static void free_aho_automaton(ACNode *root)
{
    pthread_mutex_lock(&g_tracker.lock);
    for (size_t i = 0; i < g_tracker.count; i++)
    {
        free(g_tracker.nodes[i]);
    }
    free(g_tracker.nodes);
    g_tracker.nodes = NULL;
    g_tracker.count = g_tracker.capacity = 0;
    pthread_mutex_unlock(&g_tracker.lock);
    pthread_mutex_destroy(&g_tracker.lock);
}
ACNode *build_aho_automaton(const char *pattern)
{
    set_pattern_len(strlen(pattern));
    ACNode *root = create_node();
    if (!root)
        return NULL;

    // Insert the pattern characters exactly once
    ACNode *cur = root;
    for (size_t i = 0; i < get_pattern_len(); i++)
    {
        unsigned char c = (unsigned char)tolower((unsigned char)pattern[i]);
        if (!cur->children[c])
        {
            cur->children[c] = create_node();
            if (!cur->children[c])
                return root;
            cur->children[c]->depth = cur->depth + 1;
        }
        cur = cur->children[c];
    }
    cur->output = 1;
    // BFS for fail links
    // Typically we do the classic approach:
    ACNode **queue = malloc(MAX_QUEUE_SIZE * sizeof(ACNode *)); // or bigger
    if (!queue)
        return root;

    int front = 0, back = 0;

    // For c in [0..255], if child is null => set child to root
    // otherwise set fail= root & enqueue
    for (int c = 0; c < AC_ALPHABET_SIZE; c++)
    {
        if (root->children[c])
        {
            root->children[c]->fail = root;
            queue[back++] = root->children[c];
        }
        else
        {
            // direct transition from root
            root->children[c] = root;
        }
    }

    // BFS
    while (front < back)
    {
        ACNode *u = queue[front++];
        for (int c = 0; c < AC_ALPHABET_SIZE; c++)
        {
            ACNode *v = u->children[c];
            if (!v)
            {
                // set missing transition to fail->children[c]
                u->children[c] = u->fail->children[c];
            }
            else
            {
                // set fail
                ACNode *f = u->fail;
                v->fail = f->children[c];
                v->output |= v->fail->output;
                queue[back++] = v;
            }
        }
    }
    free(queue);
    return root;
}

// simple closeness measure
static int compute_closeness(const char *haystack, const char *needle)
{
    // e.g. count consecutive matching letters from the start
    int best = 0, curr = 0;
    while (*haystack && *needle)
    {
        if (tolower(*haystack) == tolower(*needle))
        {
            curr++;
            if (curr > best)
                best = curr;
        }
        else
        {
            curr = 0;
        }
        haystack++;
        needle++;
    }
    return best;
}

// -----------------------------------------------------------------------------
//  Ring Buffer Logic
// -----------------------------------------------------------------------------
static void ringbuf_init(ACNode *root)
{
    pthread_mutex_init(&g_ring.lock, NULL);
    pthread_mutex_lock(&g_ring.lock);
    memset(g_ring.buffer, 0, sizeof(g_ring.buffer));
    g_ring.writePos = 0;
    g_ring.globalOffset = 0;
    g_ring.acState = root;
    pthread_mutex_unlock(&g_ring.lock);
}

static void ac_feed_chunk(
    ACNode **curStatePtr,
    const char *data,
    size_t dataLen,
    ACNode *root,
    off_t globalOffset,
    const char *filename,
    PrintQueue *pqueue,
    const char *searchStr)
{
    if (!curStatePtr || !*curStatePtr || !data || !pqueue || !searchStr)
    {
        return;
    }

    ACNode *st = *curStatePtr;
    size_t patLen = get_pattern_len();

    for (size_t i = 0; i < dataLen; i++)
    {
        unsigned char c = (unsigned char)tolower((unsigned char)data[i]);
        st = st->children[c];
        if (st->output)
        {
            // We found a 'match' ending at i
            size_t matchStart = (i >= patLen - 1) ? (i - (patLen - 1)) : 0;
            off_t matchOffset = globalOffset + matchStart;

            // Let's extract the entire line containing 'i'
            size_t lineStart = i;
            while (lineStart > 0 && data[lineStart - 1] != '\n')
            {
                lineStart--;
            }

            size_t lineEnd = i;
            while (lineEnd < dataLen && data[lineEnd] != '\n')
            {
                lineEnd++;
            }

            // Now [lineStart..lineEnd) is the entire line
            size_t lineLen = lineEnd - lineStart;
            if (lineLen > 1023)
            {
                // prevent overrun, or pick your own limit
                lineLen = 1023;
            }

            char preview[1024];
            memset(preview, 0, sizeof(preview));
            memcpy(preview, data + lineStart, lineLen);
            preview[lineLen] = '\0';

            // Optionally compute closeness again
            int closeness = compute_closeness(data + matchStart, searchStr);

            // Build the SearchResult
            SearchResult sr;
            strncpy(sr.filename, filename, sizeof(sr.filename) - 1);
            sr.filename[sizeof(sr.filename) - 1] = '\0';

            sr.offset = matchOffset;
            sr.closenessScore = closeness;

            strncpy(sr.preview, preview, sizeof(sr.preview) - 1);
            sr.preview[sizeof(sr.preview) - 1] = '\0';

            // Push to queue
            printqueue_push(pqueue, &sr);
        }
    }

    *curStatePtr = st;
}

static void ringbuf_feed(
    PrintQueue *pqueue,
    const char *filename,
    ACNode *root,
    const char *searchStr,
    const char *src, size_t srcLen)
{
    size_t srcPos = 0;
    size_t overlap = (get_pattern_len() > 0) ? (get_pattern_len() - 1) : 0;
    if (overlap >= ROLLING_BUF_SIZE)
        overlap = 0;

    while (srcLen > 0)
    {
        if (g_ring.writePos >= ROLLING_BUF_SIZE)
        {
            // feed entire buffer to AC
            ac_feed_chunk(&g_ring.acState, g_ring.buffer, g_ring.writePos,
                          root, (g_ring.globalOffset - g_ring.writePos),
                          filename, pqueue, searchStr);

            // keep overlap
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
}
static void ringbuf_flush(
    PrintQueue *pqueue,
    const char *filename,
    ACNode *root,
    const char *searchStr)
{
    if (g_ring.writePos > 0)
    {
        ac_feed_chunk(&g_ring.acState, g_ring.buffer, g_ring.writePos,
                      root, (g_ring.globalOffset - g_ring.writePos),
                      filename, pqueue, searchStr);
    }
    g_ring.writePos = 0;
}

// -----------------------------------------------------------------------------
//  Decompress a single chunk from .tar.zst using offsets from .idx.json
// -----------------------------------------------------------------------------
static void decompress_and_search_chunk(
    const char *zstFile,
    const ChunkInfo *c,
    PrintQueue *pqueue,
    ACNode *root,
    const char *searchStr)
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

    // approximate uncompressed
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

    size_t dSize = ZSTD_decompressDCtx(dctx, dbuf, (size_t)length, cbuf, (size_t)c->compressed_size);
    if (!ZSTD_isError(dSize))
    {
        // feed ring buffer
        ringbuf_init(root);
        ringbuf_feed(pqueue, zstFile, root, searchStr, (const char *)dbuf, dSize);
        ringbuf_flush(pqueue, zstFile, root, searchStr);
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
//  Search a single .tar.zst with .idx.json
// -----------------------------------------------------------------------------
static void search_indexed_file(
    const char *zstFile,
    const char *idxFile,
    PrintQueue *pqueue,
    ACNode *root,
    const char *searchStr)
{
    IndexInfo *info = parse_index_json(idxFile);
    if (!info)
    {
        fprintf(stderr, "Failed to parse index file: %s\n", idxFile);
        return;
    }

    // for each chunk
    for (size_t i = 0; i < info->count; i++)
    {
        decompress_and_search_chunk(zstFile, &info->chunks[i],
                                    pqueue, root, searchStr);
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
    ACNode *root;
    const char *searchStr;
} WorkerArg;

static void *worker_thread(void *arg)
{
    WorkerArg *W = (WorkerArg *)arg;
    for (int i = W->start; i < W->end; i++)
    {
        // search each pair
        search_indexed_file(W->zstFiles[i], W->idxFiles[i],
                            W->pqueue, W->root, W->searchStr);
    }
    return NULL;
}

// The printer thread
static void *printer_thread(void *arg)
{
    PrintQueue *q = (PrintQueue *)arg;
    SearchResult sr;
    while (printqueue_pop(q, &sr))
    {
        // e.g. you could do a closeness threshold
        size_t patLen = get_pattern_len();
        if (sr.closenessScore >= (int)(patLen * 0.6))
        {
            printf("{\"file\":\"%s\", \"offset\":%lld, \"score\":%d, \"preview\":\"%s\"}\n",
                   sr.filename, (long long)sr.offset,
                   sr.closenessScore,
                   sr.preview);
            fflush(stdout);
        }
    }
    return NULL;
}

// -----------------------------------------------------------------------------
//  Main
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <pattern>\n", argv[0]);
        return 1;
    }
    const char *pattern = argv[1];

    // A simple approach: we gather pairs: batch_XXXXX.tar.zst & batch_XXXXX.tar.idx.json
    // or you can pass them as arguments.

    // For example, let's scan the current dir for "batch_XXXXX.tar.zst" and see if "batch_XXXXX.tar.idx.json" exists.
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
            // build idx filename
            char idxName[512];
            strncpy(idxName, de->d_name, sizeof(idxName));
            idxName[sizeof(idxName) - 1] = '\0';
            char *p = strstr(idxName, ".tar.zst");
            if (!p)
                continue;
            strcpy(p, ".tar.idx.json"); // e.g. replace .tar.zst with .tar.idx.json

            struct stat stbuf;
            if (stat(idxName, &stbuf) == 0)
            {
                // we have a pair
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

    // Build Aho automaton
    init_node_tracker();
    ACNode *root = build_aho_automaton(pattern);
    if (!root)
    {
        fprintf(stderr, "Cannot build AC automaton\n");
        return 1;
    }

    // Prepare printing
    PrintQueue queue;
    printqueue_init(&queue);
    pthread_t printerTid;
    pthread_create(&printerTid, NULL, printer_thread, &queue);

    // Multi-thread
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
        wargs[i].root = root;
        wargs[i].searchStr = pattern;
        pthread_create(&tids[i], NULL, worker_thread, &wargs[i]);
        start += load;
    }

    for (int i = 0; i < threadCount; i++)
    {
        pthread_join(tids[i], NULL);
    }

    // Done
    printqueue_mark_done(&queue);
    pthread_join(printerTid, NULL);

    // Cleanup
    for (size_t i = 0; i < count; i++)
    {
        free(zstFiles[i]);
        free(idxFiles[i]);
    }
    free(zstFiles);
    free(idxFiles);

    printqueue_destroy(&queue);
    free_aho_automaton(root);
    return 0;
}
