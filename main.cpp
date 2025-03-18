/******************************************************************************
 # Zindex_o2

- Regex-free lazy pattern matching (NEON-accelerated BMH)
    (still faster than ugrep when stacked with normal grep as unix pipe)

- Bad-character shift table for maximizing skips.

- ARM NEON SIMD intrinsics (vld1q_u8, vceqq_u8) to compare 16-byte blocks in parallel.

- Assumes patterns are ≤64 bytes. Memcmp for larger.

- The only reason this file is .cpp and not .c is because of simdjson >:(
    ... and I'm not that mad, becuase CJSON, ugh ...
    ... ok std::vector is convenient too sometimes ...

- Lockless ring buffer for worker-printer thread ops. (Hated every moment of that.)

- Parallel subchunk processing because. Otpimization out of spite, at this point.

- Data prefetched (__builtin_prefetch) for min. cache misses

- Way too burnt out to think abouts stacking regex on top of this abomination.
 ******************************************************************************/

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdbool>
#include <string>
#include <vector>
#include <atomic>
#include <dirent.h>
#include <fnmatch.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <sched.h>
#include <sys/mman.h>

#include <zstd.h>
#include "simdjson.h"

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

/******************************************************************************
 * Configuration
 ******************************************************************************/
static const int MAX_THREADS = 16;
static const size_t MAX_FILENAME_LEN = 512;
static const uint64_t MAX_CHUNK_UNCOMPRESSED = (64ULL * 1024ULL * 1024ULL);

static const uint64_t MIN_SUBCHUNK_PARALLEL = (2ULL * 1024ULL * 1024ULL);
static const int MAX_SUBCHUNK_THREADS = 8;
static const int ASCII_SET_SIZE = 256;
static const size_t MAX_LINE_PREVIEW = 1024;
static const size_t RING_SIZE = 32768; 
static const size_t ALIGNMENT = 4096;

/******************************************************************************
 * Data Structures
 ******************************************************************************/
struct ChunkInfo
{
    int64_t chunk_id;
    int64_t compressed_offset;
    int64_t compressed_size;
    int64_t uncompressed_start;
    int64_t uncompressed_end;
};

struct IndexInfo
{
    ChunkInfo *chunks;
    size_t count;
};

struct SearchResult
{
    char filename[MAX_FILENAME_LEN];
    int64_t offset;
    int closenessScore;
    char preview[MAX_LINE_PREVIEW];
};

/******************************************************************************
 * Lockless Ring Buffer
 ******************************************************************************/
static std::atomic<size_t> ringHead{0};
static std::atomic<size_t> ringTail{0};
static SearchResult ringBuf[RING_SIZE];
static bool ringDone = false;

static inline bool ring_push(const SearchResult &res)
{
    size_t head = ringHead.load(std::memory_order_relaxed);
    size_t next = (head + 1) & (RING_SIZE - 1);
    size_t tail = ringTail.load(std::memory_order_acquire);
    if (next == tail)
    {
        
        return false;
    }
    ringBuf[head] = res;
    ringHead.store(next, std::memory_order_release);
    return true;
}

static inline bool ring_pop(SearchResult &out)
{
    size_t tail = ringTail.load(std::memory_order_relaxed);
    size_t head = ringHead.load(std::memory_order_acquire);
    if (tail == head)
    {
        
        return false;
    }
    out = ringBuf[tail];
    ringTail.store((tail + 1) & (RING_SIZE - 1), std::memory_order_release);
    return true;
}



inline bool extract_int64_field(const simdjson::dom::object &obj, const char *key, int64_t &dest)
{
    auto result = obj[key].get_int64();
    if (result.error() == simdjson::error_code::SUCCESS)
    {
        dest = result.value_unsafe();
        return true;
    }
    return false;
}

/******************************************************************************
 * JSON Index Parsing using simdjson
 ******************************************************************************/
static IndexInfo *parse_index_json(const char *idx_filename)
{
    
    simdjson::dom::parser parser;
    simdjson::padded_string doc;
    try
    {
        doc = simdjson::padded_string::load(idx_filename);
    }
    catch (...)
    {
        return nullptr;
    }

    
    simdjson::dom::element root;
    auto error = parser.parse(doc).get(root);
    if (error)
    {
        return nullptr;
    }

    
    simdjson::dom::array chunksArr;
    auto arrErr = root["chunks"].get_array().get(chunksArr);
    if (arrErr)
    {
        return nullptr;
    }

    
    IndexInfo *info = (IndexInfo *)calloc(1, sizeof(*info));
    info->count = chunksArr.size();
    info->chunks = (ChunkInfo *)calloc(info->count, sizeof(ChunkInfo));

    size_t i = 0;
    for (auto elem : chunksArr)
    {
        ChunkInfo *C = &info->chunks[i];

        
        simdjson::dom::object obj;
        if (elem.get_object().get(obj))
        {
            
            continue;
        }

        
        {
            
            extract_int64_field(obj, "chunk_id", C->chunk_id);
            extract_int64_field(obj, "compressed_offset", C->compressed_offset);
            extract_int64_field(obj, "compressed_size", C->compressed_size);
            extract_int64_field(obj, "uncompressed_start", C->uncompressed_start);
            extract_int64_field(obj, "uncompressed_end", C->uncompressed_end);
        }

        i++;
    }

    return info;
}

static void free_index_info(IndexInfo *info)
{
    if (!info)
        return;
    free(info->chunks);
    free(info);
}

/******************************************************************************
 * JSON-like Escaping
 ******************************************************************************/
static inline void json_escape(const char *src, char *dst, size_t dstSize)
{
    size_t j = 0;
    for (size_t i = 0; src[i] != '\0'; i++)
    {
        if (j >= dstSize - 2)
            break;
        char c = src[i];
        switch (c)
        {
        case '\\':
            dst[j++] = '\\';
            dst[j++] = '\\';
            break;
        case '\"':
            dst[j++] = '\\';
            dst[j++] = '\"';
            break;
        case '\n':
            dst[j++] = '\\';
            dst[j++] = 'n';
            break;
        case '\r':
            dst[j++] = '\\';
            dst[j++] = 'r';
            break;
        case '\t':
            dst[j++] = '\\';
            dst[j++] = 't';
            break;
        default:
            if ((unsigned char)c < 0x20)
            {
                dst[j++] = ' ';
            }
            else
            {
                dst[j++] = c;
            }
            break;
        }
    }
    dst[j] = '\0';
}

/******************************************************************************
 * Single-Line Snippet Extraction
 ******************************************************************************/
static inline void find_line(const char *text, size_t textLen, size_t pos,
                             size_t &start, size_t &end)
{
    size_t s = pos;
    while (s > 0 && text[s - 1] != '\n')
    {
        s--;
    }
    size_t e = pos;
    while (e < textLen && text[e] != '\n')
    {
        e++;
    }
    start = s;
    end = e;
}

static inline void single_line_snippet(const char *text, size_t textLen,
                                       size_t matchPos, char *outBuf, size_t outSize)
{
    size_t s, e;
    find_line(text, textLen, matchPos, s, e);
    size_t len = e - s;
    if (len >= outSize)
    {
        len = outSize - 1;
    }
    memcpy(outBuf, &text[s], len);
    outBuf[len] = '\0';
}

/******************************************************************************
 * Boyer-Moore-Horspool with NEON Acceleration
 ******************************************************************************/
#define BMH_BUILD_BADCHAR_TABLE(pat, patLen, badChar)      \
    do                                                     \
    {                                                      \
        for (int _i_ = 0; _i_ < ASCII_SET_SIZE; _i_++)     \
        {                                                  \
            badChar[_i_] = static_cast<int>(patLen);       \
        }                                                  \
        for (size_t _i_ = 0; _i_ < (patLen) - 1; _i_++)    \
        {                                                  \
            unsigned char _c_ = (unsigned char)(pat)[_i_]; \
            badChar[_c_] = (int)((patLen) - 1 - _i_);      \
        }                                                  \
    } while (0)

static inline bool cmp_16_asm_safe(const char *text, const char *pattern,
                                   size_t length,
                                   const char *bufBase, size_t bufLen)
{
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    

    
    
    
    size_t maxSafeBytesText = 0;
    if (text >= bufBase)
    {
        maxSafeBytesText = (bufBase + bufLen) - text;
        
        
        if ((intptr_t)maxSafeBytesText < 0)
        {
            maxSafeBytesText = 0;
        }
    }

    
    
    size_t maxSafeBytesPat = SIZE_MAX; 
    
    
    

    
    
    uint8_t localText[16];
    const uint8_t *tPtr = nullptr;
    {
        if (maxSafeBytesText < 16)
        {
            
            
            memset(localText, 0, sizeof(localText));
            size_t n = (maxSafeBytesText < length) ? maxSafeBytesText : length;
            memcpy(localText, text, n);
            tPtr = localText;
        }
        else
        {
            
            tPtr = reinterpret_cast<const uint8_t *>(text);
        }
    }

    uint8_t localPat[16];
    const uint8_t *pPtr = reinterpret_cast<const uint8_t *>(pattern);
    
    {
        if (maxSafeBytesPat < 16)
        {
            memset(localPat, 0, sizeof(localPat));
            size_t n = (maxSafeBytesPat < length) ? maxSafeBytesPat : length;
            memcpy(localPat, pattern, n);
            pPtr = localPat;
        }
    }

    
    uint8x16_t vt = vld1q_u8(tPtr);
    uint8x16_t vp = vld1q_u8(pPtr);

    
    
    alignas(16) uint8_t maskData[16];
    memset(maskData, 0, sizeof(maskData));
    for (size_t i = 0; i < length; i++)
    {
        maskData[i] = 0xFF;
    }
    uint8x16_t vMask = vld1q_u8(maskData);

    
    uint8x16_t diff = veorq_u8(vt, vp);
    diff = vandq_u8(diff, vMask);

    
    uint64x2_t c64 = vreinterpretq_u64_u8(diff);
    uint64_t or_val = (vgetq_lane_u64(c64, 0) | vgetq_lane_u64(c64, 1));

    return (or_val == 0);

#else
    
    return (memcmp(text, pattern, length) == 0);
#endif
}

static inline bool cmp_64_asm(const char *text, const char *pattern, size_t length)
{
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    size_t chunks = length / 16;
    size_t leftover = 0;
    __asm__ volatile(
        "udiv  x0, %x1, %x2       \n" 
        "msub  %0, x0, %x2, %x1   \n" 
        : "=r"(leftover)
        : "r"(length), "r"((uint64_t)16)
        : "x0");

    for (size_t i = 0; i < chunks; i++)
    {
        __builtin_prefetch(text + i * 16, 0, 0);
        uint8x16_t vt = vld1q_u8(reinterpret_cast<const uint8_t *>(text + i * 16));
        uint8x16_t vp = vld1q_u8(reinterpret_cast<const uint8_t *>(pattern + i * 16));
        uint8x16_t cmp = vceqq_u8(vt, vp);
        uint64x2_t c64 = vreinterpretq_u64_u8(cmp);
        if (vgetq_lane_u64(c64, 0) != UINT64_MAX ||
            vgetq_lane_u64(c64, 1) != UINT64_MAX)
        {
            return false;
        }
    }

    
    if (leftover)
    {
        return memcmp(text + chunks * 16, pattern + chunks * 16, leftover) == 0;
    }
    return true;
#else
    
    return memcmp(text, pattern, length) == 0;
#endif
}

static inline bool nncmp(const char *text,
                         const char *pattern,
                         size_t pTlIllIlI,
                         const char *bufBase, 
                         size_t bufLen)       
{
    bool is_le16 = ((pTlIllIlI & ~0xF) == 0) || (pTlIllIlI == 0x10);
    if (is_le16)
    {
        
        return cmp_16_asm_safe(text, pattern, pTlIllIlI,
                               bufBase, bufLen);
    }
    bool is_le64 = ((pTlIllIlI & ~0x3F) == 0) || (pTlIllIlI == 0x40);
    if (is_le64)
    {
        
        
        return cmp_64_asm(text, pattern, pTlIllIlI);
    }
    else
    {
        
        return memcmp(text, pattern, pTlIllIlI) == 0;
    }
}

static inline void do_search(const char *filename,
                             const char *buf,
                             size_t buflen,
                             const char *pattern,
                             size_t p4IlIIlIlIllIlI,
                             int64_t baseOffset)
{
    if (!buf || !pattern || p4IlIIlIlIllIlI == 0 || buflen < p4IlIIlIlIllIlI)
        return;

    static thread_local int badChar[ASCII_SET_SIZE];
    BMH_BUILD_BADCHAR_TABLE(pattern, p4IlIIlIlIllIlI, badChar);

    size_t i = 0;
    while (i + p4IlIIlIlIllIlI <= buflen)
    {
        size_t tailCheckLen = (p4IlIIlIlIllIlI > 64) ? 64 : p4IlIIlIlIllIlI;
        size_t offsetTail = p4IlIIlIlIllIlI - tailCheckLen;

        __builtin_prefetch(buf + i + 128, 0, 0);
        if (!nncmp(buf + i + offsetTail,
                   pattern + offsetTail,
                   tailCheckLen,
                   buf, buflen)) 
        {
            unsigned char mismatch = (unsigned char)buf[i + p4IlIIlIlIllIlI - 1];
            int shift = badChar[mismatch];
            if (shift < 1)
                shift = 1;
            i += (size_t)shift;
            continue;
        }

        if (offsetTail > 0)
        {
            if (!nncmp(buf + i,
                       pattern,
                       offsetTail,
                       buf, buflen)) 
            {
                unsigned char mismatch = (unsigned char)buf[i + p4IlIIlIlIllIlI - 1];
                int shift = badChar[mismatch];
                if (shift < 1)
                    shift = 1;
                i += (size_t)shift;
                continue;
            }
        }
        
        SearchResult sr;
        memset(&sr, 0, sizeof(sr));
        strncpy(sr.filename, filename, sizeof(sr.filename) - 1);
        sr.offset = baseOffset + i;
        sr.closenessScore = 0;

        char snippet[MAX_LINE_PREVIEW];
        single_line_snippet(buf, buflen, i, snippet, sizeof(snippet));
        strncpy(sr.preview, snippet, sizeof(sr.preview) - 1);

        while (!ring_push(sr))
        {
            sched_yield();
        }
        i += 1;
    }
}

/******************************************************************************
 * Threading for Sub-Chunks
 ******************************************************************************/
struct SubChunkTask
{
    const char *zstFile;
    const char *buffer;
    size_t bufStart;
    size_t bufEnd;
    int64_t offsetBase;
    const char *pattern;
    size_t patLen;
};

static void *subchunk_worker(void *arg)
{
    SubChunkTask *t = reinterpret_cast<SubChunkTask *>(arg);
    size_t chunkLen = t->bufEnd - t->bufStart;
    do_search(t->zstFile,
              t->buffer + t->bufStart,
              chunkLen,
              t->pattern,
              t->patLen,
              t->offsetBase + t->bufStart);
    return nullptr;
}

/******************************************************************************
 * Aligned Allocation
 ******************************************************************************/
static void *aligned_alloc_ext(size_t alignment, size_t size)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        return nullptr;
    }
    return ptr;
}

/******************************************************************************
 * Decompress and Search a Single Chunk (Reuses the open file descriptor)
 ******************************************************************************/
static void decompress_and_search_chunk(int fd,
                                        const char *zstFile,
                                        const ChunkInfo &C,
                                        const char *pattern,
                                        size_t patLen)
{
    int64_t length = C.uncompressed_end - C.uncompressed_start + 1;
    if (length <= 0)
        return;
    if (length > (int64_t)MAX_CHUNK_UNCOMPRESSED)
    {
        length = MAX_CHUNK_UNCOMPRESSED;
    }

    off_t off = C.compressed_offset;
    size_t sz = (size_t)C.compressed_size;
    if (lseek(fd, off, SEEK_SET) == (off_t)-1)
    {
        return;
    }

    
    void *cbuf = malloc(sz);
    if (!cbuf)
        return;
    if (read(fd, cbuf, sz) != (ssize_t)sz)
    {
        free(cbuf);
        return;
    }

    
    ZSTD_DCtx *dctx = ZSTD_createDCtx();
    if (!dctx)
    {
        free(cbuf);
        return;
    }

    void *outBuf = aligned_alloc_ext(ALIGNMENT, (size_t)length);
    if (!outBuf)
    {
        ZSTD_freeDCtx(dctx);
        free(cbuf);
        return;
    }

    ZSTD_inBuffer zin = {cbuf, sz, 0};
    ZSTD_outBuffer zout = {outBuf, (size_t)length, 0};
    size_t zr = 1;
    while (zr != 0 && !ZSTD_isError(zr) && zout.pos < zout.size)
    {
        zr = ZSTD_decompressStream(dctx, &zout, &zin);
    }
    ZSTD_freeDCtx(dctx);
    free(cbuf);

    if (ZSTD_isError(zr))
    {
        free(outBuf);
        return;
    }

    size_t actualSize = zout.pos;
    if (actualSize < MIN_SUBCHUNK_PARALLEL)
    {
        do_search(zstFile, static_cast<char *>(outBuf), actualSize,
                  pattern, patLen, C.uncompressed_start);
    }
    else
    {
        int subThreads = MAX_SUBCHUNK_THREADS;
        if (subThreads > static_cast<int>(actualSize / MIN_SUBCHUNK_PARALLEL))
        {
            subThreads = static_cast<int>(actualSize / MIN_SUBCHUNK_PARALLEL);
            if (subThreads < 1)
            {
                subThreads = 1;
            }
        }

        std::vector<pthread_t> tids(subThreads);
        std::vector<SubChunkTask> tasks(subThreads);

        size_t chunkSize = actualSize / subThreads;
        size_t remainder = actualSize % subThreads;
        size_t start = 0;
        for (int i = 0; i < subThreads; i++)
        {
            size_t load = chunkSize + ((size_t)i < remainder ? 1 : 0);
            tasks[i].zstFile = zstFile;
            tasks[i].buffer = static_cast<const char *>(outBuf);
            tasks[i].bufStart = start;
            tasks[i].bufEnd = start + load;
            tasks[i].offsetBase = C.uncompressed_start;
            tasks[i].pattern = pattern;
            tasks[i].patLen = patLen;
            pthread_create(&tids[i], nullptr, subchunk_worker, &tasks[i]);
            start += load;
        }
        for (int i = 0; i < subThreads; i++)
        {
            pthread_join(tids[i], nullptr);
        }
    }

    free(outBuf);
}

/******************************************************************************
 * Search an Indexed File (Opens the .tar.zst once)
 ******************************************************************************/
static void search_indexed_file(const char *zstFile,
                                const char *idxFile,
                                const char *pattern,
                                size_t patLen)
{
    IndexInfo *info = parse_index_json(idxFile);
    if (!info)
        return;

    int fd = open(zstFile, O_RDONLY);
    if (fd < 0)
    {
        free_index_info(info);
        return;
    }

    for (size_t i = 0; i < info->count; i++)
    {
        decompress_and_search_chunk(fd, zstFile, info->chunks[i],
                                    pattern, patLen);
    }

    close(fd);
    free_index_info(info);
}

/******************************************************************************
 * Worker and Printer Threads
 ******************************************************************************/
struct WorkerArg
{
    const char **zstFiles;
    const char **idxFiles;
    int start;
    int end;
    const char *pattern;
    size_t patLen;
};

static void *worker_thread(void *arg)
{
    WorkerArg *W = reinterpret_cast<WorkerArg *>(arg);
    for (int i = W->start; i < W->end; i++)
    {
        search_indexed_file(W->zstFiles[i],
                            W->idxFiles[i],
                            W->pattern,
                            W->patLen);
    }
    return nullptr;
}

static void *printer_thread(void *unused)
{
    (void)unused;
    SearchResult sr;
    while (!ringDone)
    {
        while (ring_pop(sr))
        {
            char escaped[2 * MAX_LINE_PREVIEW];
            json_escape(sr.preview, escaped, sizeof(escaped));
            printf("{\"file\":\"%s\",\"offset\":%lld,\"score\":%d,\"preview\":\"%s\"}\n",
                   sr.filename, (long long)sr.offset, sr.closenessScore, escaped);
        }
        sched_yield();
    }
    
    while (ring_pop(sr))
    {
        char escaped[2 * MAX_LINE_PREVIEW];
        json_escape(sr.preview, escaped, sizeof(escaped));
        printf("{\"file\":\"%s\",\"offset\":%lld,\"score\":%d,\"preview\":\"%s\"}\n",
               sr.filename, (long long)sr.offset, sr.closenessScore, escaped);
    }
    return nullptr;
}

/******************************************************************************
 * main()
 ******************************************************************************/
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <pattern>\n", argv[0]);
        return 1;
    }
    const char *pattern = argv[1];
    size_t patLen = strlen(pattern);

    DIR *d = opendir(".");
    if (!d)
    {
        fprintf(stderr, "Cannot open current directory.\n");
        return 1;
    }

    size_t capacity = 64;
    std::vector<char *> zstFiles;
    std::vector<char *> idxFiles;
    zstFiles.reserve(capacity);
    idxFiles.reserve(capacity);

    struct dirent *de;
    while ((de = readdir(d)) != nullptr)
    {
        if (fnmatch("batch_[0-9][0-9][0-9][0-9][0-9].tar.zst", de->d_name, 0) == 0)
        {
            char idxName[MAX_FILENAME_LEN];
            strncpy(idxName, de->d_name, sizeof(idxName));
            idxName[sizeof(idxName) - 1] = '\0';
            char *p = strstr(idxName, ".tar.zst");
            if (!p)
                continue;
            strcpy(p, ".tar.idx.json");

            struct stat st;
            if (stat(idxName, &st) == 0)
            {
                zstFiles.push_back(strdup(de->d_name));
                idxFiles.push_back(strdup(idxName));
            }
        }
    }
    closedir(d);

    if (zstFiles.empty())
    {
        fprintf(stderr, "No matching .tar.zst / .tar.idx.json pairs found.\n");
        return 0;
    }

    
    pthread_t ptid;
    pthread_create(&ptid, nullptr, printer_thread, nullptr);

    
    size_t count = zstFiles.size();
    int thrCount = (count < (size_t)MAX_THREADS) ? (int)count : MAX_THREADS;
    std::vector<pthread_t> tids(thrCount);
    std::vector<WorkerArg> wargs(thrCount);

    int per = (int)count / thrCount;
    int rem = (int)count % thrCount;
    int start = 0;
    for (int i = 0; i < thrCount; i++)
    {
        int load = per + (i < rem ? 1 : 0);
        wargs[i].zstFiles = const_cast<const char **>(zstFiles.data());
        wargs[i].idxFiles = const_cast<const char **>(idxFiles.data());
        wargs[i].start = start;
        wargs[i].end = start + load;
        wargs[i].pattern = pattern;
        wargs[i].patLen = patLen;
        pthread_create(&tids[i], nullptr, worker_thread, &wargs[i]);
        start += load;
    }

    
    for (int i = 0; i < thrCount; i++)
    {
        pthread_join(tids[i], nullptr);
    }

    
    ringDone = true;
    pthread_join(ptid, nullptr);

    
    for (size_t i = 0; i < count; i++)
    {
        free(zstFiles[i]);
        free(idxFiles[i]);
    }

    return 0;
}
