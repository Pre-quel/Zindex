/******************************************************************************
 # Zindex_o2

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

#define MAX_THREADS 10
#define PARTIAL_DECOMPRESS_BYTES 65536ULL
#define MAX_QUEUE_SIZE 65536ULL
#define MAX_LINE_PREVIEW 4096
#define MAX_FILENAME_LEN 512
#define ASCII_SET_SIZE 256

// If the chunk is huge, limit how much to decompress at once
// (some .tar.zst might have big uncompressed sizes per chunk).
#define MAX_CHUNK_UNCOMPRESSED (1024 * 1024ULL * 1024ULL)

/*****************************************************************************
 * SIMD / NEON utilities
 *   - Do a partial NEON optimization for the "compare loop" in
 *     Boyer–Moore (or simpler Horspool) approach.
 *****************************************************************************/

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

// 4x unrolled NEON comparison w/ alignment hints
static inline bool neon_compare_aarch64(const char *text, const char *pat,
                                        size_t len) {
    if (len < 64) return memcmp(text, pat, len) == 0;

    const uint8x16_t *t =
        (const uint8x16_t *)__builtin_assume_aligned(text, 16);
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
#else
static inline bool neon_compare_unrolled(const char *text, const char *pattern,
                                         size_t length) {
    if (!text || !pattern || length == 0) {
        return false;
    }
    return memcmp(text, pattern, length) == 0;
}
#endif

/*****************************************************************************
 * Murmur3 Hash, inline
 *****************************************************************************/
static inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

static inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

/*****************************************************************************
 * Base64 decode (for bloom filter bit array)
 *****************************************************************************/
static uint8_t *base64_decode_neon(const char *encoded, size_t encoded_len,
                                   uint8_t *out_buffer, size_t out_buffer_size,
                                   size_t *out_len) {
    if (encoded == NULL || out_len == NULL || encoded_len == 0) {
        if (out_len) *out_len = 0;
        return NULL;
    }

    static bool initialized = false;
    static int8_t d_table[256] __attribute__((aligned(16)));

    if (!initialized) {
        memset(d_table, -1, 256);

        for (int i = 0; i < 26; i++) {
            d_table['A' + i] = i;
            d_table['a' + i] = i + 26;
        }
        for (int i = 0; i < 10; i++) {
            d_table['0' + i] = i + 52;
        }
        d_table['+'] = 62;
        d_table['/'] = 63;
        d_table['='] = -2;

        initialized = true;
    }

    size_t padding = 0;

    if (encoded_len > 0 && encoded[encoded_len - 1] == '=') padding++;
    if (encoded_len > 1 && encoded[encoded_len - 2] == '=') padding++;

    size_t output_len = (encoded_len * 3) / 4 - padding;

    if (output_len == 0) {
        *out_len = 0;
        return out_buffer ? out_buffer : (uint8_t *)calloc(1, 1);
    }

    uint8_t *result = NULL;
    if (out_buffer) {
        if (out_buffer_size < output_len) {
            *out_len = 0;
            return NULL;
        }
        result = out_buffer;
    } else {
        result = (uint8_t *)calloc(output_len + 1, 1);
        if (result == NULL) {
            *out_len = 0;
            return NULL;
        }
    }

    const uint8_t *src = (const uint8_t *)encoded;
    uint8_t *dst = result;
    size_t dst_idx = 0;
    size_t src_idx = 0;

    while (src_idx + 64 <= encoded_len && dst_idx + 48 <= output_len) {
        bool all_valid = true;
        for (int i = 0; i < 64; i++) {
            int8_t val = d_table[src[src_idx + i]];
            if (val < 0 && val != -2) {
                all_valid = false;
                break;
            }
        }

        if (!all_valid) break;

        for (int chunk = 0; chunk < 4; chunk++) {
            size_t chunk_offset = chunk * 16;
            const uint8_t *chunk_src = src + src_idx + chunk_offset;

            uint8x16_t input = vld1q_u8(chunk_src);

            uint8_t values[16];
            vst1q_u8(values, input);

            for (int i = 0; i < 16; i++) {
                values[i] = d_table[values[i]];
            }

            for (int group = 0; group < 4; group++) {
                int base = group * 4;

                uint8_t v0 = values[base];
                uint8_t v1 = values[base + 1];
                uint8_t v2 = values[base + 2];
                uint8_t v3 = values[base + 3];

                uint8_t out0 = (v0 << 2) | ((v1 & 0x30) >> 4);
                uint8_t out1 = ((v1 & 0x0F) << 4) | ((v2 & 0x3C) >> 2);
                uint8_t out2 = ((v2 & 0x03) << 6) | v3;

                size_t out_offset = dst_idx + chunk * 12 + group * 3;
                if (out_offset + 2 < output_len) {
                    dst[out_offset] = out0;
                    dst[out_offset + 1] = out1;
                    dst[out_offset + 2] = out2;
                }
            }
        }

        src_idx += 64;
        dst_idx += 48;
    }

    while (src_idx + 16 <= encoded_len && dst_idx + 12 <= output_len) {
        bool all_valid = true;
        for (int i = 0; i < 16; i++) {
            int8_t val = d_table[src[src_idx + i]];
            if (val < 0 && val != -2) {
                all_valid = false;
                break;
            }
        }

        if (!all_valid) break;

        uint8x16_t input = vld1q_u8(src + src_idx);
        uint8_t values[16];
        vst1q_u8(values, input);

        for (int i = 0; i < 16; i++) {
            values[i] = d_table[values[i]];
        }

        for (int group = 0; group < 4; group++) {
            int base = group * 4;
            uint8_t v0 = values[base];
            uint8_t v1 = values[base + 1];
            uint8_t v2 = values[base + 2];
            uint8_t v3 = values[base + 3];

            uint8_t out0 = (v0 << 2) | ((v1 & 0x30) >> 4);
            uint8_t out1 = ((v1 & 0x0F) << 4) | ((v2 & 0x3C) >> 2);
            uint8_t out2 = ((v2 & 0x03) << 6) | v3;

            size_t out_offset = dst_idx + group * 3;
            if (out_offset + 2 < output_len) {
                dst[out_offset] = out0;
                dst[out_offset + 1] = out1;
                dst[out_offset + 2] = out2;
            }
        }

        src_idx += 16;
        dst_idx += 12;
    }

    uint32_t bits = 0;
    int bits_collected = 0;

    while (src_idx < encoded_len && dst_idx < output_len) {
        uint8_t c = src[src_idx++];
        int8_t val = d_table[c];

        if (val < 0) continue;

        bits = (bits << 6) | val;
        bits_collected += 6;

        if (bits_collected >= 8) {
            bits_collected -= 8;
            dst[dst_idx++] = (bits >> bits_collected) & 0xFF;
        }
    }

    *out_len = dst_idx;

    if (dst_idx < output_len) {
        dst[dst_idx] = '\0';
    }

    return result;
}

// match Python
static inline uint32_t MurmurHash3_x86_32(const void *key, size_t len,
                                          uint32_t seed) {
    if (!key || len == 0) {
        return seed;
    }

    const uint8_t *data = (const uint8_t *)key;
    const int nblocks = (int)(len / 4);

    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint32_t *blocks = (const uint32_t *)(data);
    for (int i = 0; i < nblocks; i++) {
        uint32_t k1;

        memcpy(&k1, &blocks[i], sizeof(uint32_t));

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t *tail = data + nblocks * 4;
    uint32_t k1 = 0;

    switch (len & 3) {
        case 3:
            k1 ^= tail[2] << 16; /* fallthrough */
        case 2:
            k1 ^= tail[1] << 8; /* fallthrough */
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
    }

    h1 ^= (uint32_t)len;
    h1 = fmix32(h1);

    return h1;
}

/*****************************************************************************
 * Bloom Filter
 *****************************************************************************/
typedef struct {
    int size_bits;
    int num_hashes;
    uint8_t *bit_array;
    size_t bit_array_len;
} BloomFilterC;

// Returns true if `pattern` might be in bloom, false if definitely not
static inline bool bloom_filter_might_contain(const BloomFilterC *bf,
                                              const char *pattern) {
    if (!bf || !bf->bit_array || bf->size_bits <= 0 || bf->num_hashes <= 0 ||
        !pattern) {
        return true;
    }

    size_t length = strlen(pattern);

    uint32_t h1 = MurmurHash3_x86_32(pattern, length, 0);
    uint32_t h2 = MurmurHash3_x86_32(pattern, length, 1);

    for (int i = 0; i < bf->num_hashes; i++) {
        uint64_t combinedHash = (uint64_t)h1 + i * (uint64_t)h2;

        uint64_t bitPosition = combinedHash % bf->size_bits;
        uint64_t byteIndex = bitPosition / 8;
        uint8_t bitOffset = bitPosition % 8;

        if (byteIndex >= bf->bit_array_len) {
            continue;
        }

        if ((bf->bit_array[byteIndex] & (1 << bitOffset)) == 0) {
            return false;
        }
    }

    return true;
}

// match Python's tokenizer
static bool handle_url_tokens(const BloomFilterC *bf, const char *url) {
    char *url_copy = strdup(url);
    if (!url_copy) return false;

    bool result = false;

    if (bloom_filter_might_contain(bf, url_copy)) {
        result = true;
        free(url_copy);
    }

    char *scheme_end = strstr(url_copy, "://");
    char *netloc = NULL;
    char *path = NULL;

    if (scheme_end) {
        netloc = scheme_end + 3;
    } else if (strncmp(url_copy, "www.", 4) == 0) {
        netloc = url_copy;
    } else {
        free(url_copy);
        return result;
    }

    path = strchr(netloc, '/');
    if (path) {
        *path = '\0';
        path++;
    }

    if (bloom_filter_might_contain(bf, netloc)) {
        result = true;
        return result;
    }

    char *domain_parts[10] = {0};
    int part_count = 0;

    char *domain_copy = strdup(netloc);
    if (domain_copy) {
        char *saveptr = NULL;
        char *part = strtok_r(domain_copy, ".", &saveptr);

        while (part && part_count < 10) {
            domain_parts[part_count++] = strdup(part);
            part = strtok_r(NULL, ".", &saveptr);
        }

        if (part_count > 2) {
            char main_domain[256] = {0};
            snprintf(main_domain, sizeof(main_domain), "%s.%s",
                     domain_parts[part_count - 2],
                     domain_parts[part_count - 1]);

            if (bloom_filter_might_contain(bf, main_domain)) {
                result = true;
            }
        }

        for (int i = 0; i < part_count; i++) {
            free(domain_parts[i]);
        }
        free(domain_copy);
    }

    if (result) {
        free(url_copy);
        return result;
    }

    if (path) {
        path[-1] = '/';
    }

    if (path && bloom_filter_might_contain(bf, path)) {
        result = true;
        free(url_copy);
        return result;
    }

    if (path) {
        char *path_copy = strdup(path);
        if (path_copy) {
            char *saveptr = NULL;
            char *segment = strtok_r(path_copy, "/", &saveptr);
            int segment_idx = 0;

            while (segment) {
                if (bloom_filter_might_contain(bf, segment)) {
                    result = true;
                    free(path_copy);
                    free(url_copy);
                    return result;
                }

                if (segment_idx == 0) {
                    char combined[512] = {0};
                    snprintf(combined, sizeof(combined), "%s/%s", netloc,
                             segment);

                    if (bloom_filter_might_contain(bf, combined)) {
                        result = true;
                        free(path_copy);
                        free(url_copy);
                        return result;
                    }

                    strncat(combined, "/",
                            sizeof(combined) - strlen(combined) - 1);
                    if (bloom_filter_might_contain(bf, combined)) {
                        result = true;
                        free(path_copy);
                        free(url_copy);
                        return result;
                    }
                }

                segment = strtok_r(NULL, "/", &saveptr);
                segment_idx++;
            }

            free(path_copy);
        }
    }

    free(url_copy);
    return result;
}

// Handle emails exactly like Python's extract_tokens_from_chunk
static bool handle_email_tokens(const BloomFilterC *bf, const char *email) {
    char *email_copy = strdup(email);
    if (!email_copy) return false;

    bool result = false;

    if (bloom_filter_might_contain(bf, email_copy)) {
        result = true;
        free(email_copy);
        return result;
    }

    char *at_sign = strchr(email_copy, '@');
    if (!at_sign) {
        free(email_copy);
        return false;
    }

    *at_sign = '\0';
    char *local = email_copy;
    char *domain = at_sign + 1;

    if (strlen(local) >= 4 && bloom_filter_might_contain(bf, local)) {
        result = true;
        free(email_copy);
        return result;
    }

    if (strlen(domain) >= 4 && bloom_filter_might_contain(bf, domain)) {
        result = true;
        free(email_copy);
        return result;
    }

    char *domain_parts[10] = {0};
    int part_count = 0;

    char *domain_copy = strdup(domain);
    if (domain_copy) {
        char *saveptr = NULL;
        char *part = strtok_r(domain_copy, ".", &saveptr);

        while (part && part_count < 10) {
            domain_parts[part_count++] = strdup(part);
            part = strtok_r(NULL, ".", &saveptr);
        }

        for (int i = 1; i < part_count; i++) {
            char subdomain[256] = {0};

            for (int j = i; j < part_count; j++) {
                if (j > i) strcat(subdomain, ".");
                strcat(subdomain, domain_parts[j]);
            }

            if (strlen(subdomain) >= 4 &&
                bloom_filter_might_contain(bf, subdomain)) {
                result = true;
                break;
            }
        }

        for (int i = 0; i < part_count; i++) {
            free(domain_parts[i]);
        }
        free(domain_copy);
    }

    free(email_copy);
    return result;
}

static bool bloom_filter_might_contain_full(const BloomFilterC *bf,
                                            const char *pattern) {
    if (!pattern || !bf) {
        return true;
    }

    if (bloom_filter_might_contain(bf, pattern)) {
        return true;
    }

    size_t pattern_len = strlen(pattern);
    if (pattern_len == 0) {
        return true;
    }

    char *lowercase_pattern = (char *)malloc(pattern_len + 1);
    if (!lowercase_pattern) {
        return true;
    }

    for (size_t i = 0; i < pattern_len; i++) {
        lowercase_pattern[i] = tolower((unsigned char)pattern[i]);
    }
    lowercase_pattern[pattern_len] = '\0';

    bool lower_check = bloom_filter_might_contain(bf, lowercase_pattern);
    free(lowercase_pattern);

    if (lower_check) {
        return true;
    }

    if (!strpbrk(pattern, ".:;|,@#=")) return false;

    char *token_copy = (char *)malloc(pattern_len + 1);
    if (!token_copy) return true;

    strcpy(token_copy, pattern);
    bool result = false;

    if (strncmp(pattern, "http://", 7) == 0 ||
        strncmp(pattern, "https://", 8) == 0 ||
        strncmp(pattern, "www.", 4) == 0) {
        result = handle_url_tokens(bf, token_copy);
    }

    else if (strchr(pattern, '@')) {
        result = handle_email_tokens(bf, token_copy);
    }

    else {
        const char *delims = ".:;|,@#=";
        char *saveptr;
        char *token = strtok_r(token_copy, delims, &saveptr);

        while (token) {
            if (strlen(token) >= 4 && strlen(token) <= 128) {
                if (bloom_filter_might_contain(bf, token)) {
                    result = true;
                    break;
                }

                char *lowercase_token = (char *)malloc(strlen(token) + 1);
                if (lowercase_token) {
                    for (size_t i = 0; i < strlen(token); i++) {
                        lowercase_token[i] = tolower((unsigned char)token[i]);
                    }
                    lowercase_token[strlen(token)] = '\0';

                    if (bloom_filter_might_contain(bf, lowercase_token)) {
                        result = true;
                    }
                    free(lowercase_token);
                    if (result) break;
                }
            }
            token = strtok_r(NULL, delims, &saveptr);
        }
    }

    free(token_copy);
    return result;
}
/*****************************************************************************
 * Chunk & Index Data
 *****************************************************************************/
typedef struct {
    int64_t chunk_id;
    int64_t compressed_offset;
    int64_t compressed_size;
    int64_t uncompressed_start;
    int64_t uncompressed_end;

    bool hasBloom;
    BloomFilterC bloom;
} ChunkInfo;

typedef struct {
    ChunkInfo *chunks;
    size_t count;
} IndexInfo;

/*****************************************************************************
 * Search Results + Print Queue
 *****************************************************************************/
typedef struct {
    char filename[MAX_FILENAME_LEN];
    char internalFilename[MAX_FILENAME_LEN];
    int64_t offset;
    int closenessScore;
    char preview[MAX_LINE_PREVIEW];
} SearchResult;

typedef struct PrintNode {
    SearchResult item;
    struct PrintNode *next;
} PrintNode;

typedef struct {
    PrintNode *head;
    PrintNode *tail;
    size_t size;
    size_t max_size;
    bool done;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} PrintQueue;

static void printqueue_init(PrintQueue *q) {
    if (!q) return;

    q->head = q->tail = NULL;
    q->size = 0;
    q->max_size = MAX_QUEUE_SIZE;
    q->done = false;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void printqueue_destroy(PrintQueue *q) {
    if (!q) return;

    pthread_mutex_lock(&q->lock);

    PrintNode *cur = q->head;
    while (cur) {
        PrintNode *tmp = cur->next;
        free(cur);
        cur = tmp;
    }
    q->head = q->tail = NULL;
    q->size = 0;

    pthread_mutex_unlock(&q->lock);

    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

static bool printqueue_push(PrintQueue *q, const SearchResult *res) {
    if (!q || !res) return false;

    PrintNode *node = (PrintNode *)malloc(sizeof(*node));
    if (!node) return false;

    memcpy(&node->item, res, sizeof(SearchResult));

    node->item.filename[MAX_FILENAME_LEN - 1] = '\0';
    node->item.preview[MAX_LINE_PREVIEW - 1] = '\0';

    node->next = NULL;

    pthread_mutex_lock(&q->lock);

    while (q->size >= q->max_size && !q->done) {
        pthread_cond_wait(&q->not_full, &q->lock);
    }

    if (q->done) {
        pthread_mutex_unlock(&q->lock);
        free(node);
        return false;
    }

    if (q->tail) {
        q->tail->next = node;
        q->tail = node;
    } else {
        q->head = q->tail = node;
    }
    q->size++;

    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);

    return true;
}

static bool printqueue_pop(PrintQueue *q, SearchResult *out) {
    if (!q || !out) return false;

    pthread_mutex_lock(&q->lock);

    while (q->size == 0 && !q->done) {
        pthread_cond_wait(&q->not_empty, &q->lock);
    }

    if (q->size == 0 && q->done) {
        pthread_mutex_unlock(&q->lock);
        return false;
    }

    if (!q->head) {
        pthread_mutex_unlock(&q->lock);
        return false;
    }

    PrintNode *node = q->head;
    q->head = node->next;

    if (!q->head) {
        q->tail = NULL;
    }
    q->size--;

    memcpy(out, &node->item, sizeof(SearchResult));
    free(node);

    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);

    return true;
}

static void printqueue_mark_done(PrintQueue *q) {
    if (!q) return;

    pthread_mutex_lock(&q->lock);
    q->done = true;

    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->lock);
}

/*****************************************************************************
 * Parsing .idx.json using simdjson
 *****************************************************************************/
static IndexInfo *parse_index_json(const char *idx_filename) {
    if (!idx_filename) {
        fprintf(stderr, "NULL index filename provided\n");
        return NULL;
    }

    simdjson::dom::parser parser;

    try {
        simdjson::dom::element doc = parser.load(idx_filename);

        simdjson::dom::array chunksArr = doc["chunks"];

        size_t n = 0;
        for (auto elem : chunksArr) {
            n++;
        }

        if (n == 0) {
            fprintf(stderr, "No chunks found in index file: %s\n",
                    idx_filename);
            return NULL;
        }

        IndexInfo *info = (IndexInfo *)calloc(1, sizeof(*info));
        if (!info) {
            fprintf(stderr, "Failed to allocate IndexInfo for: %s\n",
                    idx_filename);
            return NULL;
        }

        info->count = n;
        info->chunks = (ChunkInfo *)calloc(n, sizeof(ChunkInfo));
        if (!info->chunks) {
            fprintf(stderr, "Failed to allocate chunks array for: %s\n",
                    idx_filename);
            free(info);
            return NULL;
        }

        size_t i = 0;
        for (simdjson::dom::element chunk : chunksArr) {
            if (i >= n) {
                break;
            }

            ChunkInfo *C = &info->chunks[i];

            C->chunk_id = static_cast<int64_t>(chunk["chunk_id"]);
            C->compressed_offset =
                static_cast<int64_t>(chunk["compressed_offset"]);
            C->compressed_size = static_cast<int64_t>(chunk["compressed_size"]);
            C->uncompressed_start =
                static_cast<int64_t>(chunk["uncompressed_start"]);
            C->uncompressed_end =
                static_cast<int64_t>(chunk["uncompressed_end"]);

            C->hasBloom = false;
            C->bloom.bit_array = NULL;
            C->bloom.bit_array_len = 0;
            C->bloom.size_bits = 0;
            C->bloom.num_hashes = 0;

            auto bloom_filter_result = chunk["bloom_filter"];
            if (!bloom_filter_result.error()) {
                simdjson::dom::object bfObj = bloom_filter_result;

                auto size_result = bfObj["size"];
                auto num_hashes_result = bfObj["num_hashes"];
                auto bit_array_result = bfObj["bit_array_b64"];

                if (!size_result.error() && !num_hashes_result.error() &&
                    !bit_array_result.error()) {
                    int64_t size_bits = size_result;
                    int64_t num_hashes = num_hashes_result;
                    std::string_view bit_array_b64 = bit_array_result;

                    if (!bit_array_b64.empty()) {
                        std::string b64str(bit_array_b64);
                        size_t encoded_len = b64str.length();
                        size_t decoded_len = 0;

                        uint8_t *decoded = base64_decode_neon(
                            b64str.c_str(), encoded_len, NULL, 0, &decoded_len);

                        if (decoded && decoded_len > 0) {
                            C->hasBloom = true;
                            C->bloom.size_bits = static_cast<int>(size_bits);
                            C->bloom.num_hashes = static_cast<int>(num_hashes);
                            C->bloom.bit_array = decoded;
                            C->bloom.bit_array_len = decoded_len;
                        } else {
                            if (decoded) {
                                free(decoded);
                            }
                        }
                    }
                }
            }
            i++;
        }
        return info;
    } catch (const simdjson::simdjson_error &e) {
        fprintf(stderr, "JSON parsing error for %s: %s\n", idx_filename,
                e.what());
        return NULL;
    }
}

static void free_index_info(IndexInfo *info) {
    if (!info) {
        return;
    }

    if (info->chunks) {
        for (size_t i = 0; i < info->count; i++) {
            if (info->chunks[i].hasBloom && info->chunks[i].bloom.bit_array) {
                free(info->chunks[i].bloom.bit_array);
                info->chunks[i].bloom.bit_array = NULL;
            }
        }
        free(info->chunks);
    }

    free(info);
}

/*****************************************************************************
 * Boyer–Moore–Horspool with NEON
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

static bool boyer_moore_found_neon(const char *text, size_t textLen,
                                   const char *pattern, size_t patLen) {
    if (!text || !pattern || patLen == 0 || patLen > textLen) {
        return false;
    }

    int badChar[ASCII_SET_SIZE];
    build_badchar_table(pattern, patLen, badChar);

    size_t i = 0;
    while (i <= textLen - patLen) {
        if (patLen >= 16) {
            size_t tailOffset = patLen - 16;

            if (i + tailOffset + 16 > textLen) {
                return false;
            }

            if (!neon_compare_aarch64(text + i + tailOffset,
                                      pattern + tailOffset, 16)) {
                unsigned char mismatchChar =
                    (unsigned char)text[i + patLen - 1];
                int shift = badChar[mismatchChar];
                if (shift < 1) shift = 1;
                i += shift;
                continue;
            }

            ssize_t j = (ssize_t)tailOffset - 1;
            while (j >= 0 && text[i + j] == pattern[j]) {
                j--;
            }

            if (j < 0) {
                return true;
            }

            unsigned char mismatchChar = (unsigned char)text[i + patLen - 1];
            int shift = badChar[mismatchChar];
            if (shift < 1) shift = 1;
            i += shift;
        } else {
            ssize_t j = (ssize_t)patLen - 1;
            while (j >= 0 && text[i + j] == pattern[j]) {
                j--;
            }

            if (j < 0) {
                return true;
            }

            unsigned char mismatchChar = (unsigned char)text[i + patLen - 1];
            int shift = badChar[mismatchChar];
            if (shift < 1) shift = 1;
            i += shift;
        }
    }

    return false;
}

/*****************************************************************************
 * Better "Line Preview" + "Closeness" measure
 *****************************************************************************/
static void make_preview_and_score(const char *text, size_t textLen,
                                   size_t matchPos, size_t patLen,
                                   const char *pattern, SearchResult *outRes) {
    if (!text || !outRes || matchPos >= textLen) {
        if (outRes) {
            outRes->preview[0] = '\0';
            outRes->closenessScore = 0;
        }
        return;
    }

    size_t lineStart = matchPos;
    while (lineStart > 0 && text[lineStart - 1] != '\n') {
        lineStart--;
    }

    size_t lineEnd = matchPos;
    while (lineEnd < textLen && text[lineEnd] != '\n' &&
           text[lineEnd] != '\0') {
        lineEnd++;
    }

    if (lineEnd <= lineStart) {
        outRes->preview[0] = '\0';
        return;
    }

    size_t lineLen = lineEnd - lineStart;
    if (lineLen >= MAX_LINE_PREVIEW - 1) {
        lineLen = MAX_LINE_PREVIEW - 1;
    }

    memcpy(outRes->preview, text + lineStart, lineLen);
    outRes->preview[lineLen] = '\0';

    const char *extensions[] = {".txt", ".csv", ".json", ".md", ".log", NULL};
    char *possibleFilename = NULL;

    for (int i = 0; extensions[i] != NULL; i++) {
        char *extPos = strstr(outRes->preview, extensions[i]);
        if (extPos) {
            char *filenameStart = extPos;
            while (filenameStart > outRes->preview &&
                   !(*(filenameStart - 1) == ' ' ||
                     *(filenameStart - 1) == '/' ||
                     *(filenameStart - 1) == '\\' ||
                     *(filenameStart - 1) == ':')) {
                filenameStart--;
            }

            if (extPos - filenameStart + strlen(extensions[i]) > 3) {
                possibleFilename = filenameStart;
                size_t fnLen = extPos - filenameStart + strlen(extensions[i]);

                if (fnLen < MAX_FILENAME_LEN - 1) {
                    strncpy(outRes->internalFilename, possibleFilename, fnLen);
                    outRes->internalFilename[fnLen] = '\0';
                }
                break;
            }
        }
    }
}

// Replace the printer_thread function with this:
static void *printer_thread(void *arg) {
    PrintQueue *q = (PrintQueue *)arg;
    if (!q) {
        return NULL;
    }

    SearchResult sr;
    char buffer[MAX_LINE_PREVIEW + MAX_FILENAME_LEN * 2 + 256];

    while (printqueue_pop(q, &sr)) {
        int len;

        if (sr.internalFilename[0] != '\0') {
            len = snprintf(buffer, sizeof(buffer), "%s\n", sr.preview);
        } else {
            len = snprintf(buffer, sizeof(buffer), "%s\n", sr.preview);
        }

        if (len > 0 && len < (int)sizeof(buffer)) {
            write(STDOUT_FILENO, buffer, len);
        }
    }

    return NULL;
}

/*****************************************************************************
 * Exhaustive buffer Boyer–Moore–Horspool search
 *****************************************************************************/
static void search_in_buffer(PrintQueue *pqueue, const char *filename,
                             const char *text, size_t textLen,
                             const char *pattern, size_t patLen,
                             int64_t offsetBase) {
    if (!pqueue || !text || !pattern || !filename || patLen == 0 ||
        textLen < patLen) {
        return;
    }

    int badCharTable[ASCII_SET_SIZE];
    build_badchar_table(pattern, patLen, badCharTable);

    size_t i = 0;
    size_t match_count = 0;
    const size_t MAX_MATCHES_PER_CHUNK = 1000;

    while (i + patLen <= textLen && match_count < MAX_MATCHES_PER_CHUNK) {
        if (i + patLen > textLen) {
            break;
        }

        ssize_t j = (ssize_t)patLen - 1;
        bool matched = true;

        while (j >= 0) {
            if (i + j >= textLen || text[i + j] != pattern[j]) {
                matched = false;
                break;
            }
            j--;
        }

        if (matched) {
            SearchResult sr;
            memset(&sr, 0, sizeof(sr));

            strncpy(sr.filename, filename, sizeof(sr.filename) - 1);
            sr.filename[sizeof(sr.filename) - 1] = '\0';

            sr.offset = offsetBase + i;

            make_preview_and_score(text, textLen, i, patLen, pattern, &sr);

            if (sr.preview[0] != '\0') {
                if (!printqueue_push(pqueue, &sr)) {
                    break;
                }
                match_count++;
            }

            i++;
        } else {
            unsigned char c = (i + patLen - 1 < textLen)
                                  ? (unsigned char)text[i + patLen - 1]
                                  : 0;
            int shift = badCharTable[c];
            if (shift < 1) {
                shift = 1;
            }
            i += shift;
        }
    }
}

/*****************************************************************************
 * Full Decompress + Search
 *****************************************************************************/
static void full_decompress_and_search(PrintQueue *pqueue, const char *zstFile,
                                       const ChunkInfo *c, const char *pattern,
                                       size_t patLen) {
    if (!pqueue || !zstFile || !c || !pattern || patLen == 0) {
        return;
    }

    int64_t length = c->uncompressed_end - c->uncompressed_start + 1;
    if (length <= 0) {
        return;
    }

    if (length > (int64_t)MAX_CHUNK_UNCOMPRESSED) {
        length = MAX_CHUNK_UNCOMPRESSED;
    }

    int fd = -1;
    void *cbuf = NULL;
    void *outBuf = NULL;
    ZSTD_DCtx *dctx = NULL;
    bool using_mmap = false;

    fd = open(zstFile, O_RDONLY);
    if (fd < 0) {
        return;
    }

    off_t mapOff = c->compressed_offset;
    size_t mapSize = (size_t)c->compressed_size;

    if (mapOff < 0 || mapSize == 0) {
        close(fd);
        return;
    }

#if defined(__APPLE__)

    cbuf = mmap(NULL, mapSize, PROT_READ, MAP_PRIVATE, fd, mapOff);
    using_mmap = (cbuf != MAP_FAILED);

    if (cbuf == MAP_FAILED) {
        cbuf = malloc(mapSize);
        if (!cbuf) {
            close(fd);
            return;
        }

        if (lseek(fd, mapOff, SEEK_SET) == -1 ||
            read(fd, cbuf, mapSize) != (ssize_t)mapSize) {
            free(cbuf);
            close(fd);
            return;
        }
    }
#else

    cbuf = malloc(mapSize);
    if (!cbuf) {
        close(fd);
        return;
    }

    if (lseek(fd, mapOff, SEEK_SET) == -1 ||
        read(fd, cbuf, mapSize) != (ssize_t)mapSize) {
        free(cbuf);
        close(fd);
        return;
    }
#endif

    close(fd);
    fd = -1;

    dctx = ZSTD_createDCtx();
    if (!dctx) {
        if (using_mmap) {
#if defined(__APPLE__)
            munmap(cbuf, mapSize);
#endif
        } else if (cbuf) {
            free(cbuf);
        }
        return;
    }

    outBuf = malloc((size_t)length);
    if (!outBuf) {
        ZSTD_freeDCtx(dctx);
        if (using_mmap) {
#if defined(__APPLE__)
            munmap(cbuf, mapSize);
#endif
        } else if (cbuf) {
            free(cbuf);
        }
        return;
    }

    ZSTD_inBuffer zin = {cbuf, mapSize, 0};
    ZSTD_outBuffer zout = {outBuf, (size_t)length, 0};

    size_t zres = 1;
    while (zres != 0 && !ZSTD_isError(zres) && zout.pos < zout.size) {
        zres = ZSTD_decompressStream(dctx, &zout, &zin);
    }

    if (!ZSTD_isError(zres) && zout.pos > 0) {
        search_in_buffer(pqueue, zstFile, (const char *)outBuf, zout.pos,
                         pattern, patLen, c->uncompressed_start);
    }

    if (outBuf) free(outBuf);
    if (dctx) ZSTD_freeDCtx(dctx);

    if (using_mmap) {
#if defined(__APPLE__)
        if (cbuf != MAP_FAILED) {
            munmap(cbuf, mapSize);
        }
#endif
    } else if (cbuf) {
        free(cbuf);
    }
}

/*****************************************************************************
 * Searching a single .tar.zst + .tar.idx.json
 *****************************************************************************/
static void search_indexed_file(PrintQueue *pqueue, const char *zstFile,
                                const char *idxFile, const char *pattern,
                                size_t patLen) {
    if (!pqueue || !zstFile || !idxFile || !pattern || patLen == 0) {
        return;
    }

    IndexInfo *info = parse_index_json(idxFile);
    if (!info) {
        return;
    }

    for (size_t i = 0; i < info->count; i++) {
        ChunkInfo *C = &info->chunks[i];

        if (C->hasBloom) {
            if (!bloom_filter_might_contain_full(&C->bloom, pattern)) {
                continue;
            }
        }

        full_decompress_and_search(pqueue, zstFile, C, pattern, patLen);
    }

    free_index_info(info);
}

/*****************************************************************************
 * Threading
 *****************************************************************************/
typedef struct {
    const char **zstFiles;
    const char **idxFiles;
    int fileCount;

    int start;
    int end;

    const char *pattern;
    size_t patLen;

    PrintQueue *pqueue;

    pthread_t tid;
    int thread_id;
    bool running;
} WorkerArg;

static void *worker_thread(void *arg) {
    WorkerArg *W = (WorkerArg *)arg;
    if (!W) {
        return NULL;
    }

    W->running = true;

    for (int i = W->start; i < W->end && i < W->fileCount; i++) {
        if (!W->zstFiles[i] || !W->idxFiles[i]) {
            continue;
        }

        search_indexed_file(W->pqueue, W->zstFiles[i], W->idxFiles[i],
                            W->pattern, W->patLen);
    }

    W->running = false;
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

    if (patLen == 0) {
        fprintf(stderr, "Error: Empty search pattern\n");
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
        fprintf(stderr, "Memory allocation failed\n");
        if (zstFiles) free(zstFiles);
        if (idxFiles) free(idxFiles);
        closedir(d);
        return 1;
    }

    size_t capacity = initialCap;
    size_t count = 0;

    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (fnmatch("batch_[0-9][0-9][0-9][0-9][0-9].tar.zst", de->d_name, 0) ==
            0) {
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
                    char **newZstFiles = (char **)realloc(
                        zstFiles, capacity * sizeof(*zstFiles));
                    char **newIdxFiles = (char **)realloc(
                        idxFiles, capacity * sizeof(*idxFiles));

                    if (!newZstFiles || !newIdxFiles) {
                        fprintf(stderr, "Memory reallocation failed\n");

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
                    fprintf(stderr, "String duplication failed\n");

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

    PrintQueue pqueue;
    printqueue_init(&pqueue);

    pthread_t printerTid;
    if (pthread_create(&printerTid, NULL, printer_thread, &pqueue) != 0) {
        fprintf(stderr, "Failed to create printer thread\n");

        for (size_t i = 0; i < count; i++) {
            free(zstFiles[i]);
            free(idxFiles[i]);
        }
        free(zstFiles);
        free(idxFiles);
        printqueue_destroy(&pqueue);
        return 1;
    }

    int threadCount = (count < MAX_THREADS) ? (int)count : MAX_THREADS;
    WorkerArg *wargs = (WorkerArg *)calloc(threadCount, sizeof(WorkerArg));

    if (!wargs) {
        fprintf(stderr, "Failed to allocate worker thread arguments\n");

        printqueue_mark_done(&pqueue);
        pthread_join(printerTid, NULL);
        for (size_t i = 0; i < count; i++) {
            free(zstFiles[i]);
            free(idxFiles[i]);
        }
        free(zstFiles);
        free(idxFiles);
        printqueue_destroy(&pqueue);
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
        wargs[i].pqueue = &pqueue;
        wargs[i].thread_id = i;
        wargs[i].running = false;

        if (pthread_create(&wargs[i].tid, NULL, worker_thread, &wargs[i]) !=
            0) {
            fprintf(stderr, "Failed to create worker thread %d\n", i);

            continue;
        }

        start += load;
    }

    for (int i = 0; i < threadCount; i++) {
        if (wargs[i].running || wargs[i].tid != 0) {
            pthread_join(wargs[i].tid, NULL);
        }
    }

    printqueue_mark_done(&pqueue);

    pthread_join(printerTid, NULL);

    for (size_t i = 0; i < count; i++) {
        free(zstFiles[i]);
        free(idxFiles[i]);
    }
    free(zstFiles);
    free(idxFiles);
    free(wargs);

    printqueue_destroy(&pqueue);

    return 0;
}
