/* gguf_infer.c
   Single-file GGUF inference engine for Windows 2000 Beta 1.
   Compiles with Visual C++ 6.0 (C with __int64).
   Supports Llama-like models: RMSNorm, SwiGLU, RoPE, GQA.
   Quantization: F32, Q4_0, Q8_0.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <process.h>
#include <time.h>
#include <ctype.h>

typedef void *HINTERNET;
typedef unsigned short INTERNET_PORT;
#define WINHTTP_ACCESS_TYPE_DEFAULT_PROXY 0
#define WINHTTP_NO_PROXY_NAME ((LPCWSTR)0)
#define WINHTTP_NO_PROXY_BYPASS ((LPCWSTR)0)
#define WINHTTP_NO_REFERER ((LPCWSTR)0)
#define WINHTTP_NO_ADDITIONAL_HEADERS ((LPCWSTR)0)
#define WINHTTP_DEFAULT_ACCEPT_TYPES ((LPCWSTR*)0)
#define WINHTTP_FLAG_SECURE 0x00800000
#define WINHTTP_ADDREQ_FLAG_ADD 0x20000000
#define WINHTTP_QUERY_STATUS_CODE 19
#define WINHTTP_QUERY_FLAG_NUMBER 0x20000000
#define WINHTTP_HEADER_NAME_BY_INDEX ((LPCWSTR)0)
#define WINHTTP_NO_HEADER_INDEX ((LPDWORD)0)
#define INTERNET_SCHEME_HTTP 1
#define INTERNET_SCHEME_HTTPS 2

typedef HINTERNET (WINAPI *PFN_WinHttpOpen)(LPCWSTR, DWORD, LPCWSTR, LPCWSTR, DWORD);
typedef HINTERNET (WINAPI *PFN_WinHttpConnect)(HINTERNET, LPCWSTR, INTERNET_PORT, DWORD);
typedef HINTERNET (WINAPI *PFN_WinHttpOpenRequest)(HINTERNET, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR*, DWORD);
typedef BOOL (WINAPI *PFN_WinHttpAddRequestHeaders)(HINTERNET, LPCWSTR, DWORD, DWORD);
typedef BOOL (WINAPI *PFN_WinHttpSendRequest)(HINTERNET, LPCWSTR, DWORD, LPVOID, DWORD, DWORD, DWORD_PTR);
typedef BOOL (WINAPI *PFN_WinHttpReceiveResponse)(HINTERNET, LPVOID);
typedef BOOL (WINAPI *PFN_WinHttpQueryDataAvailable)(HINTERNET, LPDWORD);
typedef BOOL (WINAPI *PFN_WinHttpReadData)(HINTERNET, LPVOID, DWORD, LPDWORD);
typedef BOOL (WINAPI *PFN_WinHttpCloseHandle)(HINTERNET);
typedef BOOL (WINAPI *PFN_WinHttpQueryHeaders)(HINTERNET, DWORD, LPCWSTR, LPVOID, LPDWORD, LPDWORD);

/* --- Types --- */

#ifdef _MSC_VER
typedef unsigned __int64 u64;
#else
typedef unsigned long long u64;
#endif
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed char s8;

typedef enum {
    ARCH_LLAMA = 0,
    ARCH_LLAMA2,
    ARCH_LLAMA3,
    ARCH_LLAMA4,
    ARCH_MISTRAL,
    ARCH_MIXTRAL,
    ARCH_NEMO,
    ARCH_QWEN2,
    ARCH_QWEN3,
    ARCH_QWEN25,
    ARCH_GEMMA,
    ARCH_GEMMA2,
    ARCH_GEMMA3,
    ARCH_GPT2,
    ARCH_GPT_OSS,
    ARCH_GPT_NEOX,
    ARCH_GPT_J,
    ARCH_PHI,
    ARCH_PHI3,
    ARCH_PHI4,
    ARCH_STABLELM,
    ARCH_CODELLAMA,
    ARCH_FALCON,
    ARCH_BAICHUAN,
    ARCH_YI,
    ARCH_DEEPSEEK,
    ARCH_COMMAND_R,
    ARCH_GRANITE,
    ARCH_UNKNOWN
} Arch;

typedef struct { u16 d; u8 qs[16]; } BlockQ4;
typedef struct { u16 d; s8 qs[32]; } BlockQ8;
typedef struct { u16 d; u8 qh[4]; u8 qs[16]; } BlockQ5_0;
typedef struct { u16 d; u16 m; u8 qh[4]; u8 qs[16]; } BlockQ5_1;

typedef struct {
    char name[80];
    u32 n_dims;
    u64 dims[4];
    u32 type;
    u64 offset;
    void *data;
} Tensor;

typedef struct {
    u32 context_length;
    u32 embedding_length;
    u32 feed_forward_length;
    u32 block_count;
    u32 attention_head_count;
    u32 attention_head_count_kv;
    float attention_layer_norm_rms_epsilon;
    u32 rope_dimension_count;
    float rope_freq_base;
    u32 alignment;
    char architecture[32];
    char tokenizer_pre[32];
    char chat_template[1024];
} HParams;

typedef struct {
    u8 *base;
    size_t size;
    u64 n_tensors;
    Tensor *tensors;
    u64 data_offset;
    HParams hp;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#endif
} GGUFContext;

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, head_dim;
    float norm_eps, rope_theta, rsqrt_head_dim;
    int tok_embd_transposed; /* 1 if token_embd.weight is [dim, vocab] instead of [vocab, dim] */
    float *cached_embd;      /* exact f32 cache for transposed token embeddings/output */
    float *cached_output;    /* exact f32 cache for transposed output head */
    float *rope_cos;         /* precomputed RoPE cos table [max_seq_len][head_dim/2] */
    float *rope_sin;         /* precomputed RoPE sin table [max_seq_len][head_dim/2] */
    Tensor *tok_embd;
    Tensor *output_norm;
    Tensor *output;
    Tensor **attn_norm;
    Tensor **attn_q;
    Tensor **attn_k;
    Tensor **attn_v;
    Tensor **attn_o;
    Tensor **ffn_norm;
    Tensor **ffn_gate;
    Tensor **ffn_up;
    Tensor **ffn_down;
    Arch arch;
    float repeat_penalty;
} Model;

typedef struct {
    float *x, *q, *k, *v, *attn_out;
    float *ffn_gate, *ffn_up, *ffn_hidden, *logits, *tmp;
    float *k_cache, *v_cache, *attn_scores;
    float *dq_row;
} RunState;

typedef struct { int len; char *text; } TokenEntry;

/* --- Globals --- */

static TokenEntry *g_vocab = NULL;
static int g_vocab_n = 0;
static u8 *g_tok_buf = NULL;
static int g_clean_output = 0;
static int g_n_threads = 0;

static int contains_nocase(const char *haystack, const char *needle) {
    size_t nlen, i;
    if (!haystack || !needle) return 0;
    nlen = strlen(needle);
    if (nlen == 0) return 1;
    for (; *haystack; haystack++) {
        for (i = 0; i < nlen; i++) {
            unsigned char hc = (unsigned char)haystack[i];
            unsigned char nc = (unsigned char)needle[i];
            if (!hc) break;
            if (tolower(hc) != tolower(nc)) break;
        }
        if (i == nlen) return 1;
    }
    return 0;
}

static double now_seconds(void) {
#ifdef _WIN32
    static double inv_freq = 0.0;
    LARGE_INTEGER freq;
    LARGE_INTEGER cur;
    if (inv_freq == 0.0) {
        QueryPerformanceFrequency(&freq);
        inv_freq = 1.0 / (double)freq.QuadPart;
    }
    QueryPerformanceCounter(&cur);
    return (double)cur.QuadPart * inv_freq;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* --- GGUF I/O --- */

static u32 read_u32(u8 **p) { u32 v; memcpy(&v, *p, 4); *p += 4; return v; }
static u64 read_u64(u8 **p) { u64 v; memcpy(&v, *p, 8); *p += 8; return v; }

static void skip_string(u8 **p) { u64 len = read_u64(p); *p += len; }

static void skip_value(u32 type, u8 **p) {
    u64 i, len;
    u32 atype;
    switch (type) {
        case 0: case 1: case 7: *p += 1; break;
        case 2: case 3: *p += 2; break;
        case 4: case 5: case 6: *p += 4; break;
        case 10: case 11: case 12: *p += 8; break;
        case 8: skip_string(p); break;
        case 9:
            atype = read_u32(p); len = read_u64(p);
            for (i = 0; i < len; i++) {
                if (atype == 8) skip_string(p);
                else if (atype == 9) { read_u32(p); read_u64(p); }
                else {
                    switch (atype) { case 0:case 1:case 7:*p+=1;break; case 2:case 3:*p+=2;break; case 4:case 5:case 6:*p+=4;break; case 10:case 11:case 12:*p+=8;break; default:break; }
                }
            }
            break;
    }
}

static int load_tokenizer_from_gguf(GGUFContext *ctx) {
    u8 *p = ctx->base;
    u64 meta_count;
    u64 i;
    if (memcmp(p, "GGUF", 4) != 0) return 0;
    p += 4;
    read_u32(&p); /* version */
    read_u64(&p); /* tensor_count */
    meta_count = read_u64(&p);
    for (i = 0; i < meta_count; i++) {
        u64 keylen = read_u64(&p);
        char key[256];
        u32 vtype;
        u64 len;
        u32 atype;
        u8 *start_of_value;
        if (keylen > 255) keylen = 255;
        memcpy(key, p, (size_t)keylen); key[keylen] = '\0'; p += keylen;
        vtype = read_u32(&p);
        start_of_value = p;
        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) {
            atype = read_u32(&p);
            len = read_u64(&p);
            if (atype == 8 && len > 0 && len < 1000000) {
                u64 tok_buf_size = 0;
                u8 *save_p = p;
                u64 k;
                for (k = 0; k < len; k++) {
                    u64 slen = read_u64(&save_p);
                    tok_buf_size += 1 + slen;
                    save_p += slen;
                }
                g_vocab = (TokenEntry*)malloc(sizeof(TokenEntry) * (size_t)len);
                g_tok_buf = (u8*)malloc((size_t)tok_buf_size);
                if (!g_vocab || !g_tok_buf) {
                    free(g_vocab); g_vocab = NULL;
                    free(g_tok_buf); g_tok_buf = NULL;
                    return 0;
                }
                g_vocab_n = (int)len;
                tok_buf_size = 0;
                for (k = 0; k < len; k++) {
                    u64 slen = read_u64(&p);
                    g_vocab[k].len = (int)slen;
                    g_vocab[k].text = (char*)(g_tok_buf + tok_buf_size);
                    memcpy(g_tok_buf + tok_buf_size, p, (size_t)slen);
                    tok_buf_size += slen;
                    g_tok_buf[tok_buf_size] = '\0';
                    tok_buf_size++;
                    p += slen;
                }
                printf("Loaded tokenizer from GGUF: %d tokens\n", g_vocab_n);
                return 1;
            }
        } else {
            skip_value(vtype, &p);
        }
        (void)start_of_value;
    }
    return 0;
}

static int meta_key_eq(const char *key, const char *prefix, const char *suffix) {
    size_t plen = strlen(prefix);
    size_t slen = strlen(suffix);
    size_t klen = strlen(key);
    if (klen != plen + slen) return 0;
    if (memcmp(key, prefix, plen) != 0) return 0;
    if (memcmp(key + plen, suffix, slen) != 0) return 0;
    return 1;
}

static void parse_metadata(u8 **p, u64 n_kv, HParams *hp) {
    u64 i;
    u32 vtype;
    u64 keylen;
    char key[256];
    memset(hp, 0, sizeof(HParams));
    hp->alignment = 32;
    hp->attention_layer_norm_rms_epsilon = 1e-6f;
    hp->rope_freq_base = 10000.0f;
    strcpy(hp->architecture, "llama");
    for (i = 0; i < n_kv; i++) {
        keylen = read_u64(p);
        if (keylen > 255) keylen = 255;
        memcpy(key, *p, (size_t)keylen); key[keylen] = '\0'; *p += keylen;
        vtype = read_u32(p);
        if (strcmp(key, "general.alignment") == 0 && vtype == 4) {
            hp->alignment = read_u32(p);
        } else if (meta_key_eq(key, "llama.", "context_length") ||
                   meta_key_eq(key, "qwen2.", "context_length") ||
                   strcmp(key, "context_length") == 0) {
            if (vtype == 4) hp->context_length = read_u32(p); else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "embedding_length") ||
                   meta_key_eq(key, "qwen2.", "embedding_length") ||
                   strcmp(key, "embedding_length") == 0) {
            if (vtype == 4) hp->embedding_length = read_u32(p); else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "feed_forward_length") ||
                   meta_key_eq(key, "qwen2.", "feed_forward_length") ||
                   strcmp(key, "feed_forward_length") == 0) {
            if (vtype == 4) hp->feed_forward_length = read_u32(p); else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "block_count") ||
                   meta_key_eq(key, "qwen2.", "block_count") ||
                   strcmp(key, "block_count") == 0) {
            if (vtype == 4) hp->block_count = read_u32(p); else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.attention.", "head_count") ||
                   meta_key_eq(key, "qwen2.attention.", "head_count") ||
                   strcmp(key, "attention.head_count") == 0) {
            if (vtype == 4) hp->attention_head_count = read_u32(p); else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.attention.", "head_count_kv") ||
                   meta_key_eq(key, "qwen2.attention.", "head_count_kv") ||
                   strcmp(key, "attention.head_count_kv") == 0) {
            if (vtype == 4) hp->attention_head_count_kv = read_u32(p); else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.attention.", "layer_norm_rms_epsilon") ||
                   meta_key_eq(key, "qwen2.attention.", "layer_norm_rms_epsilon") ||
                   strcmp(key, "attention.layer_norm_rms_epsilon") == 0) {
            if (vtype == 6) { float val; memcpy(&val, *p, 4); *p += 4; hp->attention_layer_norm_rms_epsilon = val; } else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.rope.", "dimension_count") ||
                   meta_key_eq(key, "qwen2.rope.", "dimension_count") ||
                   strcmp(key, "rope.dimension_count") == 0) {
            if (vtype == 4) hp->rope_dimension_count = read_u32(p); else skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.rope.", "freq_base") ||
                   meta_key_eq(key, "qwen2.rope.", "freq_base") ||
                   strcmp(key, "rope.freq_base") == 0) {
            if (vtype == 6) { float val; memcpy(&val, *p, 4); *p += 4; hp->rope_freq_base = val; } else skip_value(vtype, p);
        } else if (strcmp(key, "general.architecture") == 0 && vtype == 8) {
            u64 slen = read_u64(p);
            if (slen > 31) slen = 31;
            memcpy(hp->architecture, *p, (size_t)slen);
            hp->architecture[slen] = '\0';
            *p += slen;
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && vtype == 8) {
            u64 slen = read_u64(p);
            if (slen > 31) slen = 31;
            memcpy(hp->tokenizer_pre, *p, (size_t)slen);
            hp->tokenizer_pre[slen] = '\0';
            *p += slen;
        } else if (strcmp(key, "tokenizer.chat_template") == 0 && vtype == 8) {
            u64 slen = read_u64(p);
            if (slen > sizeof(hp->chat_template) - 1) slen = sizeof(hp->chat_template) - 1;
            memcpy(hp->chat_template, *p, (size_t)slen);
            hp->chat_template[slen] = '\0';
            *p += slen;
        } else {
            skip_value(vtype, p);
        }
    }
}

static u64 align_u64(u64 v, u32 a) { u64 mask = a - 1; return (v + mask) & ~mask; }

static GGUFContext *gguf_load(const char *path) {
    FILE *f;
    size_t read_bytes;
    u8 *p;
    u64 tensor_count, meta_count;
    u64 file_size, i;
    u64 pos_after_tensors;
    GGUFContext *ctx;

    ctx = (GGUFContext*)malloc(sizeof(GGUFContext));
    if (!ctx) { printf("Error: cannot allocate GGUFContext (%u bytes)\n", (unsigned int)sizeof(GGUFContext)); return NULL; }
    memset(ctx, 0, sizeof(GGUFContext));
    file_size = 0;

#ifdef _WIN32
    {
        HANDLE hFile, hMapping;
        DWORD sizeLow, sizeHigh;
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile != INVALID_HANDLE_VALUE) {
            sizeLow = GetFileSize(hFile, &sizeHigh);
            if (sizeLow != INVALID_FILE_SIZE || GetLastError() == NO_ERROR) {
                file_size = ((u64)sizeHigh << 32) | (u64)sizeLow;
                printf("Loading %s (%.1f MB) via memory map...\n", path, (double)file_size / (1024.0*1024.0));
                hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
                if (hMapping) {
                    ctx->base = (u8*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
                    if (ctx->base) {
                        ctx->hFile = hFile;
                        ctx->hMapping = hMapping;
                        ctx->size = (size_t)file_size;
                        goto loaded;
                    }
                    printf("Warning: MapViewOfFile failed (error %lu), falling back to fread\n", (unsigned long)GetLastError());
                    CloseHandle(hMapping);
                } else {
                    printf("Warning: CreateFileMapping failed (error %lu), falling back to fread\n", (unsigned long)GetLastError());
                }
            }
            CloseHandle(hFile);
        }
    }
#endif

    f = fopen(path, "rb");
    if (!f) { printf("Error: cannot open %s\n", path); free(ctx); return NULL; }
    fseek(f, 0, SEEK_END);
    file_size = (u64)ftell(f);
    fseek(f, 0, SEEK_SET);
    printf("Loading %s (%.1f MB) via fread...\n", path, (double)file_size / (1024.0*1024.0));
    ctx->size = (size_t)file_size;
    ctx->base = (u8*)malloc((size_t)file_size);
    if (!ctx->base) { printf("Error: cannot allocate %.1f MB for model file (out of memory?)\n", (double)file_size / (1024.0*1024.0)); free(ctx); fclose(f); return NULL; }
    read_bytes = fread(ctx->base, 1, (size_t)file_size, f);
    fclose(f);
    if ((u64)read_bytes != file_size) { printf("Error: incomplete read\n"); free(ctx->base); free(ctx); return NULL; }

loaded:
    p = ctx->base;
    if (memcmp(p, "GGUF", 4) != 0) { printf("Error: not a GGUF file\n"); free(ctx->base); free(ctx); return NULL; }
    p += 4;
    if (read_u32(&p) != 3) { printf("Warning: GGUF version != 3\n"); }
    tensor_count = read_u64(&p);
    meta_count = read_u64(&p);

    parse_metadata(&p, meta_count, &ctx->hp);

    /* Simple tensor section finder: search first 200KB for token embedding name */
    {
        static const char *names[] = {
            "output_norm.weight", "norm.weight", "model.norm.weight",
            "token_embd.weight", "tok_embeddings.weight", "model.embed_tokens.weight",
            "token_embeddings.weight", "embedding.weight", "embed.weight", NULL
        };
        u8 *search_end = ctx->base + ctx->size;
        u8 *s;
        u8 *found = NULL;
        for (s = ctx->base + 200; s < search_end - 64; s++) {
            int ni;
            for (ni = 0; names[ni]; ni++) {
                size_t nlen = strlen(names[ni]);
                if (s + 8 + nlen <= search_end &&
                    memcmp(s + 8, names[ni], nlen) == 0) {
                    /* check preceding 8 bytes: should be name_len == nlen */
                    u64 nl; memcpy(&nl, s, 8);
                    if (nl == nlen) {
                        found = s;
                        break;
                    }
                }
            }
            if (found) break;
        }
        if (found && found != p) {
            printf("Tensor section found at offset %u (parser was at %u)\n",
                   (unsigned int)(found - ctx->base), (unsigned int)(p - ctx->base));
            p = found;
        }
    }

    ctx->n_tensors = tensor_count;
    ctx->tensors = (Tensor*)malloc(sizeof(Tensor) * (size_t)tensor_count);
    if (!ctx->tensors) { printf("Error: cannot allocate tensor table (%u bytes)\n", (unsigned int)(sizeof(Tensor) * (size_t)tensor_count)); free(ctx->base); free(ctx); return NULL; }
    memset(ctx->tensors, 0, sizeof(Tensor) * (size_t)tensor_count);
    for (i = 0; i < tensor_count; i++) {
        Tensor *t = &ctx->tensors[i];
        u32 j;
        u64 name_len = read_u64(&p);
        if (name_len > 79) name_len = 79;
        memcpy(t->name, p, (size_t)name_len); t->name[name_len] = '\0'; p += name_len;
        t->n_dims = read_u32(&p);
        for (j = 0; j < 4; j++) t->dims[j] = (j < t->n_dims) ? read_u64(&p) : 1;
        t->type = read_u32(&p);
        t->offset = read_u64(&p);
    }

    pos_after_tensors = (u64)(p - ctx->base);
    ctx->data_offset = align_u64(pos_after_tensors, ctx->hp.alignment);
    for (i = 0; i < tensor_count; i++) {
        ctx->tensors[i].data = ctx->base + ctx->data_offset + ctx->tensors[i].offset;
    }
    return ctx;
}

static void gguf_free(GGUFContext *ctx) {
    if (!ctx) return;
    free(ctx->tensors);
#ifdef _WIN32
    if (ctx->hMapping) {
        UnmapViewOfFile(ctx->base);
        CloseHandle(ctx->hMapping);
        CloseHandle(ctx->hFile);
    } else {
        free(ctx->base);
    }
#else
    free(ctx->base);
#endif
    free(ctx);
}

static Tensor *find_tensor(GGUFContext *ctx, const char *name) {
    u64 i;
    for (i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) return &ctx->tensors[i];
    }
    return NULL;
}

static Tensor *find_tensor_f(GGUFContext *ctx, const char *fmt, int n) {
    char name[128];
    sprintf(name, fmt, n);
    return find_tensor(ctx, name);
}

static void list_tensors(GGUFContext *ctx) {
    u64 i;
    printf("Tensors in file (%u total):\n", (unsigned int)ctx->n_tensors);
    for (i = 0; i < ctx->n_tensors && i < 200; i++) {
        Tensor *t = &ctx->tensors[i];
        printf("  %-50s dims=[%u,%u,%u,%u] type=%u\n",
               t->name,
               (unsigned int)t->dims[0], (unsigned int)t->dims[1],
               (unsigned int)t->dims[2], (unsigned int)t->dims[3],
               (unsigned int)t->type);
    }
    if (ctx->n_tensors > 200) printf("  ... (%u more)\n", (unsigned int)(ctx->n_tensors - 200));
}

/* --- Math --- */

static float fast_rsqrt(float x) {
    /* Quake III fast inverse sqrt, 1 Newton iteration, ~0.2% error */
    union { float f; unsigned long i; } u;
    float xhalf = 0.5f * x;
    u.f = x;
    u.i = 0x5f375a86UL - (u.i >> 1);
    u.f = u.f * (1.5f - xhalf * u.f * u.f);
    return u.f;
}

static void rmsnorm(float *x, float *w, int n, float eps) {
    int i;
    float ss = 0.0f, scale, inv_n;
    /* Unrolled sum-of-squares */
    for (i = 0; i + 3 < n; i += 4) {
        ss += x[i]*x[i] + x[i+1]*x[i+1] + x[i+2]*x[i+2] + x[i+3]*x[i+3];
    }
    for (; i < n; i++) ss += x[i] * x[i];
    inv_n = 1.0f / (float)n;
    scale = fast_rsqrt(ss * inv_n + eps);
    for (i = 0; i + 3 < n; i += 4) {
        x[i]   = x[i]   * scale * w[i];
        x[i+1] = x[i+1] * scale * w[i+1];
        x[i+2] = x[i+2] * scale * w[i+2];
        x[i+3] = x[i+3] * scale * w[i+3];
    }
    for (; i < n; i++) x[i] = x[i] * scale * w[i];
}

static void rope_1d(float *vec, int n, int pos, const float *rope_cos, const float *rope_sin) {
    int half = n / 2;
    int k;
    const float *rc = rope_cos + pos * half;
    const float *rs = rope_sin + pos * half;
    for (k = 0; k < half; k++) {
        float c = rc[k];
        float s = rs[k];
        float v0 = vec[k * 2];
        float v1 = vec[k * 2 + 1];
        vec[k * 2]     = v0 * c - v1 * s;
        vec[k * 2 + 1] = v0 * s + v1 * c;
    }
}

/* Fast exp via 2KB lookup table. Covers [-16, 0] with step 1/32, ~1% error. */
static float fast_exp_table(float x) {
    static float table[513];
    static int init = 0;
    int idx;
    float frac;
    if (!init) {
        int i;
        for (i = 0; i <= 512; i++) table[i] = (float)exp((double)(i - 512) / 32.0);
        init = 1;
    }
    {
        float fx = x * 32.0f;
        int idx_raw = (int)fx;
        if (fx < 0.0f && fx != (float)idx_raw) idx_raw--;
        idx = idx_raw + 512;
    }
    if (idx <= 0) return 0.0f;
    if (idx >= 512) return 1.0f; /* exp(0) = 1, and x >= 0 after subtracting max */
    frac = x * 32.0f + 512.0f - (float)idx;
    return table[idx] + frac * (table[idx + 1] - table[idx]);
}

static float fast_sigmoid(float x) {
    if (x >= 5.0f) return 1.0f;
    if (x <= -5.0f) return 0.0f;
    return 1.0f / (1.0f + fast_exp_table(-x));
}

static float silu(float x) {
    if (x >= 5.0f) return x;
    if (x <= -5.0f) return 0.0f;
    return x * fast_sigmoid(x);
}

/* llama.cpp-style top-k / top-p / min-p + temperature sampling */
#define MAX_TOPK 256
static int sample_topk(float *logits, int n, float temp, int top_k, float top_p, float min_p, int ban_token) {
    int i, k;
    float maxv, sum, r, cdf;
    if (ban_token >= 0 && ban_token < n) logits[ban_token] = -1e30f;
    if (temp == 0.0f) {
        int best = 0;
        for (i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    if ((top_k > 0 && top_k < n) || (top_p > 0.0f && top_p < 1.0f) || min_p > 0.0f) {
        int kk = n;
        int top_idx[MAX_TOPK];
        float top_val[MAX_TOPK];
        float top_prob[MAX_TOPK];
        float keep_threshold;
        if (top_k > 0 && top_k < n) kk = top_k;
        if (kk > MAX_TOPK) kk = MAX_TOPK;
        if (kk < 1) kk = 1;
        for (k = 0; k < kk; k++) {
            top_idx[k] = -1;
            top_val[k] = -1e30f;
        }
        for (i = 0; i < n; i++) {
            float v = logits[i];
            for (k = 0; k < kk; k++) {
                if (v > top_val[k]) {
                    int j;
                    for (j = kk - 1; j > k; j--) {
                        top_val[j] = top_val[j - 1];
                        top_idx[j] = top_idx[j - 1];
                    }
                    top_val[k] = v;
                    top_idx[k] = i;
                    break;
                }
            }
        }
        maxv = top_val[0] / temp;
        sum = 0.0f;
        for (k = 0; k < kk; k++) {
            top_prob[k] = fast_exp_table((top_val[k] / temp) - maxv);
            sum += top_prob[k];
        }
        if (sum <= 0.0f) return top_idx[0] >= 0 ? top_idx[0] : 0;
        keep_threshold = 0.0f;
        if (min_p > 0.0f) keep_threshold = top_prob[0] * min_p;
        cdf = 0.0f;
        for (k = 0; k < kk; k++) {
            if (top_prob[k] >= keep_threshold) cdf += top_prob[k];
        }
        if (cdf <= 0.0f) return top_idx[0] >= 0 ? top_idx[0] : 0;
        if (top_p > 0.0f && top_p < 1.0f) {
            float target = cdf * top_p;
            float kept_cdf = 0.0f;
            for (k = 0; k < kk; k++) {
                if (top_prob[k] < keep_threshold) continue;
                kept_cdf += top_prob[k];
                if (kept_cdf >= target) {
                    float r2 = ((float)rand() / (float)RAND_MAX) * kept_cdf;
                    float cdf2 = 0.0f;
                    int j;
                    for (j = 0; j <= k; j++) {
                        if (top_prob[j] < keep_threshold) continue;
                        cdf2 += top_prob[j];
                        if (r2 < cdf2) return top_idx[j];
                    }
                    return top_idx[k];
                }
            }
            return top_idx[0] >= 0 ? top_idx[0] : 0;
        }
        r = ((float)rand() / (float)RAND_MAX) * cdf;
        cdf = 0.0f;
        for (k = 0; k < kk; k++) {
            if (top_prob[k] < keep_threshold) continue;
            cdf += top_prob[k];
            if (r < cdf) return top_idx[k];
        }
        for (k = kk - 1; k >= 0; k--) {
            if (top_prob[k] >= keep_threshold) return top_idx[k];
        }
        return top_idx[0] >= 0 ? top_idx[0] : 0;
    }
    /* Temperature scaling + softmax */
    maxv = logits[0] / temp;
    for (i = 1; i < n; i++) {
        float v = logits[i] / temp;
        if (v > maxv) maxv = v;
    }
    sum = 0.0f;
    for (i = 0; i < n; i++) {
        if (logits[i] > -1e20f) {
            logits[i] = fast_exp_table((logits[i] / temp) - maxv);
            sum += logits[i];
        } else {
            logits[i] = 0.0f;
        }
    }
    if (sum <= 0.0f) {
        int best = 0;
        for (i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    /* Sample from distribution */
    r = ((float)rand() / (float)RAND_MAX) * sum;
    cdf = 0.0f;
    for (i = 0; i < n; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return n - 1;
}

/* --- Quantized matmul --- */

static void matvec_f32(const float *w, const float *x, float *y, int n, int d) {
    int i, j;
    for (i = 0; i < n; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum;
        const float *row = w + i * d;
        for (j = 0; j + 3 < d; j += 4) {
            sum0 += x[j]   * row[j];
            sum1 += x[j+1] * row[j+1];
            sum2 += x[j+2] * row[j+2];
            sum3 += x[j+3] * row[j+3];
        }
        sum = sum0 + sum1 + sum2 + sum3;
        for (; j < d; j++) sum += x[j] * row[j];
        y[i] = sum;
    }
}

static float fp16_to_fp32(u16 h);
static void dequantize_row(const Tensor *t, float *out, int row, int d);
static void matvec(const Tensor *t, const float *x, float *y, int n, int d, float *dq_row);

static int blocks_256(int d) {
    return (d + 255) / 256;
}

static int cache_tensor_f32_inplace(Tensor *t) {
    int n, d, i;
    float *buf;
    float *row;
    size_t bytes;

    if (!t || t->type == 0) return 1;
    n = (int)t->dims[0];
    d = (int)t->dims[1];
    if (n <= 0 || d <= 0) return 0;
    bytes = (size_t)n * (size_t)d * sizeof(float);
    buf = (float*)malloc(bytes);
    row = (float*)malloc((size_t)d * sizeof(float));
    if (!buf || !row) {
        free(buf);
        free(row);
        return 0;
    }
    for (i = 0; i < n; i++) {
        dequantize_row(t, row, i, d);
        memcpy(buf + (size_t)i * (size_t)d, row, (size_t)d * sizeof(float));
    }
    free(row);
    t->data = buf;
    t->type = 0;
    return 1;
}

static float *cache_transposed_tensor_f32(const Tensor *t, int rows, int cols) {
    int i, j;
    float *buf;
    float *row;
    size_t bytes;

    if (!t || rows <= 0 || cols <= 0) return NULL;
    bytes = (size_t)rows * (size_t)cols * sizeof(float);
    buf = (float*)malloc(bytes);
    row = (float*)malloc((size_t)cols * sizeof(float));
    if (!buf || !row) {
        free(buf);
        free(row);
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            buf[(size_t)j * (size_t)rows + (size_t)i] = 0.0f;
        }
    }
    for (i = 0; i < rows; i++) {
        dequantize_row(t, row, i, cols);
        for (j = 0; j < cols; j++) {
            buf[(size_t)j * (size_t)rows + (size_t)i] = row[j];
        }
    }
    free(row);
    return buf;
}

static void matvec_q4(const BlockQ4 *w, const float *x, float *y, int n, int d) {
    int nb = d / 32;
    int i, b, j;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ4 *blk = &w[i * nb + b];
            float ds = fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            const u8 *qs = blk->qs;
            for (j = 0; j + 3 < 16; j += 4) {
                u8 qv0 = qs[j];
                u8 qv1 = qs[j+1];
                u8 qv2 = qs[j+2];
                u8 qv3 = qs[j+3];
                sum += xb[j*2+0] * ds * ((float)(qv0 & 0x0F) - 8.0f)
                     + xb[j*2+1] * ds * ((float)(qv0 >> 4)   - 8.0f)
                     + xb[j*2+2] * ds * ((float)(qv1 & 0x0F) - 8.0f)
                     + xb[j*2+3] * ds * ((float)(qv1 >> 4)   - 8.0f)
                     + xb[j*2+4] * ds * ((float)(qv2 & 0x0F) - 8.0f)
                     + xb[j*2+5] * ds * ((float)(qv2 >> 4)   - 8.0f)
                     + xb[j*2+6] * ds * ((float)(qv3 & 0x0F) - 8.0f)
                     + xb[j*2+7] * ds * ((float)(qv3 >> 4)   - 8.0f);
            }
            for (; j < 16; j++) {
                u8 qv = qs[j];
                sum += xb[j*2+0] * ds * ((float)(qv & 0x0F) - 8.0f)
                     + xb[j*2+1] * ds * ((float)(qv >> 4)   - 8.0f);
            }
        }
        y[i] = sum;
    }
}

static void matvec_q8(const BlockQ8 *w, const float *x, float *y, int n, int d) {
    int nb = d / 32;
    int i, b, j;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ8 *blk = &w[i * nb + b];
            float ds = fp16_to_fp32(blk->d);
            const float *xb = x + b * 32;
            const s8 *qs = blk->qs;
            for (j = 0; j + 3 < 32; j += 4) {
                sum += xb[j]   * ds * (float)qs[j]
                     + xb[j+1] * ds * (float)qs[j+1]
                     + xb[j+2] * ds * (float)qs[j+2]
                     + xb[j+3] * ds * (float)qs[j+3];
            }
            for (; j < 32; j++) {
                sum += xb[j] * ds * (float)qs[j];
            }
        }
        y[i] = sum;
    }
}

static float fp16_to_fp32(u16 h) {
    u32 sign = (h >> 15) & 0x1;
    u32 exp = (h >> 10) & 0x1F;
    u32 mant = h & 0x3FF;
    u32 f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    { float fv; memcpy(&fv, &f, 4); return fv; }
}

#pragma pack(push, 1)
typedef struct { u16 d[2]; u8 scales[12]; u8 qs[128]; } BlockQ4K;
typedef struct { u16 d[2]; u8 scales[12]; u8 qh[32]; u8 qs[128]; } BlockQ5K;
typedef struct { u8 ql[128]; u8 qh[64]; s8 scales[16]; u16 d; } BlockQ6K;
#pragma pack(pop)

static void get_scale_min_k4(int j, const u8 *q, u8 *d, u8 *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void matvec_f16(const u16 *w, const float *x, float *y, int n, int d) {
    int i, j;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (j = 0; j < d; j++) sum += x[j] * fp16_to_fp32(w[i * d + j]);
        y[i] = sum;
    }
}

static void matvec_q5_0(const BlockQ5_0 *w, const float *x, float *y, int n, int d) {
    int nb = d / 32;
    int i, b, j;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ5_0 *blk = &w[i * nb + b];
            float ds = fp16_to_fp32(blk->d);
            u32 qh;
            const float *xb = x + b * 32;
            memcpy(&qh, blk->qh, sizeof(qh));
            for (j = 0; j < 16; j++) {
                u8 xh_0 = (u8)(((qh >> (j + 0)) << 4) & 0x10);
                u8 xh_1 = (u8)(((qh >> (j + 16))     ) & 0x10);
                int x0 = ((blk->qs[j] & 0x0F) | xh_0) - 16;
                int x1 = ((blk->qs[j] >>   4) | xh_1) - 16;
                sum += xb[j * 2 + 0] * (float)x0 * ds
                     + xb[j * 2 + 1] * (float)x1 * ds;
            }
        }
        y[i] = sum;
    }
}

static void matvec_q5_1(const BlockQ5_1 *w, const float *x, float *y, int n, int d) {
    int nb = d / 32;
    int i, b, j;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ5_1 *blk = &w[i * nb + b];
            float ds = fp16_to_fp32(blk->d);
            float mn = fp16_to_fp32(blk->m);
            u32 qh;
            const float *xb = x + b * 32;
            memcpy(&qh, blk->qh, sizeof(qh));
            for (j = 0; j < 16; j++) {
                u8 xh_0 = (u8)(((qh >> (j + 0)) << 4) & 0x10);
                u8 xh_1 = (u8)(((qh >> (j + 16))     ) & 0x10);
                int x0 = (blk->qs[j] & 0x0F) | xh_0;
                int x1 = (blk->qs[j] >>   4) | xh_1;
                sum += xb[j * 2 + 0] * ((float)x0 * ds + mn)
                     + xb[j * 2 + 1] * ((float)x1 * ds + mn);
            }
        }
        y[i] = sum;
    }
}

static void matvec_q4k(const BlockQ4K *w, const float *x, float *y, int n, int d) {
    int nb = blocks_256(d);
    int i, b, jj, l;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            int base = b * 256;
            int limit = d - base;
            const BlockQ4K *blk = &w[i * nb + b];
            float d_all = fp16_to_fp32(blk->d[0]);
            float min_all = fp16_to_fp32(blk->d[1]);
            const u8 *q = blk->qs;
            int is = 0;
            u8 sc, m;
            float d1, m1, d2, m2;
            const float *xb = x + base;
            if (limit > 256) limit = 256;
            if (limit <= 0) continue;
            for (jj = 0; jj < limit; jj += 64) {
                int rem = limit - jj;
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                if (rem > 64) rem = 64;
                for (l = 0; l < 32 && l < rem; l++) {
                    sum += (d1 * (q[l] & 0xF) - m1) * xb[jj + l];
                }
                if (rem > 32) {
                    int rem2 = rem - 32;
                    if (rem2 > 32) rem2 = 32;
                    for (l = 0; l < rem2; l++) {
                        sum += (d2 * (q[l] >> 4) - m2) * xb[jj + 32 + l];
                    }
                }
                q += 32;
                is += 2;
            }
        }
        y[i] = sum;
    }
}

static void matvec_q5k(const BlockQ5K *w, const float *x, float *y, int n, int d) {
    int nb = blocks_256(d);
    int i, b, jj, l;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            int base = b * 256;
            int limit = d - base;
            const BlockQ5K *blk = &w[i * nb + b];
            float d_all = fp16_to_fp32(blk->d[0]);
            float min_all = fp16_to_fp32(blk->d[1]);
            const u8 *ql = blk->qs;
            const u8 *qh = blk->qh;
            int is = 0;
            u8 sc, m;
            u8 u1 = 1, u2 = 2;
            float d1, m1, d2, m2;
            const float *xb = x + base;
            if (limit > 256) limit = 256;
            if (limit <= 0) continue;
            for (jj = 0; jj < limit; jj += 64) {
                int rem = limit - jj;
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                if (rem > 64) rem = 64;
                for (l = 0; l < 32 && l < rem; l++) {
                    int x0 = (ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0);
                    int x1 = (ql[l] >> 4) + (qh[l] & u2 ? 16 : 0);
                    sum += (d1 * (float)x0 - m1) * xb[jj + l + 0];
                    if (l + 32 < rem) {
                        sum += (d2 * (float)x1 - m2) * xb[jj + l + 32];
                    }
                }
                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        y[i] = sum;
    }
}

static void matvec_q6k(const BlockQ6K *w, const float *x, float *y, int n, int d) {
    int nb = blocks_256(d);
    int i, b, n2, l;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            int base = b * 256;
            int limit = d - base;
            const BlockQ6K *blk = &w[i * nb + b];
            float d_all = fp16_to_fp32(blk->d);
            const u8 *ql = blk->ql;
            const u8 *qh = blk->qh;
            const s8 *sc = blk->scales;
            const float *xb = x + base;
            if (limit > 256) limit = 256;
            if (limit <= 0) continue;
            for (n2 = 0; n2 < limit; n2 += 128) {
                int rem = limit - n2;
                if (rem > 128) rem = 128;
                for (l = 0; l < 32 && l < rem; l++) {
                    int is = l / 16;
                    int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    if (l + 0 < rem) sum += d_all * (float)sc[is + 0] * (float)q1 * xb[l + 0];
                    if (l + 32 < rem) sum += d_all * (float)sc[is + 2] * (float)q2 * xb[l + 32];
                    if (l + 64 < rem) sum += d_all * (float)sc[is + 4] * (float)q3 * xb[l + 64];
                    if (l + 96 < rem) sum += d_all * (float)sc[is + 6] * (float)q4 * xb[l + 96];
                }
                xb += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        y[i] = sum;
    }
}

/* --- Multi-threading helpers (WinNT only: Win2K/XP) --- */

static int is_winnt(void) {
#ifdef _WIN32
    return (GetVersion() & 0x80000000) == 0;
#else
    return 0;
#endif
}

static int default_thread_count(void) {
#ifdef _WIN32
    DWORD_PTR processMask = 0, systemMask = 0;
    int n = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
        while (processMask) {
            n += (int)(processMask & 1);
            processMask >>= 1;
        }
        if (n > 0) {
            if (n > 8) n = 8;
            return n;
        }
    }
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        if (si.dwNumberOfProcessors > 0) {
            n = (int)si.dwNumberOfProcessors;
            if (n > 8) n = 8;
            return n;
        }
    }
#endif
    return 1;
}

static int blocks_256(int d);

static size_t tensor_row_bytes(const Tensor *t, int d) {
    switch (t->type) {
        case 0: return (size_t)d * sizeof(float);
        case 1: return (size_t)d * sizeof(u16);
        case 2: case 3: return (size_t)(d / 32) * sizeof(BlockQ4);
        case 6: return (size_t)(d / 32) * sizeof(BlockQ5_0);
        case 7: return (size_t)(d / 32) * sizeof(BlockQ5_1);
        case 8: return (size_t)(d / 32) * sizeof(BlockQ8);
        case 12: return (size_t)blocks_256(d) * sizeof(BlockQ4K);
        case 13: return (size_t)blocks_256(d) * sizeof(BlockQ5K);
        case 14: return (size_t)blocks_256(d) * sizeof(BlockQ6K);
        default: return 0;
    }
}

static int should_parallelize_matvec(int n, int d) {
    u64 work;
    if (g_n_threads < 2 || !is_winnt()) return 0;
    if (n < 96) return 0;
    work = (u64)n * (u64)d;
    if (work < 80000ULL) return 0;
    if (n < 512) return work >= 100000ULL;
    if (n < 2048) return work >= 80000ULL;
    return 1;
}

static int pick_matvec_threads(int n, int d) {
    u64 work = (u64)n * (u64)d;
    int thread_count = g_n_threads;
    if (thread_count > 8) thread_count = 8;
    if (thread_count > n) thread_count = n;
    /* Keep tiny matrices serial. Use 4 threads for mid-sized work and
       allow 8 threads once the matrix is big enough to amortize the
       extra synchronization. */
    if (work < 600000ULL && thread_count > 4) thread_count = 4;
    if (work < 250000ULL) thread_count = 1;
    if (thread_count < 1) thread_count = 1;
    return thread_count;
}

#ifdef _WIN32
static void matvec_no_parallel(const Tensor *t, const float *x, float *y, int n, int d, float *dq_row);

typedef struct {
    Tensor sub_t;
    const float *x;
    float *y;
    int n;
    int d;
} MatvecThreadWork;

#define MATVEC_POOL_MAX_THREADS 8

typedef struct {
    int index;
} MatvecPoolWorker;

typedef struct {
    const Tensor *t;
    const float *x;
    float *y;
    int n;
    int d;
    size_t row_bytes;
    int thread_count;
} MatvecPoolJob;

typedef struct {
    int initialized;
    int thread_count;
    HANDLE threads[MATVEC_POOL_MAX_THREADS];
    HANDLE start_events[MATVEC_POOL_MAX_THREADS];
    HANDLE done_events[MATVEC_POOL_MAX_THREADS];
    MatvecPoolWorker workers[MATVEC_POOL_MAX_THREADS];
    MatvecPoolJob job;
} MatvecPoolState;

static MatvecPoolState g_matvec_pool;

static unsigned __stdcall matvec_thread_func(void *arg) {
    MatvecThreadWork *w = (MatvecThreadWork*)arg;
    matvec_no_parallel(&w->sub_t, w->x, w->y, w->n, w->d, NULL);
    return 0;
}

static unsigned __stdcall matvec_pool_thread(void *arg) {
    MatvecPoolWorker *w = (MatvecPoolWorker*)arg;
    int idx = w->index;
    for (;;) {
        WaitForSingleObject(g_matvec_pool.start_events[idx], INFINITE);
        if (g_matvec_pool.job.thread_count <= 0) break;
        if (idx < g_matvec_pool.job.thread_count) {
            MatvecPoolJob job = g_matvec_pool.job;
            int start = (int)(((u64)job.n * (u64)idx) / (u64)job.thread_count);
            int end = (int)(((u64)job.n * (u64)(idx + 1)) / (u64)job.thread_count);
            int rows = end - start;
            if (rows > 0) {
                Tensor sub_t = *job.t;
                sub_t.data = (u8*)job.t->data + (size_t)start * job.row_bytes;
                sub_t.dims[0] = (u64)rows;
                matvec_no_parallel(&sub_t, job.x, job.y + start, rows, job.d, NULL);
            }
        }
        SetEvent(g_matvec_pool.done_events[idx]);
    }
    return 0;
}

static int init_matvec_pool(void) {
    int i;
    if (g_matvec_pool.initialized) return 1;
    memset(&g_matvec_pool, 0, sizeof(g_matvec_pool));
    g_matvec_pool.thread_count = MATVEC_POOL_MAX_THREADS;
    for (i = 0; i < MATVEC_POOL_MAX_THREADS; i++) {
        g_matvec_pool.workers[i].index = i;
        g_matvec_pool.start_events[i] = CreateEventA(NULL, FALSE, FALSE, NULL);
        g_matvec_pool.done_events[i] = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!g_matvec_pool.start_events[i] || !g_matvec_pool.done_events[i]) return 0;
        g_matvec_pool.threads[i] = (HANDLE)_beginthreadex(NULL, 0, matvec_pool_thread, &g_matvec_pool.workers[i], 0, NULL);
        if (!g_matvec_pool.threads[i]) return 0;
    }
    g_matvec_pool.initialized = 1;
    return 1;
}

static void run_matvec_pool(const Tensor *t, const float *x, float *y, int n, int d, size_t row_bytes, int thread_count) {
    int i;
    if (!g_matvec_pool.initialized && !init_matvec_pool()) {
        matvec_no_parallel(t, x, y, n, d, NULL);
        return;
    }
    g_matvec_pool.job.t = t;
    g_matvec_pool.job.x = x;
    g_matvec_pool.job.y = y;
    g_matvec_pool.job.n = n;
    g_matvec_pool.job.d = d;
    g_matvec_pool.job.row_bytes = row_bytes;
    g_matvec_pool.job.thread_count = thread_count;
    for (i = 0; i < MATVEC_POOL_MAX_THREADS; i++) {
        ResetEvent(g_matvec_pool.done_events[i]);
    }
    for (i = 0; i < thread_count; i++) {
        SetEvent(g_matvec_pool.start_events[i]);
    }
    WaitForMultipleObjects(thread_count, g_matvec_pool.done_events, TRUE, INFINITE);
}

static int file_exists(const char *path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
#endif
}

static void dirname_from_path(const char *path, char *dir, size_t dir_size) {
    size_t len;
    const char *last_slash = strrchr(path, '\\');
    const char *last_fwd = strrchr(path, '/');
    const char *cut = last_slash;
    if (!cut || (last_fwd && last_fwd > cut)) cut = last_fwd;
    if (!cut) {
        if (dir_size > 0) dir[0] = '\0';
        return;
    }
    len = (size_t)(cut - path + 1);
    if (len >= dir_size) len = dir_size - 1;
    memcpy(dir, path, len);
    dir[len] = '\0';
}

static int discover_tokenizer_path(const char *model_path, char *out_path, size_t out_size) {
    char dir[512];
    char stem[512];
    const char *base;
    const char *dot;
    const char *candidates[4];
    int i;

    if (!model_path || !out_path || out_size == 0) return 0;
    dirname_from_path(model_path, dir, sizeof(dir));
    base = strrchr(model_path, '\\');
    if (!base) base = strrchr(model_path, '/');
    base = base ? base + 1 : model_path;
    dot = strrchr(base, '.');
    if (dot) {
        size_t stem_len = (size_t)(dot - base);
        if (stem_len >= sizeof(stem)) stem_len = sizeof(stem) - 1;
        memcpy(stem, base, stem_len);
        stem[stem_len] = '\0';
    } else {
        strncpy(stem, base, sizeof(stem) - 1);
        stem[sizeof(stem) - 1] = '\0';
    }

    candidates[0] = "tokenizer.bin";
    candidates[1] = "tokenizer.ggml.bin";
    candidates[2] = NULL;
    candidates[3] = NULL;

    for (i = 0; candidates[i]; i++) {
        char path[768];
        sprintf(path, "%s%s", dir, candidates[i]);
        if (file_exists(path)) {
            strncpy(out_path, path, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 1;
        }
    }

    {
        char path[768];
        sprintf(path, "%s%s.tokenizer.bin", dir, stem);
        if (file_exists(path)) {
            strncpy(out_path, path, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 1;
        }
        sprintf(path, "%s%s.bin", dir, stem);
        if (file_exists(path)) {
            strncpy(out_path, path, out_size - 1);
            out_path[out_size - 1] = '\0';
            return 1;
        }
    }

    return 0;
}

typedef struct {
    const Tensor *t;
    const float *x;
    float *partial;
    int start_d;
    int end_d;
    int vocab_size;
} TransposedOutputThreadWork;

static unsigned __stdcall transposed_output_thread(void *arg) {
    TransposedOutputThreadWork *w = (TransposedOutputThreadWork*)arg;
    int d2, v;
    float *dq_row = (float*)malloc(w->vocab_size * sizeof(float));
    if (!dq_row) return 1;
    memset(w->partial, 0, w->vocab_size * sizeof(float));
    for (d2 = w->start_d; d2 < w->end_d; d2++) {
        dequantize_row(w->t, dq_row, d2, w->vocab_size);
        for (v = 0; v < w->vocab_size; v++) {
            w->partial[v] += w->x[d2] * dq_row[v];
        }
    }
    free(dq_row);
    return 0;
}

typedef struct {
    const Tensor *t;
    float *x;
    int token;
    int start_d;
    int end_d;
    int vocab_size;
} TransposedEmbedThreadWork;

static unsigned __stdcall transposed_embed_thread(void *arg) {
    TransposedEmbedThreadWork *w = (TransposedEmbedThreadWork*)arg;
    int d2;
    float *dq_row = (float*)malloc((size_t)w->vocab_size * sizeof(float));
    if (!dq_row) return 1;
    for (d2 = w->start_d; d2 < w->end_d; d2++) {
        dequantize_row(w->t, dq_row, d2, w->vocab_size);
        w->x[d2] = dq_row[w->token];
    }
    free(dq_row);
    return 0;
}

static void matvec_transposed_parallel(const Tensor *t, const float *x, float *y, int n, int d) {
    static int announced = 0;
    int thread_count = g_n_threads;
    int i;
    int failed = 0;
    float *partials[16];
    HANDLE threads[16];
    DWORD tids[16];
    TransposedOutputThreadWork work[16];

    if (thread_count > 8) thread_count = 8;
    if (thread_count > d) thread_count = d;
    if (thread_count < 2) {
        float *dq_row = (float*)malloc((size_t)n * sizeof(float));
        int j, v;
        if (!dq_row) {
            memset(y, 0, (size_t)n * sizeof(float));
            return;
        }
        memset(y, 0, (size_t)n * sizeof(float));
        for (j = 0; j < d; j++) {
            dequantize_row(t, dq_row, j, n);
            for (v = 0; v < n; v++) y[v] += x[j] * dq_row[v];
        }
        free(dq_row);
        return;
    }

    if (!announced && !g_clean_output) {
        printf("Transposed parallel: %s n=%d d=%d threads=%d\n", t->name, n, d, thread_count);
        fflush(stdout);
        announced = 1;
    }

    memset(partials, 0, sizeof(partials));
    memset(threads, 0, sizeof(threads));
    memset(work, 0, sizeof(work));
    memset(tids, 0, sizeof(tids));
    for (i = 0; i < thread_count; i++) {
        int start = (int)(((u64)d * (u64)i) / (u64)thread_count);
        int end = (int)(((u64)d * (u64)(i + 1)) / (u64)thread_count);
        threads[i] = NULL;
        partials[i] = (float*)calloc((size_t)n, sizeof(float));
        if (!partials[i]) {
            failed = 1;
            break;
        }
        work[i].t = t;
        work[i].x = x;
        work[i].partial = partials[i];
        work[i].start_d = start;
        work[i].end_d = end;
        work[i].vocab_size = n;
        threads[i] = (HANDLE)_beginthreadex(NULL, 0, transposed_output_thread, &work[i], 0, &tids[i]);
        if (!threads[i]) {
            failed = 1;
            break;
        }
    }

    if (!failed) {
        int v;
        memset(y, 0, (size_t)n * sizeof(float));
        for (i = 0; i < thread_count; i++) {
            WaitForSingleObject(threads[i], INFINITE);
            CloseHandle(threads[i]);
            for (v = 0; v < n; v++) y[v] += partials[i][v];
            free(partials[i]);
        }
        return;
    }

    for (i = 0; i < thread_count; i++) {
        if (threads[i]) {
            WaitForSingleObject(threads[i], INFINITE);
            CloseHandle(threads[i]);
        }
        if (partials[i]) free(partials[i]);
    }
    {
        float *dq_row = (float*)malloc((size_t)n * sizeof(float));
        int j, v;
        if (!dq_row) {
            memset(y, 0, (size_t)n * sizeof(float));
            return;
        }
        memset(y, 0, (size_t)n * sizeof(float));
        for (j = 0; j < d; j++) {
            dequantize_row(t, dq_row, j, n);
            for (v = 0; v < n; v++) y[v] += x[j] * dq_row[v];
        }
        free(dq_row);
    }
}

#endif

static void matvec_parallel(const Tensor *t, const float *x, float *y, int n, int d) {
#ifdef _WIN32
    static int announced = 0;
    size_t row_bytes;
    int thread_count;

    if (g_n_threads < 2 || !is_winnt()) {
        matvec_no_parallel(t, x, y, n, d, NULL);
        return;
    }

    /* Only parallelize standard row-major non-transposed matrices */
    if (!((int)t->dims[0] == n && (int)t->dims[1] == d)) {
        matvec_no_parallel(t, x, y, n, d, NULL);
        return;
    }

    row_bytes = tensor_row_bytes(t, d);
    if (row_bytes == 0) {
        matvec_no_parallel(t, x, y, n, d, NULL);
        return;
    }

    thread_count = pick_matvec_threads(n, d);
    if (thread_count < 2) {
        matvec_no_parallel(t, x, y, n, d, NULL);
        return;
    }

    if (!announced && !g_clean_output) {
        printf("Row parallel: %s n=%d d=%d threads=%d\n", t->name, n, d, thread_count);
        fflush(stdout);
        announced = 1;
    }
    run_matvec_pool(t, x, y, n, d, row_bytes, thread_count);
#else
    matvec_no_parallel(t, x, y, n, d, NULL);
#endif
}

static void dequantize_row(const Tensor *t, float *out, int row, int d) {
    if (t->type == 0) {
        memcpy(out, (const float*)t->data + row * d, d * sizeof(float));
    } else if (t->type == 1) {
        const u16 *w = (const u16*)t->data + row * d;
        int j;
        for (j = 0; j < d; j++) out[j] = fp16_to_fp32(w[j]);
    } else if (t->type == 2) {
        int nb = d / 32;
        const BlockQ4 *blk = (const BlockQ4*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 32) {
            float ds = fp16_to_fp32(blk->d);
            int jj;
            for (jj = 0; jj < 16; jj++) {
                u8 qv = blk->qs[jj];
                out[j + jj * 2 + 0] = ds * ((float)(qv & 0x0F) - 8.0f);
                out[j + jj * 2 + 1] = ds * ((float)(qv >> 4) - 8.0f);
            }
            blk++;
        }
    } else if (t->type == 8) {
        int nb = d / 32;
        const BlockQ8 *blk = (const BlockQ8*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 32) {
            float ds = fp16_to_fp32(blk->d);
            int jj;
            for (jj = 0; jj < 32; jj++) out[j + jj] = ds * (float)blk->qs[jj];
            blk++;
        }
    } else if (t->type == 6) {
        int nb = d / 32;
        const BlockQ5_0 *blk = (const BlockQ5_0*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 32) {
            int jj;
            float ds = fp16_to_fp32(blk->d);
            u32 qh;
            memcpy(&qh, blk->qh, sizeof(qh));
            for (jj = 0; jj < 16; jj++) {
                u8 xh_0 = (u8)(((qh >> (jj + 0)) << 4) & 0x10);
                u8 xh_1 = (u8)(((qh >> (jj + 16))     ) & 0x10);
                int x0 = ((blk->qs[jj] & 0x0F) | xh_0) - 16;
                int x1 = ((blk->qs[jj] >>   4) | xh_1) - 16;
                out[j + jj + 0   ] = (float)x0 * ds;
                out[j + jj + 16] = (float)x1 * ds;
            }
            blk++;
        }
    } else if (t->type == 7) {
        int nb = d / 32;
        const BlockQ5_1 *blk = (const BlockQ5_1*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 32) {
            int jj;
            float ds = fp16_to_fp32(blk->d);
            float mn = fp16_to_fp32(blk->m);
            u32 qh;
            memcpy(&qh, blk->qh, sizeof(qh));
            for (jj = 0; jj < 16; jj++) {
                u8 xh_0 = (u8)(((qh >> (jj + 0)) << 4) & 0x10);
                u8 xh_1 = (u8)(((qh >> (jj + 16))     ) & 0x10);
                int x0 = (blk->qs[jj] & 0x0F) | xh_0;
                int x1 = (blk->qs[jj] >>   4) | xh_1;
                out[j + jj + 0   ] = (float)x0 * ds + mn;
                out[j + jj + 16] = (float)x1 * ds + mn;
            }
            blk++;
        }
    } else if (t->type == 12) {
        int nb = blocks_256(d);
        const BlockQ4K *blk = (const BlockQ4K*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 256) {
            int limit = d - j;
            if (limit > 256) limit = 256;
            float d_all = fp16_to_fp32(blk->d[0]);
            float min_all = fp16_to_fp32(blk->d[1]);
            const u8 *q = blk->qs;
            int is = 0;
            u8 sc, m;
            int jj;
            float d1, m1, d2, m2;
            int l;
            for (jj = 0; jj < limit; jj += 64) {
                int rem = limit - jj;
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                if (rem > 64) rem = 64;
                for (l = 0; l < 32 && l < rem; l++) {
                    out[j + jj + l] = d1 * (q[l] & 0xF) - m1;
                }
                if (rem > 32) {
                    int rem2 = rem - 32;
                    if (rem2 > 32) rem2 = 32;
                    for (l = 0; l < rem2; l++) {
                        out[j + jj + 32 + l] = d2 * (q[l] >> 4) - m2;
                    }
                }
                q += 32;
                is += 2;
            }
            blk++;
        }
    } else if (t->type == 13) {
        int nb = blocks_256(d);
        const BlockQ5K *blk = (const BlockQ5K*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 256) {
            int limit = d - j;
            if (limit > 256) limit = 256;
            float d_all = fp16_to_fp32(blk->d[0]);
            float min_all = fp16_to_fp32(blk->d[1]);
            const u8 *ql = blk->qs;
            const u8 *qh = blk->qh;
            int is = 0;
            u8 sc, m;
            u8 u1 = 1, u2 = 2;
            int jj;
            float d1, m1, d2, m2;
            int l;
            for (jj = 0; jj < limit; jj += 64) {
                int rem = limit - jj;
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                if (rem > 64) rem = 64;
                for (l = 0; l < 32 && l < rem; l++) {
                    int x0 = (ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0);
                    int x1 = (ql[l] >> 4) + (qh[l] & u2 ? 16 : 0);
                    out[j + jj + l + 0] = d1 * (float)x0 - m1;
                    if (l + 32 < rem) {
                        out[j + jj + l + 32] = d2 * (float)x1 - m2;
                    }
                }
                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
            blk++;
        }
    } else if (t->type == 14) {
        int nb = blocks_256(d);
        const BlockQ6K *blk = (const BlockQ6K*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 256) {
            int limit = d - j;
            if (limit > 256) limit = 256;
            const float d_all = fp16_to_fp32(blk->d);
            const u8 *ql = blk->ql;
            const u8 *qh = blk->qh;
            const s8 *sc = blk->scales;
            float *y = out + j;
            int n2;
            for (n2 = 0; n2 < limit; n2 += 128) {
                int rem = limit - n2;
                int l;
                if (rem > 128) rem = 128;
                for (l = 0; l < 32 && l < rem; l++) {
                    int is = l / 16;
                    int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    if (l + 0 < rem)  y[l + 0]  = d_all * (float)sc[is + 0] * (float)q1;
                    if (l + 32 < rem) y[l + 32] = d_all * (float)sc[is + 2] * (float)q2;
                    if (l + 64 < rem) y[l + 64] = d_all * (float)sc[is + 4] * (float)q3;
                    if (l + 96 < rem) y[l + 96] = d_all * (float)sc[is + 6] * (float)q4;
                }
                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
            blk++;
        }
    } else {
        memset(out, 0, d * sizeof(float));
    }
}

static int effective_seq_limit(int requested, int n_layers, int n_kv_heads, int head_dim) {
    /* Cap sequence length to keep KV cache under ~256MB total (k+v) */
    unsigned int max_kv_bytes = 64U * 1024U * 1024U;
    unsigned int bytes_per_pos = (unsigned int)((u64)n_layers * (u64)n_kv_heads * (u64)head_dim * 2U * 4U);
    int limit;
    if (bytes_per_pos == 0) return requested;
    limit = (int)(max_kv_bytes / bytes_per_pos);
    if (limit < 128) limit = 128;
    if (limit > requested) limit = requested;
    return limit;
}

static void matvec_transposed_parallel(const Tensor *t, const float *x, float *y, int n, int d);

static void matvec_no_parallel(const Tensor *t, const float *x, float *y, int n, int d, float *dq_row) {
    int i, j;
    /* Transposed tensor fallback: [d, n] instead of [n, d].
       CRITICAL: when n==d (square matrices), we cannot tell orientation.
       GGUF convention is [n, d], so assume standard layout for squares.
       Only use fallback when dims are clearly swapped and non-square. */
    if ((int)t->dims[0] == d && (int)t->dims[1] == n && n != d) {
        for (i = 0; i < n; i++) {
            float sum = 0.0f;
            for (j = 0; j < d; j++) {
                dequantize_row(t, dq_row, j, n);
                sum += x[j] * dq_row[i];
            }
            y[i] = sum;
        }
        return;
    }
    if (t->type == 0) {
        matvec_f32((const float*)t->data, x, y, n, d);
    } else if (t->type == 1) {
        matvec_f16((const u16*)t->data, x, y, n, d);
    } else if (t->type == 2) {
        matvec_q4((const BlockQ4*)t->data, x, y, n, d);
    } else if (t->type == 6) {
        matvec_q5_0((const BlockQ5_0*)t->data, x, y, n, d);
    } else if (t->type == 7) {
        matvec_q5_1((const BlockQ5_1*)t->data, x, y, n, d);
    } else if (t->type == 8) {
        matvec_q8((const BlockQ8*)t->data, x, y, n, d);
    } else if (t->type == 12) {
        matvec_q4k((const BlockQ4K*)t->data, x, y, n, d);
    } else if (t->type == 13) {
        matvec_q5k((const BlockQ5K*)t->data, x, y, n, d);
    } else if (t->type == 14) {
        matvec_q6k((const BlockQ6K*)t->data, x, y, n, d);
    } else {
        fprintf(stderr, "Warning: unsupported type %u for %s, using zeros\n", t->type, t->name);
        memset(y, 0, n * sizeof(float));
    }
}

static void matvec(const Tensor *t, const float *x, float *y, int n, int d, float *dq_row) {
    if (should_parallelize_matvec(n, d) &&
        (int)t->dims[0] == n && (int)t->dims[1] == d) {
        matvec_parallel(t, x, y, n, d);
        return;
    }
    matvec_no_parallel(t, x, y, n, d, dq_row);
}

/* --- Model builder --- */

static int build_model(GGUFContext *ctx, Model *m, HParams *hp, int n_heads_user, int n_kv_user, int hidden_user, int seq_user) {
    int i;
    Tensor *t;

    t = find_tensor(ctx, "token_embd.weight");
    if (!t) t = find_tensor(ctx, "tok_embeddings.weight");
    if (!t) t = find_tensor(ctx, "model.embed_tokens.weight");
    if (!t) t = find_tensor(ctx, "embed_tokens.weight");
    if (!t) t = find_tensor(ctx, "token_embeddings.weight");
    if (!t) t = find_tensor(ctx, "embedding.weight");
    if (!t) t = find_tensor(ctx, "embed.weight");
    if (!t) t = find_tensor(ctx, "transformer.wte.weight");
    if (!t) {
        fprintf(stderr, "Missing token embeddings\n");
        list_tensors(ctx);
        return 1;
    }
    m->tok_embd = t;
    /* Auto-detect transposed embeddings: vocab is usually much larger than dim */
    if (t->dims[1] > t->dims[0]) {
        m->vocab_size = (int)t->dims[1];
        m->dim = (int)t->dims[0];
        m->tok_embd_transposed = 1;
        printf("Note: token_embd.weight is transposed ([%u,%u])\n", (unsigned int)t->dims[0], (unsigned int)t->dims[1]);
    } else {
        m->vocab_size = (int)t->dims[0];
        m->dim = (int)t->dims[1];
        m->tok_embd_transposed = 0;
    }

    t = find_tensor(ctx, "output_norm.weight");
    if (!t) t = find_tensor(ctx, "norm.weight");
    if (!t) t = find_tensor(ctx, "model.norm.weight");
    if (!t) t = find_tensor(ctx, "ln_f.weight");
    if (!t) t = find_tensor(ctx, "transformer.ln_f.weight");
    if (!t) {
        fprintf(stderr, "Missing output norm\n");
        list_tensors(ctx);
        return 1;
    }
    m->output_norm = t;

    t = find_tensor(ctx, "output.weight");
    if (!t) t = find_tensor(ctx, "lm_head.weight");
    if (!t) t = find_tensor(ctx, "model.lm_head.weight");
    if (!t) t = find_tensor(ctx, "transformer.wte.weight");
    if (t) m->output = t; else m->output = m->tok_embd;

    if (hp->block_count > 0) m->n_layers = (int)hp->block_count;
    else {
        for (i = 0; i < (int)ctx->n_tensors; i++) {
            int l = -1;
            if (strncmp(ctx->tensors[i].name, "blk.", 4) == 0) l = atoi(ctx->tensors[i].name + 4);
            else if (strncmp(ctx->tensors[i].name, "model.layers.", 13) == 0) l = atoi(ctx->tensors[i].name + 13);
            if (l >= m->n_layers) m->n_layers = l + 1;
        }
    }
    if (m->n_layers == 0) { fprintf(stderr, "Cannot determine layer count\n"); return 1; }

    if (hp->attention_head_count > 0) m->n_heads = (int)hp->attention_head_count;
    else if (n_heads_user > 0) m->n_heads = n_heads_user;
    else { fprintf(stderr, "Need --n_heads (not in GGUF metadata)\n"); return 1; }
    m->head_dim = m->dim / m->n_heads;
    m->rsqrt_head_dim = 1.0f / (float)sqrt((double)m->head_dim);

    if (hp->attention_head_count_kv > 0) m->n_kv_heads = (int)hp->attention_head_count_kv;
    else if (n_kv_user > 0) m->n_kv_heads = n_kv_user;
    else {
        t = find_tensor_f(ctx, "blk.%d.attn_k.weight", 0);
        if (!t) t = find_tensor_f(ctx, "model.layers.%d.self_attn.k_proj.weight", 0);
        if (t && m->head_dim > 0) m->n_kv_heads = (int)(t->dims[0] / (u64)m->head_dim);
    }
    if (m->n_kv_heads == 0) m->n_kv_heads = m->n_heads;

    if (hidden_user > 0) m->hidden_dim = hidden_user;
    else if (hp->feed_forward_length > 0) m->hidden_dim = (int)hp->feed_forward_length;
    else {
        t = find_tensor_f(ctx, "blk.%d.ffn_gate.weight", 0);
        if (!t) t = find_tensor_f(ctx, "model.layers.%d.mlp.gate_proj.weight", 0);
        if (t) m->hidden_dim = (int)t->dims[0];
    }
    if (m->hidden_dim == 0) { fprintf(stderr, "Need --hidden (cannot determine)\n"); return 1; }

    if (m->tok_embd_transposed) {
        size_t cache_bytes = (size_t)m->vocab_size * (size_t)m->dim * sizeof(float);
        printf("Caching transposed token embeddings in F32 for speed...\n");
        m->cached_embd = cache_transposed_tensor_f32(m->tok_embd, m->dim, m->vocab_size);
        if (!m->cached_embd) {
            fprintf(stderr, "Failed to allocate embedding cache (%u MB)\n",
                    (unsigned int)(cache_bytes / (1024U * 1024U)));
            return 1;
        }
        if (m->output == m->tok_embd) {
            m->cached_output = m->cached_embd;
        }
        printf("Embedding cache ready.\n");
    }

    if (seq_user > 0) m->max_seq_len = seq_user;
    else if (hp->context_length > 0) m->max_seq_len = (int)hp->context_length;
    else m->max_seq_len = 2048;

    {
        int safe_seq = effective_seq_limit(m->max_seq_len, m->n_layers, m->n_kv_heads, m->head_dim);
        if (safe_seq < m->max_seq_len) {
            printf("WARNING: capping seq_len %d -> %d to keep KV cache under 256MB\n",
                   m->max_seq_len, safe_seq);
            m->max_seq_len = safe_seq;
        }
    }

    m->norm_eps = hp->attention_layer_norm_rms_epsilon;
    if (m->norm_eps == 0.0f) m->norm_eps = 1e-5f;
    m->rope_theta = hp->rope_freq_base;
    if (m->rope_theta == 0.0f) m->rope_theta = 10000.0f;

    {
        int half = m->head_dim / 2;
        int p, k;
        float *cos_tab = (float*)malloc((size_t)m->max_seq_len * half * sizeof(float));
        float *sin_tab = (float*)malloc((size_t)m->max_seq_len * half * sizeof(float));
        if (!cos_tab || !sin_tab) {
            fprintf(stderr, "Failed to allocate RoPE tables\n");
            return 1;
        }
        for (p = 0; p < m->max_seq_len; p++) {
            for (k = 0; k < half; k++) {
                float freq = 1.0f / (float)pow((double)m->rope_theta, (double)(k * 2) / (double)m->head_dim);
                float val = (float)p * freq;
                cos_tab[p * half + k] = (float)cos((double)val);
                sin_tab[p * half + k] = (float)sin((double)val);
            }
        }
        m->rope_cos = cos_tab;
        m->rope_sin = sin_tab;
    }

    m->attn_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_q = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_k = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_v = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_o = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_gate = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_up = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_down = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    if (!m->attn_norm || !m->attn_q || !m->attn_k || !m->attn_v || !m->attn_o ||
        !m->ffn_norm || !m->ffn_gate || !m->ffn_up || !m->ffn_down) {
        fprintf(stderr, "Failed to allocate layer tensor pointers (out of memory?)\n");
        return 1;
    }

    for (i = 0; i < m->n_layers; i++) {
        char name[128];
        m->attn_norm[i] = find_tensor_f(ctx, "blk.%d.attn_norm.weight", i);
        m->attn_q[i]    = find_tensor_f(ctx, "blk.%d.attn_q.weight", i);
        m->attn_k[i]    = find_tensor_f(ctx, "blk.%d.attn_k.weight", i);
        m->attn_v[i]    = find_tensor_f(ctx, "blk.%d.attn_v.weight", i);
        m->attn_o[i]    = find_tensor_f(ctx, "blk.%d.attn_output.weight", i);
        m->ffn_norm[i]  = find_tensor_f(ctx, "blk.%d.ffn_norm.weight", i);
        m->ffn_gate[i]  = find_tensor_f(ctx, "blk.%d.ffn_gate.weight", i);
        m->ffn_up[i]    = find_tensor_f(ctx, "blk.%d.ffn_up.weight", i);
        m->ffn_down[i]  = find_tensor_f(ctx, "blk.%d.ffn_down.weight", i);

        if (!m->attn_norm[i]) { sprintf(name, "model.layers.%d.input_layernorm.weight", i); m->attn_norm[i] = find_tensor(ctx, name); }
        if (!m->attn_q[i]) { sprintf(name, "model.layers.%d.self_attn.q_proj.weight", i); m->attn_q[i] = find_tensor(ctx, name); }
        if (!m->attn_k[i]) { sprintf(name, "model.layers.%d.self_attn.k_proj.weight", i); m->attn_k[i] = find_tensor(ctx, name); }
        if (!m->attn_v[i]) { sprintf(name, "model.layers.%d.self_attn.v_proj.weight", i); m->attn_v[i] = find_tensor(ctx, name); }
        if (!m->attn_o[i]) { sprintf(name, "model.layers.%d.self_attn.o_proj.weight", i); m->attn_o[i] = find_tensor(ctx, name); }
        if (!m->ffn_norm[i]) { sprintf(name, "model.layers.%d.post_attention_layernorm.weight", i); m->ffn_norm[i] = find_tensor(ctx, name); }
        if (!m->ffn_gate[i]) { sprintf(name, "model.layers.%d.mlp.gate_proj.weight", i); m->ffn_gate[i] = find_tensor(ctx, name); }
        if (!m->ffn_up[i]) { sprintf(name, "model.layers.%d.mlp.up_proj.weight", i); m->ffn_up[i] = find_tensor(ctx, name); }
        if (!m->ffn_down[i]) { sprintf(name, "model.layers.%d.mlp.down_proj.weight", i); m->ffn_down[i] = find_tensor(ctx, name); }

        if (!m->attn_q[i] || !m->attn_k[i] || !m->attn_v[i] || !m->attn_o[i] ||
            !m->ffn_gate[i] || !m->ffn_up[i] || !m->ffn_down[i]) {
            fprintf(stderr, "Missing tensors for layer %d\n", i);
            return 1;
        }
        if (m->attn_norm[i] && m->attn_norm[i]->type != 0)
            printf("Warning: layer %d attn_norm is type %u (expected F32)\n", i, m->attn_norm[i]->type);
        if (m->ffn_norm[i] && m->ffn_norm[i]->type != 0)
            printf("Warning: layer %d ffn_norm is type %u (expected F32)\n", i, m->ffn_norm[i]->type);
    }

    if (!m->cached_output && m->output && m->output != m->tok_embd &&
        m->output->dims[1] > m->output->dims[0] && m->output->type != 0) {
        size_t cache_bytes = (size_t)m->output->dims[0] * (size_t)m->output->dims[1] * sizeof(float);
        printf("Caching transposed output head in F32 for speed...\n");
        m->cached_output = cache_transposed_tensor_f32(m->output, (int)m->output->dims[0], (int)m->output->dims[1]);
        if (!m->cached_output) {
            fprintf(stderr, "Warning: failed to cache output head (%u MB)\n",
                    (unsigned int)(cache_bytes / (1024U * 1024U)));
        } else {
            printf("Output cache ready.\n");
        }
    }

    {
        int l;

        printf("Caching hot projection weights in F32 for speed...\n");
        for (l = 0; l < m->n_layers; l++) {
            if (m->attn_q[l] && m->attn_q[l]->type != 0) {
                if (!cache_tensor_f32_inplace(m->attn_q[l])) fprintf(stderr, "Warning: failed to cache %s\n", m->attn_q[l]->name);
            }
            if (m->attn_k[l] && m->attn_k[l]->type != 0) {
                if (!cache_tensor_f32_inplace(m->attn_k[l])) fprintf(stderr, "Warning: failed to cache %s\n", m->attn_k[l]->name);
            }
            if (m->attn_v[l] && m->attn_v[l]->type != 0) {
                if (!cache_tensor_f32_inplace(m->attn_v[l])) fprintf(stderr, "Warning: failed to cache %s\n", m->attn_v[l]->name);
            }
            if (m->attn_o[l] && m->attn_o[l]->type != 0) {
                if (!cache_tensor_f32_inplace(m->attn_o[l])) fprintf(stderr, "Warning: failed to cache %s\n", m->attn_o[l]->name);
            }
            if (m->ffn_gate[l] && m->ffn_gate[l]->type != 0) {
                if (!cache_tensor_f32_inplace(m->ffn_gate[l])) fprintf(stderr, "Warning: failed to cache %s\n", m->ffn_gate[l]->name);
            }
            if (m->ffn_up[l] && m->ffn_up[l]->type != 0) {
                if (!cache_tensor_f32_inplace(m->ffn_up[l])) fprintf(stderr, "Warning: failed to cache %s\n", m->ffn_up[l]->name);
            }
            if (m->ffn_down[l] && m->ffn_down[l]->type != 0) {
                if (!cache_tensor_f32_inplace(m->ffn_down[l])) fprintf(stderr, "Warning: failed to cache %s\n", m->ffn_down[l]->name);
            }
        }
        printf("Hot weights cache pass complete.\n");
    }

    /* Diagnostics: verify block sizes and tensor offsets */
    printf("Block sizes: Q4_0=%u Q5_0=%u Q5_1=%u Q8_0=%u Q4K=%u(exp144) Q5K=%u(exp176) Q6K=%u(exp210)\n",
           (unsigned int)sizeof(BlockQ4),
           (unsigned int)sizeof(BlockQ5_0),
           (unsigned int)sizeof(BlockQ5_1),
           (unsigned int)sizeof(BlockQ8),
           (unsigned int)sizeof(BlockQ4K),
           (unsigned int)sizeof(BlockQ5K),
           (unsigned int)sizeof(BlockQ6K));
    printf("tok_embd: dims=[%u,%u] type=%u transposed=%d\n",
           (unsigned int)m->tok_embd->dims[0],
           (unsigned int)m->tok_embd->dims[1],
           (unsigned int)m->tok_embd->type,
           m->tok_embd_transposed);
    for (i = 0; i < (int)ctx->n_tensors && i < 5; i++) {
        printf("Tensor[%d]: %s offset=%u\n", i,
               ctx->tensors[i].name,
               (unsigned int)(ctx->tensors[i].offset & 0xFFFFFFFFUL));
    }
    printf("data_offset=%u file_size=%u\n",
           (unsigned int)(ctx->data_offset & 0xFFFFFFFFUL),
           (unsigned int)(ctx->size & 0xFFFFFFFFUL));

    /* Architecture detection */
    m->arch = ARCH_LLAMA;
    if (strstr(hp->architecture, "qwen")) m->arch = ARCH_QWEN2;
    else if (strstr(hp->architecture, "gemma3")) m->arch = ARCH_GEMMA3;
    else if (strstr(hp->architecture, "gemma2")) m->arch = ARCH_GEMMA2;
    else if (strstr(hp->architecture, "gemma")) m->arch = ARCH_GEMMA;
    else if (strstr(hp->architecture, "gpt2")) m->arch = ARCH_GPT2;
    else if (strstr(hp->architecture, "phi4")) m->arch = ARCH_PHI4;
    else if (strstr(hp->architecture, "phi3")) m->arch = ARCH_PHI3;
    else if (strstr(hp->architecture, "phi")) m->arch = ARCH_PHI;
    else if (strstr(hp->architecture, "mistral")) m->arch = ARCH_MISTRAL;
    else if (strstr(hp->architecture, "mixtral")) m->arch = ARCH_MIXTRAL;
    else if (strstr(hp->architecture, "falcon")) m->arch = ARCH_FALCON;
    else if (strstr(hp->architecture, "deepseek")) m->arch = ARCH_DEEPSEEK;
    else if (strstr(hp->architecture, "granite")) m->arch = ARCH_GRANITE;
    else if (strstr(hp->architecture, "yi")) m->arch = ARCH_YI;
    else if (strstr(hp->architecture, "baichuan")) m->arch = ARCH_BAICHUAN;
    else if (strstr(hp->architecture, "stablelm")) m->arch = ARCH_STABLELM;
    else if (strstr(hp->architecture, "command-r")) m->arch = ARCH_COMMAND_R;
    else if (strstr(hp->architecture, "codellama")) m->arch = ARCH_CODELLAMA;
    else if (strstr(hp->architecture, "nemo")) m->arch = ARCH_NEMO;
    else if (strstr(hp->architecture, "llama4")) m->arch = ARCH_LLAMA4;
    else if (strstr(hp->architecture, "llama3")) m->arch = ARCH_LLAMA3;
    else if (strstr(hp->architecture, "llama2")) m->arch = ARCH_LLAMA2;

    printf("Model: dim=%d layers=%d heads=%d kv_heads=%d hidden=%d vocab=%d seq=%d arch=%s\n",
           m->dim, m->n_layers, m->n_heads, m->n_kv_heads, m->hidden_dim, m->vocab_size, m->max_seq_len,
           hp->architecture);
    return 0;
}

static void free_model(Model *m) {
    if (m->cached_output && m->cached_output != m->cached_embd) { free(m->cached_output); m->cached_output = NULL; }
    if (m->cached_embd) { free(m->cached_embd); m->cached_embd = NULL; }
    if (m->rope_cos)   { free(m->rope_cos);   m->rope_cos = NULL; }
    if (m->rope_sin)   { free(m->rope_sin);   m->rope_sin = NULL; }
    free(m->attn_norm); free(m->attn_q); free(m->attn_k); free(m->attn_v); free(m->attn_o);
    free(m->ffn_norm); free(m->ffn_gate); free(m->ffn_up); free(m->ffn_down);
}

static int alloc_runstate(Model *m, RunState *s, GGUFContext *ctx) {
    int dim = m->dim;
    int hidden_dim = m->hidden_dim;
    int vocab_size = m->vocab_size;
    int n_layers = m->n_layers;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int max_seq_len = m->max_seq_len;
    int kv_size = n_layers * n_kv_heads * max_seq_len * head_dim;
    int max_d = dim;
    if (hidden_dim > max_d) max_d = hidden_dim;
    if (vocab_size > max_d) max_d = vocab_size;

    printf("Allocating run state...\n");
    printf("  x/q/attn_out/tmp: %u bytes each\n", (unsigned int)(dim * sizeof(float)));
    printf("  k/v per head: %u bytes each\n", (unsigned int)(n_kv_heads * head_dim * sizeof(float)));
    printf("  ffn_gate/up/hidden: %u bytes each\n", (unsigned int)(hidden_dim * sizeof(float)));
    printf("  logits: %u bytes\n", (unsigned int)(vocab_size * sizeof(float)));
    printf("  k_cache+v_cache: %u bytes\n", (unsigned int)(kv_size * sizeof(float) * 2));
    printf("  dq_row: %u bytes\n", (unsigned int)(max_d * sizeof(float)));
    printf("  Total state: ~%u KB\n", (unsigned int)((dim*5 + n_kv_heads*head_dim*2 + hidden_dim*3 + vocab_size + kv_size*2 + max_seq_len + max_d) * sizeof(float) / 1024));

    s->x = (float*)malloc(dim * sizeof(float));
    s->q = (float*)malloc(dim * sizeof(float));
    s->k = (float*)malloc(n_kv_heads * head_dim * sizeof(float));
    s->v = (float*)malloc(n_kv_heads * head_dim * sizeof(float));
    s->attn_out = (float*)malloc(dim * sizeof(float));
    s->ffn_gate = (float*)malloc(hidden_dim * sizeof(float));
    s->ffn_up = (float*)malloc(hidden_dim * sizeof(float));
    s->ffn_hidden = (float*)malloc(hidden_dim * sizeof(float));
    s->logits = (float*)malloc(vocab_size * sizeof(float));
    s->tmp = (float*)malloc(dim * sizeof(float));
    s->k_cache = (float*)malloc(kv_size * sizeof(float));
    s->v_cache = (float*)malloc(kv_size * sizeof(float));
    s->attn_scores = (float*)malloc(max_seq_len * sizeof(float));
    s->dq_row = (float*)malloc(max_d * sizeof(float));

    if (!s->x) { fprintf(stderr, "Failed: x\n"); }
    if (!s->q) { fprintf(stderr, "Failed: q\n"); }
    if (!s->k) { fprintf(stderr, "Failed: k\n"); }
    if (!s->v) { fprintf(stderr, "Failed: v\n"); }
    if (!s->attn_out) { fprintf(stderr, "Failed: attn_out\n"); }
    if (!s->ffn_gate) { fprintf(stderr, "Failed: ffn_gate\n"); }
    if (!s->ffn_up) { fprintf(stderr, "Failed: ffn_up\n"); }
    if (!s->ffn_hidden) { fprintf(stderr, "Failed: ffn_hidden\n"); }
    if (!s->logits) { fprintf(stderr, "Failed: logits\n"); }
    if (!s->tmp) { fprintf(stderr, "Failed: tmp\n"); }
    if (!s->k_cache) { fprintf(stderr, "Failed: k_cache\n"); }
    if (!s->v_cache) { fprintf(stderr, "Failed: v_cache\n"); }
    if (!s->attn_scores) { fprintf(stderr, "Failed: attn_scores\n"); }
    if (!s->dq_row) { fprintf(stderr, "Failed: dq_row\n"); }

    if (!s->x || !s->q || !s->k || !s->v || !s->attn_out || !s->ffn_gate || !s->ffn_up ||
        !s->ffn_hidden || !s->logits || !s->tmp || !s->k_cache || !s->v_cache || !s->attn_scores || !s->dq_row) {
        fprintf(stderr, "Failed to allocate run state (out of memory?)\n");
        fprintf(stderr, "Model file alone is ~%u MB. You need at least that much + run state.\n",
                (unsigned int)(ctx->size / (1024*1024)));
        return 1;
    }
    memset(s->k_cache, 0, kv_size * sizeof(float));
    memset(s->v_cache, 0, kv_size * sizeof(float));
    return 0;
}

static void free_runstate(RunState *s) {
    free(s->x); free(s->q); free(s->k); free(s->v); free(s->attn_out);
    free(s->ffn_gate); free(s->ffn_up); free(s->ffn_hidden); free(s->logits); free(s->tmp);
    free(s->k_cache); free(s->v_cache); free(s->attn_scores); free(s->dq_row);
}



/* --- Forward pass --- */

static int forward(Model *m, RunState *s, int token, int pos, float temp, int top_k, float top_p, float min_p, int *recent, int recent_n, int ban_token) {
    int dim = m->dim;
    int hidden_dim = m->hidden_dim;
    int n_layers = m->n_layers;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int max_seq_len = m->max_seq_len;
    int vocab_size = m->vocab_size;
    int q_per_kv = n_heads / n_kv_heads;
    int l, h, h_kv, t, i;
    float *x = s->x;
    float *tmp = s->tmp;
    float *q = s->q;
    float *k = s->k;
    float *v = s->v;
    float *attn_out = s->attn_out;
    float *k_cache = s->k_cache;
    float *v_cache = s->v_cache;
    float *attn_scores = s->attn_scores;

    /* Token embedding row lookup */
    if (m->cached_embd) {
        memcpy(x, m->cached_embd + (size_t)token * (size_t)dim, (size_t)dim * sizeof(float));
    } else if (m->tok_embd_transposed) {
        int j;
        for (j = 0; j < dim; j++) {
            dequantize_row(m->tok_embd, s->dq_row, j, vocab_size);
            x[j] = s->dq_row[token];
        }
    } else {
        dequantize_row(m->tok_embd, x, token, dim);
    }

    for (l = 0; l < n_layers; l++) {
        /* Attention */
        for (i = 0; i < dim; i++) tmp[i] = x[i];
        rmsnorm(x, (float*)m->attn_norm[l]->data, dim, m->norm_eps);

        matvec(m->attn_q[l], x, q, dim, dim, s->dq_row);
        matvec(m->attn_k[l], x, k, n_kv_heads * head_dim, dim, s->dq_row);
        matvec(m->attn_v[l], x, v, n_kv_heads * head_dim, dim, s->dq_row);

        for (h = 0; h < n_heads; h++) rope_1d(q + h * head_dim, head_dim, pos, m->rope_cos, m->rope_sin);
        for (h_kv = 0; h_kv < n_kv_heads; h_kv++) rope_1d(k + h_kv * head_dim, head_dim, pos, m->rope_cos, m->rope_sin);

        for (h_kv = 0; h_kv < n_kv_heads; h_kv++) {
            float *kc = k_cache + ((l * n_kv_heads + h_kv) * max_seq_len + pos) * head_dim;
            float *vc = v_cache + ((l * n_kv_heads + h_kv) * max_seq_len + pos) * head_dim;
            float *kp = k + h_kv * head_dim;
            float *vp = v + h_kv * head_dim;
            for (i = 0; i + 3 < head_dim; i += 4) {
                kc[i]   = kp[i];   vc[i]   = vp[i];
                kc[i+1] = kp[i+1]; vc[i+1] = vp[i+1];
                kc[i+2] = kp[i+2]; vc[i+2] = vp[i+2];
                kc[i+3] = kp[i+3]; vc[i+3] = vp[i+3];
            }
            for (; i < head_dim; i++) { kc[i] = kp[i]; vc[i] = vp[i]; }
        }

        for (h = 0; h < n_heads; h++) {
            float max_score = -1e30f;
            float sum_score = 0.0f;
            float *q_head = q + h * head_dim;
            float *out_head = attn_out + h * head_dim;
            h_kv = h / q_per_kv;
            for (t = 0; t <= pos; t++) {
                float *kt = k_cache + ((l * n_kv_heads + h_kv) * max_seq_len + t) * head_dim;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, score;
                for (i = 0; i + 3 < head_dim; i += 4) {
                    s0 += q_head[i]   * kt[i];
                    s1 += q_head[i+1] * kt[i+1];
                    s2 += q_head[i+2] * kt[i+2];
                    s3 += q_head[i+3] * kt[i+3];
                }
                score = s0 + s1 + s2 + s3;
                for (; i < head_dim; i++) score += q_head[i] * kt[i];
                score *= m->rsqrt_head_dim;
                attn_scores[t] = score;
                if (score > max_score) max_score = score;
            }
            for (t = 0; t <= pos; t++) {
                attn_scores[t] = fast_exp_table(attn_scores[t] - max_score);
                sum_score += attn_scores[t];
            }
            for (t = 0; t <= pos; t++) attn_scores[t] /= sum_score;
            for (i = 0; i < head_dim; i++) out_head[i] = 0.0f;
            for (t = 0; t <= pos; t++) {
                float *vt = v_cache + ((l * n_kv_heads + h_kv) * max_seq_len + t) * head_dim;
                float score = attn_scores[t];
                for (i = 0; i + 3 < head_dim; i += 4) {
                    out_head[i]   += score * vt[i];
                    out_head[i+1] += score * vt[i+1];
                    out_head[i+2] += score * vt[i+2];
                    out_head[i+3] += score * vt[i+3];
                }
                for (; i < head_dim; i++) out_head[i] += score * vt[i];
            }
        }

        matvec(m->attn_o[l], attn_out, x, dim, dim, s->dq_row);
        for (i = 0; i < dim; i++) x[i] += tmp[i];

        /* FFN */
        for (i = 0; i < dim; i++) tmp[i] = x[i];
        rmsnorm(x, (float*)m->ffn_norm[l]->data, dim, m->norm_eps);

        matvec(m->ffn_gate[l], x, s->ffn_gate, hidden_dim, dim, s->dq_row);
        matvec(m->ffn_up[l], x, s->ffn_up, hidden_dim, dim, s->dq_row);
        for (i = 0; i < hidden_dim; i++) s->ffn_hidden[i] = silu(s->ffn_gate[i]) * s->ffn_up[i];

        matvec(m->ffn_down[l], s->ffn_hidden, x, dim, hidden_dim, s->dq_row);
        for (i = 0; i < dim; i++) x[i] += tmp[i];
    }

    rmsnorm(x, (float*)m->output_norm->data, dim, m->norm_eps);
    if (m->cached_output) {
        Tensor fake;
        fake = *m->output;
        fake.type = 0;
        fake.dims[0] = (u64)vocab_size;
        fake.dims[1] = (u64)dim;
        fake.data = m->cached_output;
        matvec(&fake, x, s->logits, vocab_size, dim, s->dq_row);
    } else if (m->cached_embd && m->output == m->tok_embd) {
        Tensor fake;
        fake = *m->output;
        fake.type = 0;
        fake.dims[0] = (u64)vocab_size;
        fake.dims[1] = (u64)dim;
        fake.data = m->cached_embd;
        matvec(&fake, x, s->logits, vocab_size, dim, s->dq_row);
    } else if (m->tok_embd_transposed && m->output == m->tok_embd) {
        int d2, v;
        memset(s->logits, 0, vocab_size * sizeof(float));
        for (d2 = 0; d2 < dim; d2++) {
            dequantize_row(m->output, s->dq_row, d2, vocab_size);
            for (v = 0; v < vocab_size; v++) {
                s->logits[v] += x[d2] * s->dq_row[v];
            }
        }
    } else {
        matvec(m->output, x, s->logits, vocab_size, dim, s->dq_row);
    }
    /* Repetition penalty: discourage recent tokens */
    if (recent && recent_n > 0) {
        int j;
        for (j = 0; j < recent_n && j < 32; j++) {
            int tok = recent[j];
            if (tok > 0 && tok < vocab_size) {
                if (s->logits[tok] > 0.0f) s->logits[tok] /= m->repeat_penalty;
                else s->logits[tok] *= m->repeat_penalty;
            }
        }
    }
    return sample_topk(s->logits, vocab_size, temp, top_k, top_p, min_p, ban_token);
}

/* --- Tokenizer --- */

static int *g_sorted_vocab = NULL;
static int g_sorted_init = 0;

static int load_tokenizer(const char *path) {
    FILE *f;
    size_t sz;
    u8 *buf;
    u32 count;
    int i;
    size_t pos;
    int parsed;
    if (g_tok_buf) { free(g_tok_buf); g_tok_buf = NULL; free(g_vocab); g_vocab = NULL; }
    if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; g_sorted_init = 0; }
    f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 4) { fclose(f); return 0; }
    buf = (u8*)malloc(sz);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, sz, f) != sz) { free(buf); fclose(f); return 0; }
    fclose(f);
    memcpy(&count, buf, 4);
    if (count == 0) { free(buf); return 0; }
    g_vocab = (TokenEntry*)malloc(sizeof(TokenEntry) * count);
    if (!g_vocab) { free(buf); return 0; }
    pos = 4;
    parsed = 0;
    for (i = 0; i < (int)count && pos < sz; i++) {
        int tlen = (int)buf[pos++];
        if (pos + (size_t)tlen > sz) break;
        g_vocab[i].len = tlen;
        g_vocab[i].text = (char*)(buf + pos);
        pos += tlen;
        parsed++;
    }
    if (parsed <= 0) {
        free(g_vocab); g_vocab = NULL;
        free(buf); return 0;
    }
    g_vocab_n = parsed;
    g_tok_buf = buf;
    printf("Loaded tokenizer: %d tokens\n", g_vocab_n);
    return 1;
}

static int load_tokenizer_auto(const char *model_path, const char *tok_path, GGUFContext *ctx) {
    char found[768];

    if (tok_path && tok_path[0] && strcmp(tok_path, "auto") != 0) {
        if (load_tokenizer(tok_path)) {
            printf("Tokenizer source: %s\n", tok_path);
            return 1;
        }
        printf("Warning: failed to load tokenizer file %s\n", tok_path);
    }

    if (discover_tokenizer_path(model_path, found, sizeof(found))) {
        if (load_tokenizer(found)) {
            printf("Tokenizer source: %s\n", found);
            return 1;
        }
        printf("Warning: failed to load discovered tokenizer file %s\n", found);
    }

    if (load_tokenizer_from_gguf(ctx)) {
        printf("Tokenizer source: embedded GGUF metadata\n");
        return 1;
    }

    printf("Warning: no tokenizer file found; falling back to raw byte tokens\n");
    return 0;
}

static void init_sorted_vocab(void) {
    /* Counting sort by token length - O(n) instead of O(n^2) bubble sort.
       Sorts descending by length (longest tokens first for greedy matching). */
    int i;
    int counts[256];
    int offsets[256];
    int cursors[256];
    int len;
    if (g_sorted_init || g_vocab_n <= 0) return;
    g_sorted_vocab = (int*)malloc(sizeof(int) * g_vocab_n);
    if (!g_sorted_vocab) return;
    memset(counts, 0, sizeof(counts));
    for (i = 0; i < g_vocab_n; i++) {
        len = g_vocab[i].len;
        if (len > 255) len = 255;
        if (len < 0) len = 0;
        counts[len]++;
    }
    offsets[255] = 0;
    for (len = 254; len >= 0; len--) {
        offsets[len] = offsets[len + 1] + counts[len + 1];
    }
    memcpy(cursors, offsets, sizeof(cursors));
    for (i = 0; i < g_vocab_n; i++) {
        len = g_vocab[i].len;
        if (len > 255) len = 255;
        if (len < 0) len = 0;
        g_sorted_vocab[cursors[len]++] = i;
    }
    g_sorted_init = 1;
    printf("Tokenizer sorted: %d tokens by length\n", g_vocab_n);
}

static int tokenize(const char *text, int *tokens, int max_tokens, int vocab_size) {
    int len = (int)strlen(text);
    int pos = 0;
    int n = 0;
    int i;
    init_sorted_vocab();
    while (pos < len && n < max_tokens) {
        int best = -1;
        int best_len = 0;
        if (g_vocab && g_sorted_init) {
            for (i = 0; i < g_vocab_n; i++) {
                int idx = g_sorted_vocab[i];
                int tlen = g_vocab[idx].len;
                if (tlen > best_len && pos + tlen <= len &&
                    memcmp(text + pos, g_vocab[idx].text, tlen) == 0) {
                    best_len = tlen;
                    best = idx;
                }
                if (tlen < best_len) break;
            }
        } else if (g_vocab) {
            for (i = 0; i < g_vocab_n; i++) {
                int tlen = g_vocab[i].len;
                if (tlen > 0 && pos + tlen <= len && memcmp(text + pos, g_vocab[i].text, tlen) == 0) {
                    if (tlen > best_len) { best_len = tlen; best = i; }
                }
            }
        }
        if (best >= 0) {
            tokens[n++] = best;
            pos += best_len;
        } else {
            tokens[n++] = (unsigned char)text[pos];
            pos++;
        }
    }
    return n;
}

static int append_text_tokens(int *dst, int cap, int n, const char *text, int vocab_size) {
    int tmp[256];
    int tmp_n;
    int i;
    if (!dst || !text || cap <= 0 || n >= cap) return n;
    tmp_n = tokenize(text, tmp, 256, vocab_size);
    for (i = 0; i < tmp_n && n < cap; i++) dst[n++] = tmp[i];
    return n;
}

static void print_token_text_clean(const char *text, int len) {
    int i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)text[i];
        if (c == '<' && i + 5 < len &&
            text[i + 1] == '0' &&
            text[i + 2] == 'x' &&
            isxdigit((unsigned char)text[i + 3]) &&
            isxdigit((unsigned char)text[i + 4]) &&
            text[i + 5] == '>') {
            /* Byte-fallback token like <0xE4>; drop it instead of printing noise. */
            i += 6;
            continue;
        }
        if (c < 0x80) {
            if (c == '\n' || c == '\r' || c == '\t' || (c >= 32 && c < 127)) {
                putchar((char)c);
            } else if (c == ' ') {
                putchar(' ');
            }
            i++;
        } else if (i + 2 < len &&
                   (unsigned char)text[i + 0] == 0xE2 &&
                   (unsigned char)text[i + 1] == 0x96 &&
                   (unsigned char)text[i + 2] == 0x81) {
            /* SentencePiece leading marker "▁" -> plain space. */
            putchar(' ');
            i += 3;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
            i += 4;
        } else {
            i++;
        }
    }
}

static void detokenize(int token, int vocab_size) {
    /* Only skip padding token 0. Print everything else including EOS/BOS. */
    if (token == 0) return;
    if (g_vocab && token >= 0 && token < g_vocab_n && g_vocab[token].len > 0) {
        print_token_text_clean(g_vocab[token].text, g_vocab[token].len);
    } else {
        if (token >= 32 && token < 127) printf("%c", (char)token);
    }
}

static int find_special_token(const char *name) {
    int i;
    int namelen = (int)strlen(name);
    if (!g_vocab) return -1;
    for (i = 0; i < g_vocab_n; i++) {
        if (g_vocab[i].len == namelen && memcmp(g_vocab[i].text, name, namelen) == 0)
            return i;
    }
    return -1;
}

static void scan_special_tokens(void) {
    int ids[9];
    const char *names[9] = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<s>", "</s>",
        "<bos>", "<|user|>", "<|assistant|>", "<start_of_turn>"
    };
    int i;
    printf("Special tokens: ");
    for (i = 0; i < 9; i++) {
        ids[i] = find_special_token(names[i]);
        if (ids[i] >= 0) printf("%s=%d ", names[i], ids[i]);
    }
    if (find_special_token("<end_of_turn>") >= 0) printf("<end_of_turn>=%d ", find_special_token("<end_of_turn>"));
    if (find_special_token("\n") >= 0) printf("\\n=%d ", find_special_token("\n"));
    printf("\n");
}

static void path_stem(const char *path, char *out, size_t out_cap) {
    const char *base = path;
    const char *p;
    size_t len;
    if (!out || out_cap == 0) return;
    out[0] = '\0';
    if (!path) return;
    for (p = path; *p; p++) {
        if (*p == '\\' || *p == '/' || *p == ':') base = p + 1;
    }
    len = 0;
    while (base[len] && base[len] != '.' && len + 1 < out_cap) len++;
    memcpy(out, base, len);
    out[len] = '\0';
}

static int append_raw(char *dst, size_t cap, size_t *len, const char *src) {
    size_t slen = strlen(src);
    if (*len + slen + 1 > cap) return 0;
    memcpy(dst + *len, src, slen);
    *len += slen;
    dst[*len] = '\0';
    return 1;
}

static int append_char_raw(char *dst, size_t cap, size_t *len, char c) {
    if (*len + 2 > cap) return 0;
    dst[(*len)++] = c;
    dst[*len] = '\0';
    return 1;
}

static int append_hex4(char *dst, size_t cap, size_t *len, unsigned int v) {
    char tmp[7];
    sprintf(tmp, "\\u%04X", v & 0xFFFFU);
    return append_raw(dst, cap, len, tmp);
}

static int append_utf8_codepoint(char *dst, size_t cap, size_t *len, unsigned int cp) {
    if (cp <= 0x7FU) {
        return append_char_raw(dst, cap, len, (char)cp);
    } else if (cp <= 0x7FFU) {
        if (*len + 3 > cap) return 0;
        dst[(*len)++] = (char)(0xC0 | (cp >> 6));
        dst[(*len)++] = (char)(0x80 | (cp & 0x3F));
        dst[*len] = '\0';
        return 1;
    } else if (cp <= 0xFFFFU) {
        if (*len + 4 > cap) return 0;
        dst[(*len)++] = (char)(0xE0 | (cp >> 12));
        dst[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        dst[(*len)++] = (char)(0x80 | (cp & 0x3F));
        dst[*len] = '\0';
        return 1;
    } else {
        if (*len + 5 > cap) return 0;
        dst[(*len)++] = (char)(0xF0 | (cp >> 18));
        dst[(*len)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        dst[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        dst[(*len)++] = (char)(0x80 | (cp & 0x3F));
        dst[*len] = '\0';
        return 1;
    }
}

static int append_json_string(char *dst, size_t cap, size_t *len, const char *src) {
    const unsigned char *p = (const unsigned char*)src;
    while (*p) {
        unsigned char c = *p++;
        if (c == '\"') {
            if (!append_raw(dst, cap, len, "\\\"")) return 0;
        } else if (c == '\\') {
            if (!append_raw(dst, cap, len, "\\\\")) return 0;
        } else if (c == '\b') {
            if (!append_raw(dst, cap, len, "\\b")) return 0;
        } else if (c == '\f') {
            if (!append_raw(dst, cap, len, "\\f")) return 0;
        } else if (c == '\n') {
            if (!append_raw(dst, cap, len, "\\n")) return 0;
        } else if (c == '\r') {
            if (!append_raw(dst, cap, len, "\\r")) return 0;
        } else if (c == '\t') {
            if (!append_raw(dst, cap, len, "\\t")) return 0;
        } else if (c < 0x20) {
            if (!append_hex4(dst, cap, len, c)) return 0;
        } else {
            if (!append_char_raw(dst, cap, len, (char)c)) return 0;
        }
    }
    return 1;
}

static int append_u32_text(char *dst, size_t cap, size_t *len, unsigned int v) {
    char tmp[32];
    sprintf(tmp, "%u", v);
    return append_raw(dst, cap, len, tmp);
}

static int append_float_text(char *dst, size_t cap, size_t *len, float v) {
    char tmp[64];
    sprintf(tmp, "%.6g", (double)v);
    return append_raw(dst, cap, len, tmp);
}

static int read_file_text(const char *path, char *buf, size_t cap) {
    FILE *fp;
    size_t r;
    if (!path || !buf || cap == 0) return 0;
    fp = fopen(path, "rb");
    if (!fp) return 0;
    r = fread(buf, 1, cap - 1, fp);
    buf[r] = '\0';
    fclose(fp);
    return 1;
}

static int decode_json_string(const char *src, char *dst, size_t dst_cap) {
    const char *p = src;
    size_t out = 0;
    while (*p && out + 1 < dst_cap) {
        unsigned char c = (unsigned char)*p++;
        if (c == '\"') break;
        if (c != '\\') {
            dst[out++] = (char)c;
            continue;
        }
        c = (unsigned char)*p++;
        if (c == 'n') dst[out++] = '\n';
        else if (c == 'r') dst[out++] = '\r';
        else if (c == 't') dst[out++] = '\t';
        else if (c == 'b') dst[out++] = '\b';
        else if (c == 'f') dst[out++] = '\f';
        else if (c == '\\') dst[out++] = '\\';
        else if (c == '\"') dst[out++] = '\"';
        else if (c == 'u') {
            unsigned int cp = 0;
            int i;
            for (i = 0; i < 4 && isxdigit((unsigned char)p[i]); i++) {
                unsigned char ch = (unsigned char)p[i];
                cp <<= 4;
                if (ch >= '0' && ch <= '9') cp |= (unsigned int)(ch - '0');
                else if (ch >= 'a' && ch <= 'f') cp |= (unsigned int)(10 + ch - 'a');
                else if (ch >= 'A' && ch <= 'F') cp |= (unsigned int)(10 + ch - 'A');
            }
            if (i == 4) {
                p += 4;
                if (!append_utf8_codepoint(dst, dst_cap, &out, cp)) return 0;
            }
        } else {
            dst[out++] = (char)c;
        }
    }
    dst[out] = '\0';
    return 1;
}

static int json_extract_string_field(const char *json, const char *field, char *out, size_t out_cap) {
    const char *p = json;
    size_t flen = strlen(field);
    char needle[128];
    if (!json || !field || !out || out_cap == 0) return 0;
    if (flen + 3 >= sizeof(needle)) return 0;
    needle[0] = '\"';
    memcpy(needle + 1, field, flen);
    needle[flen + 1] = '\"';
    needle[flen + 2] = '\0';
    while ((p = strstr(p, needle)) != NULL) {
        const char *q = p + flen + 2;
        while (*q && *q != ':') q++;
        if (*q != ':') { p += flen + 2; continue; }
        q++;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q != '\"') { p += flen + 2; continue; }
        q++;
        return decode_json_string(q, out, out_cap);
    }
    return 0;
}

static int utf8_to_wide(const char *src, WCHAR *dst, int dst_cap) {
    int need;
    if (!src || !dst || dst_cap <= 0) return 0;
    need = MultiByteToWideChar(CP_UTF8, 0, src, -1, dst, dst_cap);
    return need != 0;
}

static int parse_http_url(const char *url, int *secure, char *host, size_t host_cap,
                          INTERNET_PORT *port, char *path, size_t path_cap) {
    const char *p;
    const char *host_start;
    const char *host_end;
    const char *path_start;
    const char *colon;
    size_t host_len;
    size_t path_len;
    if (!url || !secure || !host || !port || !path || host_cap == 0 || path_cap == 0) return 0;
    p = strstr(url, "://");
    if (!p) return 0;
    if (strncmp(url, "https", 5) == 0) *secure = 1;
    else if (strncmp(url, "http", 4) == 0) *secure = 0;
    else return 0;
    p += 3;
    host_start = p;
    while (*p && *p != '/' && *p != '?' && *p != '#') p++;
    host_end = p;
    colon = NULL;
    for (p = host_start; p < host_end; p++) {
        if (*p == ':') colon = p;
    }
    if (colon) {
        host_len = (size_t)(colon - host_start);
        *port = (INTERNET_PORT)atoi(colon + 1);
        if (*port == 0) *port = *secure ? 443 : 80;
    } else {
        host_len = (size_t)(host_end - host_start);
        *port = *secure ? 443 : 80;
    }
    if (host_len >= host_cap) host_len = host_cap - 1;
    memcpy(host, host_start, host_len);
    host[host_len] = '\0';
    path_start = host_end;
    if (*path_start == '\0') {
        path[0] = '/';
        path[1] = '\0';
        return 1;
    }
    if (*path_start != '/') {
        if (path_cap < 2) return 0;
        path[0] = '/';
        path_len = strlen(path_start);
        if (path_len + 1 >= path_cap) path_len = path_cap - 2;
        memcpy(path + 1, path_start, path_len);
        path[path_len + 1] = '\0';
        return 1;
    }
    path_len = strlen(path_start);
    if (path_len >= path_cap) path_len = path_cap - 1;
    memcpy(path, path_start, path_len);
    path[path_len] = '\0';
    return 1;
}

static int http_post_json(const char *url, const char *api_key, const char *body, char **out_resp) {
    HMODULE dll;
    PFN_WinHttpOpen pOpen;
    PFN_WinHttpConnect pConnect;
    PFN_WinHttpOpenRequest pOpenRequest;
    PFN_WinHttpAddRequestHeaders pAddHeaders;
    PFN_WinHttpSendRequest pSendRequest;
    PFN_WinHttpReceiveResponse pReceiveResponse;
    PFN_WinHttpQueryDataAvailable pQueryDataAvailable;
    PFN_WinHttpReadData pReadData;
    PFN_WinHttpCloseHandle pCloseHandle;
    PFN_WinHttpQueryHeaders pQueryHeaders;
    WCHAR w_host[256];
    WCHAR w_path[2048];
    WCHAR w_headers[1024];
    char host[256];
    char path[2048];
    char request_path[2304];
    char header_ascii[1024];
    HINTERNET hSession = NULL;
    HINTERNET hConnect = NULL;
    HINTERNET hRequest = NULL;
    DWORD status = 0;
    DWORD status_len = sizeof(status);
    const char *body_ptr = body ? body : "";
    DWORD body_len = (DWORD)strlen(body_ptr);
    char *resp = NULL;
    size_t resp_len = 0;
    size_t resp_cap = 0;
    DWORD available = 0;
    DWORD read_len = 0;
    int secure = 0;
    INTERNET_PORT port = 0;

    if (!url || !out_resp) return 0;
    *out_resp = NULL;

    dll = LoadLibraryA("winhttp.dll");
    if (!dll) {
        fprintf(stderr, "API error: winhttp.dll not available\n");
        return 0;
    }
    pOpen = (PFN_WinHttpOpen)GetProcAddress(dll, "WinHttpOpen");
    pConnect = (PFN_WinHttpConnect)GetProcAddress(dll, "WinHttpConnect");
    pOpenRequest = (PFN_WinHttpOpenRequest)GetProcAddress(dll, "WinHttpOpenRequest");
    pAddHeaders = (PFN_WinHttpAddRequestHeaders)GetProcAddress(dll, "WinHttpAddRequestHeaders");
    pSendRequest = (PFN_WinHttpSendRequest)GetProcAddress(dll, "WinHttpSendRequest");
    pReceiveResponse = (PFN_WinHttpReceiveResponse)GetProcAddress(dll, "WinHttpReceiveResponse");
    pQueryDataAvailable = (PFN_WinHttpQueryDataAvailable)GetProcAddress(dll, "WinHttpQueryDataAvailable");
    pReadData = (PFN_WinHttpReadData)GetProcAddress(dll, "WinHttpReadData");
    pCloseHandle = (PFN_WinHttpCloseHandle)GetProcAddress(dll, "WinHttpCloseHandle");
    pQueryHeaders = (PFN_WinHttpQueryHeaders)GetProcAddress(dll, "WinHttpQueryHeaders");
    if (!pOpen || !pConnect || !pOpenRequest || !pAddHeaders || !pSendRequest ||
        !pReceiveResponse || !pQueryDataAvailable || !pReadData || !pCloseHandle || !pQueryHeaders) {
        fprintf(stderr, "API error: WinHTTP exports missing\n");
        FreeLibrary(dll);
        return 0;
    }

    if (!parse_http_url(url, &secure, host, sizeof(host), &port, path, sizeof(path))) {
        fprintf(stderr, "API error: invalid URL '%s'\n", url);
        FreeLibrary(dll);
        return 0;
    }
    sprintf(request_path, "%s", path[0] ? path : "/");
    if (!utf8_to_wide(host, w_host, sizeof(w_host) / sizeof(w_host[0])) ||
        !utf8_to_wide(request_path, w_path, sizeof(w_path) / sizeof(w_path[0]))) {
        fprintf(stderr, "API error: could not convert request URL\n");
        FreeLibrary(dll);
        return 0;
    }
    if (api_key && api_key[0]) {
        sprintf(header_ascii, "Content-Type: application/json\r\nAccept: application/json\r\nAuthorization: Bearer %s\r\n", api_key);
    } else {
        sprintf(header_ascii, "Content-Type: application/json\r\nAccept: application/json\r\n");
    }
    if (!utf8_to_wide(header_ascii, w_headers, sizeof(w_headers) / sizeof(w_headers[0]))) {
        fprintf(stderr, "API error: could not convert headers\n");
        FreeLibrary(dll);
        return 0;
    }
    hSession = pOpen(L"windows2000ai/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                     WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) goto cleanup;
    hConnect = pConnect(hSession, w_host, port, 0);
    if (!hConnect) goto cleanup;
    hRequest = pOpenRequest(hConnect, L"POST", w_path, NULL, WINHTTP_NO_REFERER,
                            WINHTTP_DEFAULT_ACCEPT_TYPES,
                            secure ? WINHTTP_FLAG_SECURE : 0);
    if (!hRequest) goto cleanup;
    if (!pAddHeaders(hRequest, w_headers, (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD)) goto cleanup;
    if (!pSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                      (LPVOID)body_ptr, body_len, body_len, 0)) goto cleanup;
    if (!pReceiveResponse(hRequest, NULL)) goto cleanup;
    if (pQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                      WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_len,
                      WINHTTP_NO_HEADER_INDEX)) {
        if (status < 200 || status >= 300) {
            fprintf(stderr, "API error: HTTP status %lu\n", (unsigned long)status);
        }
    }

    for (;;) {
        if (!pQueryDataAvailable(hRequest, &available)) break;
        if (available == 0) break;
        if (resp_len + available + 1 > resp_cap) {
            size_t new_cap = resp_cap ? resp_cap * 2 : 8192;
            while (new_cap < resp_len + available + 1) new_cap *= 2;
            {
                char *tmp = (char*)realloc(resp, new_cap);
                if (!tmp) goto cleanup;
                resp = tmp;
            }
            resp_cap = new_cap;
        }
        if (!pReadData(hRequest, resp + resp_len, available, &read_len)) break;
        resp_len += read_len;
        resp[resp_len] = '\0';
    }

    if (!resp) {
        resp = (char*)malloc(1);
        if (!resp) goto cleanup;
        resp[0] = '\0';
    }

    *out_resp = resp;
    resp = NULL;

cleanup:
    free(resp);
    if (hRequest) pCloseHandle(hRequest);
    if (hConnect) pCloseHandle(hConnect);
    if (hSession) pCloseHandle(hSession);
    FreeLibrary(dll);
    return *out_resp != NULL;
}

static int run_api_mode(const char *model_path, const char *prompt, int n_generate,
                        float temp, float top_p, const char *api_url,
                        const char *api_model, const char *api_key,
                        const char *api_system) {
    char model_name[256];
    char *body;
    char *resp;
    char content[65536];
    size_t cap;
    size_t len = 0;
    double t0 = now_seconds();
    int ok;

    if (!api_url || !api_url[0]) api_url = "https://api.openai.com/v1/chat/completions";
    if (!api_model || !api_model[0]) {
        path_stem(model_path, model_name, sizeof(model_name));
        if (!model_name[0]) strcpy(model_name, "gpt-4o-mini");
        api_model = model_name;
    }

    cap = (prompt ? strlen(prompt) : 0) * 6 + (api_system ? strlen(api_system) : 0) * 6 + 4096;
    body = (char*)malloc(cap);
    if (!body) {
        fprintf(stderr, "API error: out of memory\n");
        return 1;
    }
    body[0] = '\0';

    if (!append_raw(body, cap, &len, "{")) goto body_fail;
    if (!append_raw(body, cap, &len, "\"model\":\"")) goto body_fail;
    if (!append_json_string(body, cap, &len, api_model)) goto body_fail;
    if (!append_raw(body, cap, &len, "\",\"messages\":[")) goto body_fail;
    if (api_system && api_system[0]) {
        if (!append_raw(body, cap, &len, "{\"role\":\"system\",\"content\":\"")) goto body_fail;
        if (!append_json_string(body, cap, &len, api_system)) goto body_fail;
        if (!append_raw(body, cap, &len, "\"},")) goto body_fail;
    }
    if (!append_raw(body, cap, &len, "{\"role\":\"user\",\"content\":\"")) goto body_fail;
    if (!append_json_string(body, cap, &len, prompt ? prompt : "")) goto body_fail;
    if (!append_raw(body, cap, &len, "\"}")) goto body_fail;
    if (!append_raw(body, cap, &len, "],\"temperature\":")) goto body_fail;
    if (!append_float_text(body, cap, &len, temp)) goto body_fail;
    if (!append_raw(body, cap, &len, ",\"top_p\":")) goto body_fail;
    if (!append_float_text(body, cap, &len, top_p)) goto body_fail;
    if (!append_raw(body, cap, &len, ",\"max_tokens\":")) goto body_fail;
    if (!append_u32_text(body, cap, &len, (unsigned int)(n_generate > 0 ? n_generate : 1))) goto body_fail;
    if (!append_raw(body, cap, &len, ",\"stream\":false}")) goto body_fail;

    if (!g_clean_output) {
        printf("API mode: %s\n", api_url);
        printf("API model: %s\n", api_model);
    }

    ok = http_post_json(api_url, api_key, body, &resp);
    free(body);
    if (!ok || !resp) {
        fprintf(stderr, "API error: request failed\n");
        return 1;
    }

    if (!json_extract_string_field(resp, "content", content, sizeof(content)) &&
        !json_extract_string_field(resp, "text", content, sizeof(content))) {
        if (!g_clean_output) fprintf(stderr, "API warning: could not extract text field\n");
        printf("%s\n", resp);
    } else {
        printf("%s\n", content);
    }
    if (!g_clean_output) {
        printf("API latency: %.3f s\n", now_seconds() - t0);
    }
    free(resp);
    return 0;

body_fail:
    free(body);
    fprintf(stderr, "API error: failed to build request body\n");
    return 1;
}

/* --- Main --- */

int main(int argc, char **argv) {
    GGUFContext *ctx;
    Model m;
    RunState s;
    HParams hp;
    char *model_path = NULL;
    char *tok_path = NULL;
    char *prompt = "";
    char *prompt_file = NULL;
    char *api_url = NULL;
    char *api_model = NULL;
    char *api_key = NULL;
    char *api_system = NULL;
    int n_generate = 64;
    float temp = 0.8f;
    float top_p = 0.95f;
    float min_p = 0.0f;
    int top_k = 40;
    int eos_token = 2;
    int no_eos_stop = 0;
    int do_chat = 0;
    int use_api = 0;
    int do_info = 0;
    int do_list = 0;
    int do_raw_tokens = 0;
    int tinyllama_mode = 0;
    int gemma3_mode = 0;
    int smollm2_mode = 0;
    int slopllm_mode = 0;
    int seq_user = 0;
    int n_heads_user = 0;
    int n_kv_user = 0;
    int hidden_user = 0;
    int temp_user = 0;
    int top_k_user = 0;
    int top_p_user = 0;
    int min_p_user = 0;
    unsigned int seed = 0;
    int arg;
    int pos;
    int prompt_len;
    int *prompt_tokens;
    int next_token = 0;
    int i;
    double t_request_start;
    double t_first_token = 0.0;
    double t_end;
    int tokens_printed = 0;
    if (argc < 2) {
        printf("Usage: gguf_infer.exe <model.gguf> [options]\n");
        printf("Options:\n");
        printf("  --info          Dump model info and exit\n");
        printf("  --list          List all tensors and exit\n");
        printf("  --clean         Suppress non-essential output\n");
        printf("  --no-eos-stop   Do not stop on EOS token\n");
        printf("  --raw-tokens    Print token IDs instead of text\n");
        printf("  --chat          Wrap prompt in chat template\n");
        printf("  --api           Use an OpenAI-compatible API instead of local GGUF inference\n");
        printf("  --api-url <u>   API endpoint URL (default OPENAI_BASE_URL or OpenAI chat completions)\n");
        printf("  --api-model <m> API model name (default OPENAI_MODEL or model file stem)\n");
        printf("  --api-key <k>   API key (default OPENAI_API_KEY)\n");
        printf("  --api-system <s> System prompt for API mode\n");
        printf("  -f <file>       Read prompt from file\n");
        printf("  -n <num>        Max tokens to generate (default 64)\n");
        printf("  -t <temp>       Temperature, 0=argmax (default 0.8)\n");
        printf("  -s <seed>       Random seed (default time)\n");
        printf("  --prompt <s>    Prompt string\n");
        printf("  --tok <file|auto>   tokenizer.bin or auto-discover\n");
        printf("  --n_heads <n>   Override head count\n");
        printf("  --n_kv <n>      Override KV head count\n");
        printf("  --hidden <n>    Override hidden dim\n");
        printf("  --seq <n>       Override max sequence length\n");
        printf("  --top-k <n>     Top-k sampling, 0=disabled (default 40)\n");
        printf("  --top-p <p>     Top-p sampling (default 0.95)\n");
        printf("  --min-p <p>     Min-p sampling (default 0.0)\n");
        printf("  --eos <id>      EOS token ID (default 2)\n");
        printf("  --repeat-penalty <p>  Repetition penalty (default 1.15)\n");
        printf("  --threads <n|auto>   Parallel threads for large matmul (WinNT only, default auto)\n");
        return 1;
    }

    memset(&m, 0, sizeof(Model));
    m.repeat_penalty = 1.15f;

    model_path = argv[1];
    tinyllama_mode = contains_nocase(model_path, "tinyllama");
    for (arg = 2; arg < argc; arg++) {
        if (strcmp(argv[arg], "--info") == 0) { do_info = 1; }
        else if (strcmp(argv[arg], "--clean") == 0) { g_clean_output = 1; }
        else if (strcmp(argv[arg], "--no-eos-stop") == 0) { no_eos_stop = 1; }
        else if (strcmp(argv[arg], "--raw-tokens") == 0) { do_raw_tokens = 1; }
        else if (strcmp(argv[arg], "--chat") == 0) { do_chat = 1; }
        else if (strcmp(argv[arg], "--api") == 0) { use_api = 1; }
        else if (strcmp(argv[arg], "--api-url") == 0 && arg + 1 < argc) { api_url = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--api-model") == 0 && arg + 1 < argc) { api_model = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--api-key") == 0 && arg + 1 < argc) { api_key = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--api-system") == 0 && arg + 1 < argc) { api_system = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--threads") == 0 && arg + 1 < argc) {
            if (strcmp(argv[arg+1], "auto") == 0 || strcmp(argv[arg+1], "AUTO") == 0) g_n_threads = 0;
            else g_n_threads = atoi(argv[arg+1]);
            arg++;
        }
        else if (strcmp(argv[arg], "--list") == 0) { do_list = 1; }
        else if (strcmp(argv[arg], "-f") == 0 && arg + 1 < argc) { prompt_file = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "-n") == 0 && arg + 1 < argc) { n_generate = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "-t") == 0 && arg + 1 < argc) { temp = (float)atof(argv[arg+1]); temp_user = 1; arg++; }
        else if (strcmp(argv[arg], "-s") == 0 && arg + 1 < argc) { seed = (unsigned int)atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--prompt") == 0 && arg + 1 < argc) { prompt = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--tok") == 0 && arg + 1 < argc) { tok_path = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--n_heads") == 0 && arg + 1 < argc) { n_heads_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--n_kv") == 0 && arg + 1 < argc) { n_kv_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--hidden") == 0 && arg + 1 < argc) { hidden_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--seq") == 0 && arg + 1 < argc) { seq_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--top-k") == 0 && arg + 1 < argc) { top_k = atoi(argv[arg+1]); top_k_user = 1; arg++; }
        else if (strcmp(argv[arg], "--top-p") == 0 && arg + 1 < argc) { top_p = (float)atof(argv[arg+1]); top_p_user = 1; arg++; }
        else if (strcmp(argv[arg], "--min-p") == 0 && arg + 1 < argc) { min_p = (float)atof(argv[arg+1]); min_p_user = 1; arg++; }
        else if (strcmp(argv[arg], "--eos") == 0 && arg + 1 < argc) { eos_token = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--repeat-penalty") == 0 && arg + 1 < argc) { m.repeat_penalty = (float)atof(argv[arg+1]); arg++; }
    }

    if (g_n_threads <= 0) g_n_threads = default_thread_count();
    if (g_n_threads < 1) g_n_threads = 1;

    if (seed == 0) seed = (unsigned int)time(NULL);
    srand(seed);

    if (!g_clean_output) {
        printf("gguf_infer v6 (multi-arch + sorted tokenizer)\n");
        printf("Threads: %d\n", g_n_threads);
    }

    if (use_api) {
        char prompt_buf[8192];
        if (prompt_file && read_file_text(prompt_file, prompt_buf, sizeof(prompt_buf))) {
            prompt = prompt_buf;
        }
        if (!api_url) api_url = getenv("OPENAI_BASE_URL");
        if (!api_url) api_url = getenv("OPENAI_API_BASE");
        if (!api_key) api_key = getenv("OPENAI_API_KEY");
        if (!api_model) api_model = getenv("OPENAI_MODEL");
        if (!api_system) api_system = getenv("OPENAI_SYSTEM_PROMPT");
        return run_api_mode(model_path, prompt, n_generate, temp, top_p, api_url, api_model, api_key, api_system);
    }

    ctx = gguf_load(model_path);
    if (!ctx) return 1;

    if (do_list) {
        list_tensors(ctx);
        gguf_free(ctx);
        return 0;
    }

    if (do_info) {
        printf("GGUF file: %s\n", model_path);
        printf("  Version: 3\n");
        printf("  Tensors: %u\n", (unsigned int)ctx->n_tensors);
        printf("  Metadata keys: (parsed)\n");
        printf("  Architecture: %s\n", ctx->hp.architecture);
        printf("  Context length: %u\n", (unsigned int)ctx->hp.context_length);
        printf("  Embedding length: %u\n", (unsigned int)ctx->hp.embedding_length);
        printf("  Block count: %u\n", (unsigned int)ctx->hp.block_count);
        printf("  Head count: %u\n", (unsigned int)ctx->hp.attention_head_count);
        printf("  KV head count: %u\n", (unsigned int)ctx->hp.attention_head_count_kv);
        printf("  FFN length: %u\n", (unsigned int)ctx->hp.feed_forward_length);
        gguf_free(ctx);
        return 0;
    }

    hp = ctx->hp;
    gemma3_mode = (m.arch == ARCH_GEMMA3) ||
                  contains_nocase(model_path, "gemma3") ||
                  contains_nocase(hp.chat_template, "start_of_turn");
    smollm2_mode = contains_nocase(model_path, "smollm2") ||
                   contains_nocase(hp.tokenizer_pre, "smollm") ||
                   contains_nocase(hp.chat_template, "SmolLM");
    slopllm_mode = contains_nocase(model_path, "slopllm") ||
                   contains_nocase(hp.chat_template, "SlopLLM") ||
                   contains_nocase(hp.chat_template, "SlopAI");

    if (m.arch == ARCH_GEMMA3) {
        if (!temp_user) temp = 1.0f;
        if (!top_k_user) top_k = 64;
        if (!top_p_user) top_p = 0.95f;
        if (!min_p_user) min_p = 0.0f;
    }

    load_tokenizer_auto(model_path, tok_path, ctx);

    {
        int est_dim = (int)hp.embedding_length;
        int est_layers = (int)hp.block_count;
        int est_hidden = (int)hp.feed_forward_length;
        int est_seq = seq_user ? seq_user : ((int)hp.context_length ? (int)hp.context_length : 512);
        unsigned int file_mb = (unsigned int)(ctx->size / (1024*1024));
        unsigned int cache_kb = (unsigned int)((u64)est_layers * (2 * est_dim * est_dim + est_dim * est_hidden) * 4 / 1024);
        unsigned int state_kb = (unsigned int)((u64)est_seq * est_dim * 2 * 4 / 1024);
        printf("Estimated memory: file=%uMB + matmul_workspace=%uKB + run_state=%uKB\n",
               file_mb, cache_kb, state_kb);
        printf("Total RAM needed: ~%u MB\n", file_mb + cache_kb/1024 + state_kb/1024);
    }

    if (build_model(ctx, &m, &hp, n_heads_user, n_kv_user, hidden_user, seq_user) != 0) {
        gguf_free(ctx);
        return 1;
    }

    if (alloc_runstate(&m, &s, ctx) != 0) {
        free_model(&m);
        gguf_free(ctx);
        return 1;
    }

    /* Load prompt from file if requested */
    if (prompt_file) {
        FILE *fp = fopen(prompt_file, "rb");
        if (fp) {
            static char file_buf[8192];
            size_t r = fread(file_buf, 1, sizeof(file_buf)-1, fp);
            file_buf[r] = '\0';
            prompt = file_buf;
            fclose(fp);
        }
    }

    /* Scan special tokens */
    scan_special_tokens();

    prompt_tokens = (int*)malloc(m.max_seq_len * sizeof(int));

    /* Build prompt tokens */
    if (smollm2_mode || tinyllama_mode || gemma3_mode || slopllm_mode || do_chat) {
        int tok_bos = find_special_token("<bos>");
        int tok_im_start = find_special_token("<|im_start|>");
        int tok_im_end = find_special_token("<|im_end|>");
        int n = 0;

        if (smollm2_mode) {
            if (!g_clean_output) printf("Chat tokens: smollm2 ChatML style\n");
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "<|im_start|>user\n", m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, prompt, m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "<|im_end|>\n", m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "<|im_start|>assistant\n", m.vocab_size);
        } else if (gemma3_mode) {
            if (!g_clean_output) printf("Chat tokens: gemma3 start_of_turn style\n");
            if (tok_bos >= 0 && n < m.max_seq_len) prompt_tokens[n++] = tok_bos;
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "<start_of_turn>user\n", m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, prompt, m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "<end_of_turn>\n", m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "<start_of_turn>model\n", m.vocab_size);
        } else if (tinyllama_mode) {
            if (!g_clean_output) printf("Chat tokens: tinyllama Llama-2 style\n");
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "<s>[INST] ", m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, prompt, m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, " [/INST]", m.vocab_size);
        } else if (slopllm_mode) {
            if (!g_clean_output) printf("Chat tokens: slopllm user/assistant style\n");
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, "user: ", m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, prompt, m.vocab_size);
            n = append_text_tokens(prompt_tokens, m.max_seq_len, n, " assistant:", m.vocab_size);
        } else {
            int tmp_tok[256];
            int tmp_n;
            if (tok_im_start < 0) tok_im_start = find_special_token("<s>"); /* fallback */
            if (tok_im_end < 0) tok_im_end = find_special_token("</s>");  /* fallback to EOS */

            if (!g_clean_output) {
                printf("Chat tokens: im_start=%d im_end=%d\n", tok_im_start, tok_im_end);
            }

            /* <|im_start|> */
            if (tok_im_start >= 0 && n < m.max_seq_len) prompt_tokens[n++] = tok_im_start;

            /* "user\n" */
            tmp_n = tokenize("user\n", tmp_tok, 256, m.vocab_size);
            for (i = 0; i < tmp_n && n < m.max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];

            /* user prompt */
            tmp_n = tokenize(prompt, tmp_tok, 256, m.vocab_size);
            for (i = 0; i < tmp_n && n < m.max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];

            /* "\n" */
            tmp_n = tokenize("\n", tmp_tok, 256, m.vocab_size);
            for (i = 0; i < tmp_n && n < m.max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];

            /*  =2 */
            if (tok_im_end >= 0 && n < m.max_seq_len) prompt_tokens[n++] = tok_im_end;

            /* "\n" */
            tmp_n = tokenize("\n", tmp_tok, 256, m.vocab_size);
            for (i = 0; i < tmp_n && n < m.max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];

            /* <|im_start|> */
            if (tok_im_start >= 0 && n < m.max_seq_len) prompt_tokens[n++] = tok_im_start;

            /* "assistant\n" */
            tmp_n = tokenize("assistant\n", tmp_tok, 256, m.vocab_size);
            for (i = 0; i < tmp_n && n < m.max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];
        }

        prompt_len = n;
    } else {
        prompt_len = tokenize(prompt, prompt_tokens, m.max_seq_len, m.vocab_size);
    }

    /* Debug: show first 30 prompt tokens */
    if (!g_clean_output) {
        printf("Prompt tokens (%d): ", prompt_len);
        for (i = 0; i < prompt_len && i < 30; i++) {
            printf("[%d]", prompt_tokens[i]);
            if (g_vocab && prompt_tokens[i] >= 0 && prompt_tokens[i] < g_vocab_n) {
                int tlen = g_vocab[prompt_tokens[i]].len;
                if (tlen > 0 && tlen < 20) {
                    printf("'");
                    print_token_text_clean(g_vocab[prompt_tokens[i]].text, tlen);
                    printf("' ");
                }
            }
        }
        printf("\n");
    }

    /* Process prompt tokens silently to fill KV cache */
    t_request_start = now_seconds();
    if (prompt_len > 1 && !g_clean_output) {
        printf("Prefill %d tokens...", prompt_len);
        fflush(stdout);
    }
    for (pos = 0; pos < prompt_len - 1 && prompt_len > 0; pos++) {
        forward(&m, &s, prompt_tokens[pos], pos, 0.0f, 0, 0.0f, 0.0f, NULL, 0, -1);
        if (prompt_len > 1 && !g_clean_output && (pos % 2 == 1 || pos == prompt_len - 2)) {
            printf(".");
            fflush(stdout);
        }
    }

    /* Last prompt token produces first generated token using sampling */
    if (prompt_len > 1 && !g_clean_output) printf("\n");
    if (prompt_len > 0) {
        next_token = forward(&m, &s, prompt_tokens[prompt_len - 1], pos, temp, top_k, top_p, min_p, NULL, 0, eos_token);
        pos = prompt_len;
    } else {
        /* Empty prompt: use BOS token as seed */
        next_token = forward(&m, &s, 1, 0, temp, top_k, top_p, min_p, NULL, 0, eos_token);
        pos = 1;
    }

    /* Generation: stream tokens */
    {
        int *last_tokens = (int*)malloc(m.max_seq_len * sizeof(int));
        int n_last = 0;
        /* Print first generated token immediately */
        if (do_raw_tokens || next_token != 0 || no_eos_stop) {
            if (tokens_printed == 0) t_first_token = now_seconds();
            if (do_raw_tokens) {
                printf("[%d]", next_token);
            } else if (next_token != 0) {
                detokenize(next_token, m.vocab_size);
            } else {
                printf("<|endoftext|>");
            }
            fflush(stdout);
            if (n_last < m.max_seq_len) last_tokens[n_last++] = next_token;
            tokens_printed++;
        }
        for (i = 1; i < n_generate; i++) {
            if (pos >= m.max_seq_len) break;
            next_token = forward(&m, &s, next_token, pos, temp, top_k, top_p, min_p, last_tokens, n_last, -1);
            pos++;
            if (!do_raw_tokens && next_token == 0) break;
            if (next_token == eos_token && !no_eos_stop) break;
            if (do_raw_tokens) {
                printf("[%d]", next_token);
            } else if (next_token != 0) {
                detokenize(next_token, m.vocab_size);
            } else {
                printf("<|endoftext|>");
            }
            if (tokens_printed == 0) t_first_token = now_seconds();
            if (n_last < m.max_seq_len) last_tokens[n_last++] = next_token;
            tokens_printed++;
            fflush(stdout);
        }
        free(last_tokens);
    }
    printf("\n");
    t_end = now_seconds();
    if (tokens_printed > 0) {
        double ttft_s = t_first_token - t_request_start;
        double gen_s = t_end - t_first_token;
        double tps = 0.0;
        int gen_tokens = tokens_printed > 1 ? (tokens_printed - 1) : 0;
        if (gen_tokens > 0 && gen_s > 0.0) tps = (double)gen_tokens / gen_s;
        printf("Stats: TTFT=%.3fs, TPS=%.2f tok/s, output=%d tokens, prompt=%d tokens, total=%.3fs\n",
               ttft_s, tps, tokens_printed, prompt_len, t_end - t_request_start);
    } else {
        printf("Stats: no tokens produced, total=%.3fs\n", t_end - t_request_start);
    }

    free(prompt_tokens);
    free_runstate(&s);
    free_model(&m);
    gguf_free(ctx);
    if (g_tok_buf) { free(g_tok_buf); free(g_vocab); }
    if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; g_sorted_init = 0; }
    return 0;
}
