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
typedef signed __int64 s64;
#else
typedef unsigned long long u64;
typedef signed long long s64;
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
    u64 size_bytes;
    int force_transpose;
    void *data;
} Tensor;

static u64 tensor_storage_bytes(const Tensor *t);

typedef struct {
    u32 context_length;
    u32 embedding_length;
    u32 feed_forward_length;
    u32 block_count;
    u32 attention_head_count;
    u32 attention_head_count_kv;
    u32 attention_key_length;
    u32 attention_value_length;
    float attention_layer_norm_rms_epsilon;
    u32 rope_dimension_count;
    float rope_freq_base;
    float attn_logit_softcapping;
    float final_logit_softcapping;
    u32 attention_sliding_window;
    u32 alignment;
    u32 tokenizer_bos_token_id;
    u32 tokenizer_eos_token_id;
    u32 tokenizer_padding_token_id;
    int tokenizer_add_bos_token;
    int tokenizer_add_space_prefix;
    char architecture[32];
    char tokenizer_model[32];
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
    int rope_dim;
    int q_dim, kv_dim;
    int attention_sliding_window;
    float norm_eps, rope_theta, rsqrt_head_dim;
    float attn_logit_softcapping;
    float final_logit_softcapping;
    int tok_embd_transposed; /* 1 if token_embd.weight is [dim, vocab] instead of [vocab, dim] */
    float *cached_embd;      /* exact f32 cache for transposed token embeddings/output */
    float *cached_output;    /* exact f32 cache for transposed output head */
    float *cached_pos_embd;  /* exact f32 cache for transposed position embeddings */
    float *rope_cos;         /* precomputed RoPE cos table [max_seq_len][head_dim/2] */
    float *rope_sin;         /* precomputed RoPE sin table [max_seq_len][head_dim/2] */
    float *rope_cos_global;
    float *rope_sin_global;
    Tensor *tok_embd;
    Tensor *pos_embd;
    Tensor *output_norm;
    Tensor *output_norm_bias;
    Tensor *output;
    Tensor **attn_norm;
    Tensor **attn_norm_bias;
    Tensor **attn_q_norm;
    Tensor **attn_k_norm;
    Tensor **attn_q;
    Tensor **attn_q_bias;
    Tensor **attn_k;
    Tensor **attn_v;
    Tensor **attn_o;
    Tensor **attn_o_bias;
    Tensor **ffn_norm;
    Tensor **ffn_norm_bias;
    Tensor **post_attn_norm;
    Tensor **post_ffn_norm;
    Tensor **ffn_gate;
    Tensor **ffn_gate_bias;
    Tensor **ffn_up;
    Tensor **ffn_down;
    Tensor **ffn_down_bias;
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
typedef struct { const char *key; int id; } StrIdEntry;
typedef struct { int left; int right; int rank; int new_id; } BpePairEntry;

/* --- Globals --- */

static TokenEntry *g_vocab = NULL;
static int g_vocab_n = 0;
static u8 *g_tok_buf = NULL;
static int g_byte_token[256];
static int *g_sorted_vocab = NULL;
static int g_sorted_init = 0;
static void reset_byte_token_map(void);
static int g_clean_output = 0;
static int g_n_threads = 0;
static int g_allow_eog_sampling = 0;
static int g_interactive_ui = 0;
static int g_tokenizer_is_gpt2 = 0;
static int g_gpt2_add_space_prefix = 0;
static int g_legacy_windows = -1;
static int g_eog_tokens[8];
static int g_eog_token_count = 0;
static int *g_token_types = NULL;
static StrIdEntry *g_token_lookup = NULL;
static int g_token_lookup_cap = 0;
static BpePairEntry *g_bpe_pairs = NULL;
static int g_bpe_pair_cap = 0;
static int g_bpe_pair_count = 0;
static int g_gpt2_byte_token[256];
static int g_gpt2_codepoint_to_byte[65536];
static int probe_tensor_section(const u8 *base, size_t size, u64 offset, u64 tensor_count);
static int is_legacy_windows(void);
static void gguf_free(GGUFContext *ctx);
static void print_token_text_clean(const char *text, int len);
static int token_is_blocked_for_generation(int tok);

#define MATVEC_POOL_MAX_THREADS 8

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

static int read_meta_u32_value(u32 type, u8 **p, u32 *out) {
    if (!p || !*p || !out) return 0;
    switch (type) {
        case 0: case 1: case 7: {
            u8 v = *(*p)++;
            *out = (u32)v;
            return 1;
        }
        case 2: case 3: {
            u16 v;
            memcpy(&v, *p, 2);
            *p += 2;
            *out = (u32)v;
            return 1;
        }
        case 4: {
            *out = read_u32(p);
            return 1;
        }
        case 5: {
            int v;
            memcpy(&v, *p, 4);
            *p += 4;
            *out = (u32)(v < 0 ? 0 : v);
            return 1;
        }
        case 6: {
            float v;
            memcpy(&v, *p, 4);
            *p += 4;
            *out = (u32)(v <= 0.0f ? 0U : (u32)v);
            return 1;
        }
        case 10: {
            u64 v = read_u64(p);
            *out = (u32)v;
            return 1;
        }
        case 11: {
            s64 v;
            memcpy(&v, *p, 8);
            *p += 8;
            *out = (u32)(v < 0 ? 0 : (u32)v);
            return 1;
        }
        case 12: {
            double v;
            memcpy(&v, *p, 8);
            *p += 8;
            *out = (u32)(v <= 0.0 ? 0U : (u32)v);
            return 1;
        }
        default:
            return 0;
    }
}

static int read_meta_float_value(u32 type, u8 **p, float *out) {
    if (!p || !*p || !out) return 0;
    switch (type) {
        case 0: case 1: case 7: {
            u8 v = *(*p)++;
            *out = (float)v;
            return 1;
        }
        case 2: case 3: {
            u16 v;
            memcpy(&v, *p, 2);
            *p += 2;
            *out = (float)v;
            return 1;
        }
        case 4: {
            u32 v = read_u32(p);
            *out = (float)v;
            return 1;
        }
        case 5: {
            int v;
            memcpy(&v, *p, 4);
            *p += 4;
            *out = (float)v;
            return 1;
        }
        case 6: {
            memcpy(out, *p, 4);
            *p += 4;
            return 1;
        }
        case 10: {
            u64 v = read_u64(p);
            *out = (float)v;
            return 1;
        }
        case 11: {
            s64 v;
            memcpy(&v, *p, 8);
            *p += 8;
            *out = (float)v;
            return 1;
        }
        case 12: {
            double v;
            memcpy(&v, *p, 8);
            *p += 8;
            *out = (float)v;
            return 1;
        }
        default:
            return 0;
    }
}

static int read_meta_bool_value(u32 type, u8 **p, int *out) {
    u32 v = 0;
    if (!out) return 0;
    if (!read_meta_u32_value(type, p, &v)) return 0;
    *out = v ? 1 : 0;
    return 1;
}

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

static int parse_byte_fallback_token(const char *text, int len) {
    int hi, lo;
    if (len != 6) return -1;
    if (text[0] != '<' || text[1] != '0' || text[2] != 'x' || text[5] != '>') return -1;
    if (!isxdigit((unsigned char)text[3]) || !isxdigit((unsigned char)text[4])) return -1;
    hi = isdigit((unsigned char)text[3]) ? (text[3] - '0') : (10 + (toupper((unsigned char)text[3]) - 'A'));
    lo = isdigit((unsigned char)text[4]) ? (text[4] - '0') : (10 + (toupper((unsigned char)text[4]) - 'A'));
    return (hi << 4) | lo;
}

static void rebuild_byte_token_map(void) {
    int i;
    for (i = 0; i < 256; i++) g_byte_token[i] = -1;
    if (!g_vocab) return;
    for (i = 0; i < g_vocab_n; i++) {
        int byte = parse_byte_fallback_token(g_vocab[i].text, g_vocab[i].len);
        if (byte >= 0 && byte < 256 && g_byte_token[byte] < 0) {
            g_byte_token[byte] = i;
        }
    }
}

static unsigned int hash_str(const char *s) {
    unsigned int h = 2166136261U;
    while (s && *s) {
        h ^= (unsigned int)(unsigned char)*s++;
        h *= 16777619U;
    }
    return h;
}

static unsigned int hash_pair_int(int left, int right) {
    unsigned int h = 2166136261U;
    h ^= (unsigned int)left; h *= 16777619U;
    h ^= (unsigned int)right; h *= 16777619U;
    return h;
}

static unsigned int utf8_decode_one(const unsigned char *s, int len, int *used) {
    unsigned int cp;
    if (!s || len <= 0) { if (used) *used = 0; return 0; }
    if (s[0] < 0x80) { if (used) *used = 1; return (unsigned int)s[0]; }
    if ((s[0] & 0xE0) == 0xC0 && len >= 2 && (s[1] & 0xC0) == 0x80) {
        cp = ((unsigned int)(s[0] & 0x1F) << 6) | (unsigned int)(s[1] & 0x3F);
        if (used) *used = 2;
        return cp;
    }
    if ((s[0] & 0xF0) == 0xE0 && len >= 3 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        cp = ((unsigned int)(s[0] & 0x0F) << 12) |
             ((unsigned int)(s[1] & 0x3F) << 6) |
             (unsigned int)(s[2] & 0x3F);
        if (used) *used = 3;
        return cp;
    }
    if ((s[0] & 0xF8) == 0xF0 && len >= 4 &&
        (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        cp = ((unsigned int)(s[0] & 0x07) << 18) |
             ((unsigned int)(s[1] & 0x3F) << 12) |
             ((unsigned int)(s[2] & 0x3F) << 6) |
             (unsigned int)(s[3] & 0x3F);
        if (used) *used = 4;
        return cp;
    }
    if (used) *used = 1;
    return (unsigned int)s[0];
}

static int utf8_encode_one(unsigned int cp, char *out, int out_cap) {
    if (!out || out_cap < 2) return 0;
    if (cp <= 0x7FU) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    } else if (cp <= 0x7FFU) {
        if (out_cap < 3) return 0;
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
        return 1;
    } else if (cp <= 0xFFFFU) {
        if (out_cap < 4) return 0;
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = '\0';
        return 1;
    } else if (cp <= 0x10FFFFU) {
        if (out_cap < 5) return 0;
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        out[4] = '\0';
        return 1;
    }
    return 0;
}

static int template_mentions(const HParams *hp, const char *needle) {
    if (!hp || !needle) return 0;
    return contains_nocase(hp->architecture, needle) ||
           contains_nocase(hp->tokenizer_pre, needle) ||
           contains_nocase(hp->chat_template, needle);
}

static Arch detect_architecture_name(const char *arch_name) {
    if (!arch_name || !arch_name[0]) return ARCH_LLAMA;
    if (strstr(arch_name, "gpt2")) return ARCH_GPT2;
    if (strstr(arch_name, "qwen3")) return ARCH_QWEN3;
    if (strstr(arch_name, "qwen2.5") || strstr(arch_name, "qwen25")) return ARCH_QWEN25;
    if (strstr(arch_name, "qwen2")) return ARCH_QWEN2;
    if (strstr(arch_name, "qwen")) return ARCH_QWEN2;
    if (strstr(arch_name, "gemma3")) return ARCH_GEMMA3;
    if (strstr(arch_name, "gemma2")) return ARCH_GEMMA2;
    if (strstr(arch_name, "gemma")) return ARCH_GEMMA;
    if (strstr(arch_name, "phi4")) return ARCH_PHI4;
    if (strstr(arch_name, "phi3")) return ARCH_PHI3;
    if (strstr(arch_name, "phi")) return ARCH_PHI;
    if (strstr(arch_name, "mistral")) return ARCH_MISTRAL;
    if (strstr(arch_name, "mixtral")) return ARCH_MIXTRAL;
    if (strstr(arch_name, "falcon")) return ARCH_FALCON;
    if (strstr(arch_name, "deepseek")) return ARCH_DEEPSEEK;
    if (strstr(arch_name, "granite")) return ARCH_GRANITE;
    if (strstr(arch_name, "yi")) return ARCH_YI;
    if (strstr(arch_name, "baichuan")) return ARCH_BAICHUAN;
    if (strstr(arch_name, "stablelm")) return ARCH_STABLELM;
    if (strstr(arch_name, "command-r")) return ARCH_COMMAND_R;
    if (strstr(arch_name, "codellama")) return ARCH_CODELLAMA;
    if (strstr(arch_name, "nemo")) return ARCH_NEMO;
    if (strstr(arch_name, "llama4")) return ARCH_LLAMA4;
    if (strstr(arch_name, "llama3")) return ARCH_LLAMA3;
    if (strstr(arch_name, "llama2")) return ARCH_LLAMA2;
    return ARCH_LLAMA;
}

static void free_gpt2_tokenizer_tables(void) {
    free(g_token_lookup); g_token_lookup = NULL; g_token_lookup_cap = 0;
    free(g_bpe_pairs); g_bpe_pairs = NULL; g_bpe_pair_cap = 0; g_bpe_pair_count = 0;
    memset(g_gpt2_byte_token, -1, sizeof(g_gpt2_byte_token));
    memset(g_gpt2_codepoint_to_byte, -1, sizeof(g_gpt2_codepoint_to_byte));
    g_tokenizer_is_gpt2 = 0;
}

static int build_token_lookup(void) {
    int cap, i;
    if (!g_vocab || g_vocab_n <= 0) return 0;
    cap = 1;
    while (cap < g_vocab_n * 2) cap <<= 1;
    g_token_lookup = (StrIdEntry*)malloc((size_t)cap * sizeof(StrIdEntry));
    if (!g_token_lookup) return 0;
    memset(g_token_lookup, 0, (size_t)cap * sizeof(StrIdEntry));
    g_token_lookup_cap = cap;
    for (i = 0; i < g_vocab_n; i++) {
        unsigned int h = hash_str(g_vocab[i].text) & (unsigned int)(cap - 1);
        while (g_token_lookup[h].key) h = (h + 1) & (cap - 1);
        g_token_lookup[h].key = g_vocab[i].text;
        g_token_lookup[h].id = i;
    }
    return 1;
}

static int lookup_token_id(const char *text) {
    unsigned int h;
    int mask;
    if (!g_token_lookup || g_token_lookup_cap <= 0 || !text) return -1;
    mask = g_token_lookup_cap - 1;
    h = hash_str(text) & (unsigned int)mask;
    for (;;) {
        if (!g_token_lookup[h].key) return -1;
        if (strcmp(g_token_lookup[h].key, text) == 0) return g_token_lookup[h].id;
        h = (h + 1) & mask;
    }
}

static int lookup_pair_rank(int left, int right, int *new_id) {
    unsigned int h;
    int mask;
    if (!g_bpe_pairs || g_bpe_pair_cap <= 0) return -1;
    mask = g_bpe_pair_cap - 1;
    h = hash_pair_int(left, right) & (unsigned int)mask;
    for (;;) {
        if (g_bpe_pairs[h].left < 0) return -1;
        if (g_bpe_pairs[h].left == left && g_bpe_pairs[h].right == right) {
            if (new_id) *new_id = g_bpe_pairs[h].new_id;
            return g_bpe_pairs[h].rank;
        }
        h = (h + 1) & mask;
    }
}

static void build_gpt2_byte_encoder(int byte_to_cp[256], int cp_to_byte[65536]) {
    int used[256];
    int i, n = 0;
    memset(used, 0, sizeof(used));
    for (i = 33; i <= 126; i++) { byte_to_cp[i] = i; used[i] = 1; }
    for (i = 161; i <= 172; i++) { byte_to_cp[i] = i; used[i] = 1; }
    for (i = 174; i <= 255; i++) { byte_to_cp[i] = i; used[i] = 1; }
    for (i = 0; i < 256; i++) {
        if (!used[i]) {
            byte_to_cp[i] = 256 + n;
            n++;
        }
    }
    memset(cp_to_byte, -1, 65536 * sizeof(int));
    for (i = 0; i < 256; i++) {
        int cp = byte_to_cp[i];
        if (cp >= 0 && cp < 65536) cp_to_byte[cp] = i;
    }
}

static int build_gpt2_tokenizer_tables(void) {
    int byte_to_cp[256];
    int i;
    int missing = 0;
    if (!g_vocab || g_vocab_n <= 0) return 0;
    if (!build_token_lookup()) return 0;
    build_gpt2_byte_encoder(byte_to_cp, g_gpt2_codepoint_to_byte);
    for (i = 0; i < 256; i++) {
        char tmp[8];
        int id;
        if (!utf8_encode_one((unsigned int)byte_to_cp[i], tmp, sizeof(tmp))) {
            g_gpt2_byte_token[i] = -1;
            missing++;
            continue;
        }
        id = lookup_token_id(tmp);
        if (id < 0) {
            g_gpt2_byte_token[i] = -1;
            missing++;
            continue;
        }
        g_gpt2_byte_token[i] = id;
    }
    if (missing > 0 && !g_clean_output) {
        printf("GPT2 tokenizer warning: %d byte tokens missing; using fallback byte mapping where needed\n", missing);
    }
    return 1;
}

static int build_gpt2_bpe_pairs_from_merges(const char **merge_texts, const int *merge_lens, int merge_n) {
    int cap, i;
    if (!merge_texts || merge_n <= 0) return 0;
    cap = 1;
    while (cap < merge_n * 2) cap <<= 1;
    g_bpe_pairs = (BpePairEntry*)malloc((size_t)cap * sizeof(BpePairEntry));
    if (!g_bpe_pairs) return 0;
    g_bpe_pair_cap = cap;
    for (i = 0; i < cap; i++) {
        g_bpe_pairs[i].left = -1;
        g_bpe_pairs[i].right = -1;
        g_bpe_pairs[i].rank = -1;
        g_bpe_pairs[i].new_id = -1;
    }
    for (i = 0; i < merge_n; i++) {
        const char *m = merge_texts[i];
        int mlen = merge_lens ? merge_lens[i] : (int)strlen(m);
        int sp = 0;
        char left[256];
        char right[256];
        char merged[512];
        int left_id, right_id, new_id;
        unsigned int h;
        int mask;
        while (sp < mlen && m[sp] != ' ') sp++;
        if (sp <= 0 || sp >= mlen) continue;
        if (sp >= (int)sizeof(left) || mlen - sp - 1 >= (int)sizeof(right)) continue;
        memcpy(left, m, (size_t)sp);
        left[sp] = '\0';
        memcpy(right, m + sp + 1, (size_t)(mlen - sp - 1));
        right[mlen - sp - 1] = '\0';
        left_id = lookup_token_id(left);
        right_id = lookup_token_id(right);
        if (left_id < 0 || right_id < 0) continue;
        if ((int)strlen(left) + (int)strlen(right) >= (int)sizeof(merged)) continue;
        strcpy(merged, left);
        strcat(merged, right);
        new_id = lookup_token_id(merged);
        if (new_id < 0) continue;
        mask = g_bpe_pair_cap - 1;
        h = hash_pair_int(left_id, right_id) & (unsigned int)mask;
        while (g_bpe_pairs[h].left >= 0) h = (h + 1) & mask;
        g_bpe_pairs[h].left = left_id;
        g_bpe_pairs[h].right = right_id;
        g_bpe_pairs[h].rank = i;
        g_bpe_pairs[h].new_id = new_id;
        g_bpe_pair_count++;
    }
    return g_bpe_pair_count > 0;
}

static int ascii_is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static int ascii_is_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static int ascii_is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static int match_gpt2_contraction(const char *text, int len) {
    if (len >= 4 &&
        text[0] == '\'' &&
        (text[1] == 'r' || text[1] == 'R') &&
        (text[2] == 'e' || text[2] == 'E')) return 3;
    if (len >= 4 &&
        text[0] == '\'' &&
        (text[1] == 'v' || text[1] == 'V') &&
        (text[2] == 'e' || text[2] == 'E')) return 3;
    if (len >= 4 &&
        text[0] == '\'' &&
        (text[1] == 'l' || text[1] == 'L') &&
        (text[2] == 'l' || text[2] == 'L')) return 3;
    if (len >= 3 &&
        text[0] == '\'' &&
        (text[1] == 's' || text[1] == 'S' ||
         text[1] == 't' || text[1] == 'T' ||
         text[1] == 'm' || text[1] == 'M' ||
         text[1] == 'd' || text[1] == 'D')) return 2;
    return 0;
}

static int match_special_token_text(const char *text, int len, int *token_id) {
    int end, tok;
    char tmp[128];
    if (!text || len <= 2 || text[0] != '<') return 0;
    for (end = 1; end < len && end < (int)sizeof(tmp) - 1; end++) {
        if (text[end] == '>') {
            memcpy(tmp, text, (size_t)(end + 1));
            tmp[end + 1] = '\0';
            tok = lookup_token_id(tmp);
            if (tok >= 0) {
                if (token_id) *token_id = tok;
                return end + 1;
            }
            break;
        }
    }
    return 0;
}

static int tokenize_gpt2_chunk(const char *text, int len, int *tokens, int max_tokens) {
    int i, n;
    int *ids;
    static int warned_missing_byte = 0;
    if (!text || len <= 0 || !tokens || max_tokens <= 0) return 0;
    ids = (int*)malloc((size_t)len * sizeof(int));
    if (!ids) return -1;
    n = 0;
    for (i = 0; i < len && n < max_tokens; i++) {
        unsigned char c = (unsigned char)text[i];
        int byte_id = g_gpt2_byte_token[c];
        if (byte_id < 0) byte_id = g_byte_token[c];
        if (byte_id < 0) {
            if (!warned_missing_byte) {
                fprintf(stderr, "Error: GPT2 tokenizer is missing byte token mappings; refusing to inject raw token IDs\n");
                warned_missing_byte = 1;
            }
            free(ids);
            return -1;
        }
        ids[n++] = byte_id;
    }
    if (g_bpe_pair_count > 0) {
        for (;;) {
            int best_rank = 0x7fffffff;
            int best_pos = -1;
            int best_new_id = -1;
            for (i = 0; i + 1 < n; i++) {
                int new_id;
                int rank = lookup_pair_rank(ids[i], ids[i + 1], &new_id);
                if (rank >= 0 && rank < best_rank) {
                    best_rank = rank;
                    best_pos = i;
                    best_new_id = new_id;
                }
            }
            if (best_pos < 0) break;
            ids[best_pos] = best_new_id;
            for (i = best_pos + 1; i + 1 < n; i++) ids[i] = ids[i + 1];
            n--;
        }
    }
    if (n > max_tokens) n = max_tokens;
    for (i = 0; i < n; i++) tokens[i] = ids[i];
    free(ids);
    return n;
}

static int tokenize_gpt2(const char *text, int *tokens, int max_tokens) {
    int len, pos, n;
    int needs_prefix_space = 0;
    const char *scan;
    if (!text || !tokens || max_tokens <= 0) return 0;
    if (!g_tokenizer_is_gpt2 || !g_vocab || !g_token_lookup) return -1;
    scan = text;
    while (*scan == '\r' || *scan == '\n' || *scan == '\t' || *scan == ' ') scan++;
    if (g_gpt2_add_space_prefix && *scan && *scan != '<' &&
        text[0] != ' ' && text[0] != '\t' && text[0] != '\r' && text[0] != '\n') {
        needs_prefix_space = 1;
    }
    if (needs_prefix_space) {
        char *tmp = (char*)malloc(strlen(text) + 2);
        int out_n;
        if (!tmp) return -1;
        tmp[0] = ' ';
        strcpy(tmp + 1, text);
        out_n = tokenize_gpt2(tmp, tokens, max_tokens);
        free(tmp);
        return out_n;
    }
    len = (int)strlen(text);
    pos = 0;
    n = 0;
    while (pos < len && n < max_tokens) {
        int special_id = -1;
        int slen = match_special_token_text(text + pos, len - pos, &special_id);
        if (slen > 0) {
            tokens[n++] = special_id;
            pos += slen;
            continue;
        }
        if (ascii_is_space((unsigned char)text[pos])) {
            if (pos + 1 < len && !ascii_is_space((unsigned char)text[pos + 1])) {
                int start = pos;
                int chunk_start = pos + 1;
                int chunk_end = chunk_start;
                int clen;
                if ((clen = match_gpt2_contraction(text + chunk_start, len - chunk_start)) > 0) {
                    chunk_end = chunk_start + clen;
                } else if (ascii_is_letter((unsigned char)text[chunk_start])) {
                    while (chunk_end < len && ascii_is_letter((unsigned char)text[chunk_end])) chunk_end++;
                } else if (ascii_is_digit((unsigned char)text[chunk_start])) {
                    while (chunk_end < len && ascii_is_digit((unsigned char)text[chunk_end])) chunk_end++;
                } else {
                    while (chunk_end < len &&
                           !ascii_is_space((unsigned char)text[chunk_end]) &&
                           !ascii_is_letter((unsigned char)text[chunk_end]) &&
                           !ascii_is_digit((unsigned char)text[chunk_end])) {
                        if (match_special_token_text(text + chunk_end, len - chunk_end, NULL) > 0) break;
                        chunk_end++;
                    }
                }
                clen = tokenize_gpt2_chunk(text + start, chunk_end - start, tokens + n, max_tokens - n);
                if (clen < 0) return clen;
                n += clen;
                pos = chunk_end;
                continue;
            } else {
                int start = pos;
                while (pos < len && ascii_is_space((unsigned char)text[pos])) pos++;
                {
                    int clen = tokenize_gpt2_chunk(text + start, pos - start, tokens + n, max_tokens - n);
                    if (clen < 0) return clen;
                    n += clen;
                }
                continue;
            }
        } else {
            int start = pos;
            int clen = match_gpt2_contraction(text + pos, len - pos);
            if (clen > 0) {
                pos += clen;
            } else if (ascii_is_letter((unsigned char)text[pos])) {
                while (pos < len && ascii_is_letter((unsigned char)text[pos])) pos++;
            } else if (ascii_is_digit((unsigned char)text[pos])) {
                while (pos < len && ascii_is_digit((unsigned char)text[pos])) pos++;
            } else {
                while (pos < len &&
                       !ascii_is_space((unsigned char)text[pos]) &&
                       !ascii_is_letter((unsigned char)text[pos]) &&
                       !ascii_is_digit((unsigned char)text[pos])) {
                    if (pos > start && match_special_token_text(text + pos, len - pos, NULL) > 0) break;
                    pos++;
                }
            }
            clen = tokenize_gpt2_chunk(text + start, pos - start, tokens + n, max_tokens - n);
            if (clen < 0) return clen;
            n += clen;
        }
    }
    return n;
}

static void detokenize_gpt2(int token) {
    const char *text;
    int pos, len;
    if (token == 0) return;
    if (!g_vocab || token < 0 || token >= g_vocab_n) return;
    text = g_vocab[token].text;
    len = g_vocab[token].len;
    if (!text || len <= 0) return;
    if (text[0] == '<') {
        fwrite(text, 1, (size_t)len, stdout);
        return;
    }
    pos = 0;
    while (pos < len) {
        int used = 0;
        unsigned int cp = utf8_decode_one((const unsigned char*)text + pos, len - pos, &used);
        int byte = -1;
        if (cp < 65536) byte = g_gpt2_codepoint_to_byte[cp];
        if (byte < 0) byte = (int)(unsigned char)text[pos];
        if (byte == '\n' || byte == '\r' || byte == '\t' || (byte >= 32 && byte < 127)) {
            putchar((char)byte);
        } else if (byte == ' ') {
            putchar(' ');
        }
        pos += used > 0 ? used : 1;
    }
}

static int load_tokenizer_from_gguf(GGUFContext *ctx) {
    u8 *p = ctx->base;
    u64 meta_count;
    u64 i;
    int want_gpt2;
    const char **merge_texts = NULL;
    int *merge_lens = NULL;
    char *merge_buf = NULL;
    u64 merge_count = 0;
    u64 merge_buf_size = 0;
    if (g_tok_buf) { free(g_tok_buf); g_tok_buf = NULL; free(g_vocab); g_vocab = NULL; }
    if (g_token_types) { free(g_token_types); g_token_types = NULL; }
    if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; }
    g_sorted_init = 0;
    free_gpt2_tokenizer_tables();
    reset_byte_token_map();
    g_gpt2_add_space_prefix = 0;
    if (memcmp(p, "GGUF", 4) != 0) return 0;
    p += 4;
    read_u32(&p); /* version */
    read_u64(&p); /* tensor_count */
    meta_count = read_u64(&p);
    want_gpt2 = contains_nocase(ctx->hp.architecture, "gpt2") || contains_nocase(ctx->hp.tokenizer_model, "gpt2");
    for (i = 0; i < meta_count; i++) {
        u64 keylen_full = read_u64(&p);
        u64 keylen = keylen_full;
        char key[256];
        u32 vtype;
        u64 len;
        u32 atype;
        u8 *start_of_value;
        if (keylen > 255) keylen = 255;
        memcpy(key, p, (size_t)keylen); key[keylen] = '\0'; p += keylen;
        p += keylen_full - keylen;
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
        if (!g_clean_output) printf("Loaded tokenizer from GGUF: %d tokens\n", g_vocab_n);
                rebuild_byte_token_map();
            }
        } else if (strcmp(key, "tokenizer.ggml.token_type") == 0 && vtype == 9) {
            atype = read_u32(&p);
            len = read_u64(&p);
            if (atype == 5 && len > 0 && len < 1000000) {
                u64 k;
                if (g_token_types) { free(g_token_types); g_token_types = NULL; }
                g_token_types = (int*)malloc(sizeof(int) * (size_t)len);
                if (!g_token_types) return 0;
                for (k = 0; k < len; k++) {
                    int tv;
                    memcpy(&tv, p, 4);
                    p += 4;
                    g_token_types[k] = tv;
                }
            } else {
                skip_value(vtype, &p);
            }
        } else if (want_gpt2 && strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == 9) {
            atype = read_u32(&p);
            len = read_u64(&p);
            if (atype == 8 && len > 0 && len < 1000000) {
                u8 *save_p = p;
                u64 k;
                merge_count = len;
                for (k = 0; k < len; k++) {
                    u64 slen = read_u64(&save_p);
                    merge_buf_size += slen + 1;
                    save_p += slen;
                }
                merge_texts = (const char**)malloc((size_t)len * sizeof(char*));
                merge_lens = (int*)malloc((size_t)len * sizeof(int));
                merge_buf = (char*)malloc((size_t)merge_buf_size);
                if (!merge_texts || !merge_lens || !merge_buf) {
                    free((void*)merge_texts);
                    free(merge_lens);
                    free(merge_buf);
                    return 0;
                }
                {
                    u64 off = 0;
                    for (k = 0; k < len; k++) {
                        u64 slen = read_u64(&p);
                        merge_texts[k] = merge_buf + off;
                        merge_lens[k] = (int)slen;
                        memcpy(merge_buf + off, p, (size_t)slen);
                        merge_buf[off + slen] = '\0';
                        off += slen + 1;
                        p += slen;
                    }
                }
            }
        } else {
            skip_value(vtype, &p);
        }
        (void)start_of_value;
    }
    if (g_vocab && want_gpt2) {
        g_tokenizer_is_gpt2 = 1;
        g_gpt2_add_space_prefix = ctx->hp.tokenizer_add_space_prefix;
        if (!build_gpt2_tokenizer_tables()) {
            free((void*)merge_texts);
            free(merge_lens);
            free(merge_buf);
            return 0;
        }
        if (merge_texts && merge_count > 0) {
            if (!build_gpt2_bpe_pairs_from_merges(merge_texts, merge_lens, (int)merge_count)) {
                free((void*)merge_texts);
                free(merge_lens);
                free(merge_buf);
                return 0;
            }
        }
        if (!g_clean_output) printf("Loaded GPT2 tokenizer from GGUF: %d tokens, %d merges\n", g_vocab_n, g_bpe_pair_count);
        free((void*)merge_texts);
        free(merge_lens);
        free(merge_buf);
        return 1;
    }
    free((void*)merge_texts);
    free(merge_lens);
    free(merge_buf);
    return g_vocab != NULL;
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
    u64 keylen, keylen_full;
    char key[256];
    memset(hp, 0, sizeof(HParams));
    hp->alignment = 32;
    hp->attention_layer_norm_rms_epsilon = 1e-6f;
    hp->rope_freq_base = 10000.0f;
    hp->attn_logit_softcapping = 0.0f;
    hp->final_logit_softcapping = 0.0f;
    strcpy(hp->architecture, "llama");
    for (i = 0; i < n_kv; i++) {
        keylen_full = read_u64(p);
        keylen = keylen_full;
        if (keylen > 255) keylen = 255;
        memcpy(key, *p, (size_t)keylen); key[keylen] = '\0'; *p += keylen;
        *p += keylen_full - keylen;
        vtype = read_u32(p);
        if (strcmp(key, "general.alignment") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->alignment)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "context_length") ||
                   meta_key_eq(key, "qwen2.", "context_length") ||
                   meta_key_eq(key, "qwen3.", "context_length") ||
                   meta_key_eq(key, "gpt2.", "context_length") ||
                   meta_key_eq(key, "gemma3.", "context_length") ||
                   strcmp(key, "context_length") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->context_length)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "embedding_length") ||
                   meta_key_eq(key, "qwen2.", "embedding_length") ||
                   meta_key_eq(key, "qwen3.", "embedding_length") ||
                   meta_key_eq(key, "gpt2.", "embedding_length") ||
                   meta_key_eq(key, "gemma3.", "embedding_length") ||
                   strcmp(key, "embedding_length") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->embedding_length)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "feed_forward_length") ||
                   meta_key_eq(key, "qwen2.", "feed_forward_length") ||
                   meta_key_eq(key, "qwen3.", "feed_forward_length") ||
                   meta_key_eq(key, "gpt2.", "feed_forward_length") ||
                   meta_key_eq(key, "gemma3.", "feed_forward_length") ||
                   strcmp(key, "feed_forward_length") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->feed_forward_length)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "block_count") ||
                   meta_key_eq(key, "qwen2.", "block_count") ||
                   meta_key_eq(key, "qwen3.", "block_count") ||
                   meta_key_eq(key, "gpt2.", "block_count") ||
                   meta_key_eq(key, "gemma3.", "block_count") ||
                   strcmp(key, "block_count") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->block_count)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.attention.", "head_count") ||
                   meta_key_eq(key, "qwen2.attention.", "head_count") ||
                   meta_key_eq(key, "qwen3.attention.", "head_count") ||
                   meta_key_eq(key, "gpt2.attention.", "head_count") ||
                   meta_key_eq(key, "gemma3.attention.", "head_count") ||
                   strcmp(key, "attention.head_count") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->attention_head_count)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.attention.", "head_count_kv") ||
                   meta_key_eq(key, "qwen2.attention.", "head_count_kv") ||
                   meta_key_eq(key, "qwen3.attention.", "head_count_kv") ||
                   meta_key_eq(key, "gpt2.attention.", "head_count_kv") ||
                   meta_key_eq(key, "gemma3.attention.", "head_count_kv") ||
                   strcmp(key, "attention.head_count_kv") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->attention_head_count_kv)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "gpt2.attention.", "layer_norm_epsilon") ||
                   strcmp(key, "attention.layer_norm_epsilon") == 0) {
            if (!read_meta_float_value(vtype, p, &hp->attention_layer_norm_rms_epsilon)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "gemma3.attention.", "key_length") ||
                   meta_key_eq(key, "qwen3.attention.", "key_length") ||
                   strcmp(key, "attention.key_length") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->attention_key_length)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "gemma3.attention.", "value_length") ||
                   meta_key_eq(key, "qwen3.attention.", "value_length") ||
                   strcmp(key, "attention.value_length") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->attention_value_length)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.attention.", "layer_norm_rms_epsilon") ||
                   meta_key_eq(key, "qwen2.attention.", "layer_norm_rms_epsilon") ||
                   meta_key_eq(key, "qwen3.attention.", "layer_norm_rms_epsilon") ||
                   meta_key_eq(key, "gemma3.attention.", "layer_norm_rms_epsilon") ||
                   strcmp(key, "attention.layer_norm_rms_epsilon") == 0) {
            if (!read_meta_float_value(vtype, p, &hp->attention_layer_norm_rms_epsilon)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.rope.", "dimension_count") ||
                   meta_key_eq(key, "qwen2.rope.", "dimension_count") ||
                   meta_key_eq(key, "qwen3.rope.", "dimension_count") ||
                   meta_key_eq(key, "gemma3.rope.", "dimension_count") ||
                   strcmp(key, "rope.dimension_count") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->rope_dimension_count)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.rope.", "freq_base") ||
                   meta_key_eq(key, "qwen2.rope.", "freq_base") ||
                   meta_key_eq(key, "qwen3.rope.", "freq_base") ||
                   meta_key_eq(key, "gemma3.rope.", "freq_base") ||
                   strcmp(key, "rope.freq_base") == 0) {
            if (!read_meta_float_value(vtype, p, &hp->rope_freq_base)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "attn_logit_softcapping") ||
                   meta_key_eq(key, "qwen2.", "attn_logit_softcapping") ||
                   meta_key_eq(key, "qwen3.", "attn_logit_softcapping") ||
                   meta_key_eq(key, "gemma2.", "attn_logit_softcapping") ||
                   meta_key_eq(key, "gemma3.", "attn_logit_softcapping") ||
                   strcmp(key, "attn_logit_softcapping") == 0) {
            if (!read_meta_float_value(vtype, p, &hp->attn_logit_softcapping)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "llama.", "final_logit_softcapping") ||
                   meta_key_eq(key, "qwen2.", "final_logit_softcapping") ||
                   meta_key_eq(key, "qwen3.", "final_logit_softcapping") ||
                   meta_key_eq(key, "gemma2.", "final_logit_softcapping") ||
                   meta_key_eq(key, "gemma3.", "final_logit_softcapping") ||
                   strcmp(key, "final_logit_softcapping") == 0) {
            if (!read_meta_float_value(vtype, p, &hp->final_logit_softcapping)) skip_value(vtype, p);
        } else if (meta_key_eq(key, "gemma3.attention.", "sliding_window") ||
                   strcmp(key, "attention.sliding_window") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->attention_sliding_window)) skip_value(vtype, p);
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->tokenizer_bos_token_id)) skip_value(vtype, p);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->tokenizer_eos_token_id)) skip_value(vtype, p);
        } else if (strcmp(key, "tokenizer.ggml.padding_token_id") == 0) {
            if (!read_meta_u32_value(vtype, p, &hp->tokenizer_padding_token_id)) skip_value(vtype, p);
        } else if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0) {
            if (!read_meta_bool_value(vtype, p, &hp->tokenizer_add_bos_token)) skip_value(vtype, p);
        } else if (strcmp(key, "tokenizer.ggml.add_space_prefix") == 0) {
            if (!read_meta_bool_value(vtype, p, &hp->tokenizer_add_space_prefix)) skip_value(vtype, p);
        } else if (strcmp(key, "general.architecture") == 0 && vtype == 8) {
            u64 slen_full = read_u64(p);
            u64 slen = slen_full;
            if (slen > 31) slen = 31;
            memcpy(hp->architecture, *p, (size_t)slen);
            hp->architecture[slen] = '\0';
            *p += slen;
            *p += slen_full - slen;
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && vtype == 8) {
            u64 slen_full = read_u64(p);
            u64 slen = slen_full;
            if (slen > 31) slen = 31;
            memcpy(hp->tokenizer_model, *p, (size_t)slen);
            hp->tokenizer_model[slen] = '\0';
            *p += slen;
            *p += slen_full - slen;
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && vtype == 8) {
            u64 slen_full = read_u64(p);
            u64 slen = slen_full;
            if (slen > 31) slen = 31;
            memcpy(hp->tokenizer_pre, *p, (size_t)slen);
            hp->tokenizer_pre[slen] = '\0';
            *p += slen;
            *p += slen_full - slen;
        } else if (strcmp(key, "tokenizer.chat_template") == 0 && vtype == 8) {
            u64 slen_full = read_u64(p);
            u64 slen = slen_full;
            if (slen > sizeof(hp->chat_template) - 1) slen = sizeof(hp->chat_template) - 1;
            memcpy(hp->chat_template, *p, (size_t)slen);
            hp->chat_template[slen] = '\0';
            *p += slen;
            *p += slen_full - slen;
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
        if (!is_legacy_windows()) {
            hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile != INVALID_HANDLE_VALUE) {
                sizeLow = GetFileSize(hFile, &sizeHigh);
                if (sizeLow != INVALID_FILE_SIZE || GetLastError() == NO_ERROR) {
                    file_size = ((u64)sizeHigh << 32) | (u64)sizeLow;
                    if (!g_clean_output) printf("Loading %s (%.1f MB) via memory map...\n", path, (double)file_size / (1024.0*1024.0));
                    hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
                    if (hMapping) {
                        ctx->base = (u8*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
                        if (ctx->base) {
                            ctx->hFile = hFile;
                            ctx->hMapping = hMapping;
                            ctx->size = (size_t)file_size;
                            goto loaded;
                        }
                        if (!g_clean_output) printf("Warning: MapViewOfFile failed (error %lu), falling back to fread\n", (unsigned long)GetLastError());
                        CloseHandle(hMapping);
                    }
                    else {
                        if (!g_clean_output) printf("Warning: CreateFileMapping failed (error %lu), falling back to fread\n", (unsigned long)GetLastError());
                    }
                }
                CloseHandle(hFile);
            }
        }
    }
#endif

    f = fopen(path, "rb");
    if (!f) { printf("Error: cannot open %s\n", path); free(ctx); return NULL; }
    fseek(f, 0, SEEK_END);
    file_size = (u64)ftell(f);
    fseek(f, 0, SEEK_SET);
    if (!g_clean_output) printf("Loading %s (%.1f MB) via fread...\n", path, (double)file_size / (1024.0*1024.0));
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
    if (read_u32(&p) != 3) { if (!g_clean_output) printf("Warning: GGUF version != 3\n"); }
    tensor_count = read_u64(&p);
    meta_count = read_u64(&p);
    if (tensor_count == 0 || tensor_count > 1000000) {
        printf("Error: invalid tensor count %llu\n", (unsigned long long)tensor_count);
        free(ctx->base);
        free(ctx);
        return NULL;
    }
    if (meta_count > 1000000) {
        printf("Error: invalid metadata count %llu\n", (unsigned long long)meta_count);
        free(ctx->base);
        free(ctx);
        return NULL;
    }

    parse_metadata(&p, meta_count, &ctx->hp);

    /* Tensor infos are not alignment-padded in GGUF. If parsing drifted, recover nearby. */
    if (!probe_tensor_section(ctx->base, ctx->size, (u64)(p - ctx->base), tensor_count)) {
        u8 *search_floor = p > ctx->base + 4096 ? p - 4096 : ctx->base + 16;
        u8 *search_ceil = p + 4096;
        u8 *file_end = ctx->base + ctx->size;
        u8 *found = NULL;
        u8 *s;
        if (search_ceil > file_end - 64) search_ceil = file_end - 64;
        for (s = search_floor; s < search_ceil; s++) {
            if (probe_tensor_section(ctx->base, ctx->size, (u64)(s - ctx->base), tensor_count)) {
                found = s;
                break;
            }
        }
        if (found) {
            if (!g_clean_output) {
                printf("Tensor section found at offset %u (parser was at %u)\n",
                       (unsigned int)(found - ctx->base), (unsigned int)(p - ctx->base));
            }
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
        u8 *file_end = ctx->base + ctx->size;
        u64 name_len = read_u64(&p);
        u64 copy_len = name_len;
        if (p > file_end || name_len > (u64)(file_end - p)) {
            printf("Error: truncated tensor name at index %llu\n", (unsigned long long)i);
            free(ctx->tensors);
            free(ctx->base);
            free(ctx);
            return NULL;
        }
        if (copy_len > 79) copy_len = 79;
        memcpy(t->name, p, (size_t)copy_len); t->name[copy_len] = '\0';
        p += name_len;
        if (p + 4 > file_end) {
            printf("Error: truncated tensor dims at index %llu\n", (unsigned long long)i);
            free(ctx->tensors);
            free(ctx->base);
            free(ctx);
            return NULL;
        }
        t->n_dims = read_u32(&p);
        if (t->n_dims > 4) {
            printf("Error: invalid tensor dims count %u at index %llu\n", t->n_dims, (unsigned long long)i);
            free(ctx->tensors);
            free(ctx->base);
            free(ctx);
            return NULL;
        }
        for (j = 0; j < t->n_dims; j++) t->dims[j] = read_u64(&p);
        for (; j < 4; j++) t->dims[j] = 1;
        if (p + 4 > file_end) {
            printf("Error: truncated tensor type at index %llu\n", (unsigned long long)i);
            free(ctx->tensors);
            free(ctx->base);
            free(ctx);
            return NULL;
        }
        t->type = read_u32(&p);
        if (p + 8 > file_end) {
            printf("Error: truncated tensor offset at index %llu\n", (unsigned long long)i);
            free(ctx->tensors);
            free(ctx->base);
            free(ctx);
            return NULL;
        }
        t->offset = read_u64(&p);
        t->size_bytes = 0;
        t->force_transpose = 0;
    }

    pos_after_tensors = (u64)(p - ctx->base);
    ctx->data_offset = align_u64(pos_after_tensors, ctx->hp.alignment);
    for (i = 0; i < tensor_count; i++) {
        Tensor *t = &ctx->tensors[i];
        u64 data_pos = ctx->data_offset + t->offset;
        u64 next_offset = (u64)ctx->size - ctx->data_offset;
        u64 j;
        u64 guessed = tensor_storage_bytes(t);
        for (j = 0; j < tensor_count; j++) {
            if (ctx->tensors[j].offset > t->offset && ctx->tensors[j].offset < next_offset) {
                next_offset = ctx->tensors[j].offset;
            }
        }
        if (data_pos > (u64)ctx->size || t->offset > next_offset) {
            printf("Error: tensor %s out of range (offset=%llu size=%llu data_offset=%llu file=%llu)\n",
                   t->name,
                   (unsigned long long)t->offset,
                   (unsigned long long)guessed,
                   (unsigned long long)ctx->data_offset,
                   (unsigned long long)ctx->size);
            gguf_free(ctx);
            return NULL;
        }
        t->size_bytes = next_offset - t->offset;
        t->data = ctx->base + (size_t)data_pos;
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

static int tensor_width(const Tensor *t, int model_dim) {
    int a, b;
    if (!t || t->n_dims < 2) return model_dim;
    a = (int)t->dims[0];
    b = (int)t->dims[1];
    if (a == model_dim && b != model_dim) return b;
    if (b == model_dim && a != model_dim) return a;
    if (a > b) return a;
    return b;
}

static int gcd_int(int a, int b) {
    int t;
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        t = a % b;
        a = b;
        b = t;
    }
    return a;
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
    if (x <= 0.0f) return 0.0f;
    return 1.0f / (float)sqrt((double)x);
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

static void rmsnorm_ex(float *x, float *w, int n, float eps, int add_unit_offset) {
    int i;
    float ss = 0.0f, scale, inv_n;
    for (i = 0; i + 3 < n; i += 4) {
        ss += x[i]*x[i] + x[i+1]*x[i+1] + x[i+2]*x[i+2] + x[i+3]*x[i+3];
    }
    for (; i < n; i++) ss += x[i] * x[i];
    inv_n = 1.0f / (float)n;
    scale = fast_rsqrt(ss * inv_n + eps);
    for (i = 0; i + 3 < n; i += 4) {
        float w0 = add_unit_offset ? (1.0f + w[i]) : w[i];
        float w1 = add_unit_offset ? (1.0f + w[i + 1]) : w[i + 1];
        float w2 = add_unit_offset ? (1.0f + w[i + 2]) : w[i + 2];
        float w3 = add_unit_offset ? (1.0f + w[i + 3]) : w[i + 3];
        x[i]     = x[i]     * scale * w0;
        x[i + 1] = x[i + 1] * scale * w1;
        x[i + 2] = x[i + 2] * scale * w2;
        x[i + 3] = x[i + 3] * scale * w3;
    }
    for (; i < n; i++) {
        float wi = add_unit_offset ? (1.0f + w[i]) : w[i];
        x[i] = x[i] * scale * wi;
    }
}

static void layernorm(float *x, const float *w, const float *b, int n, float eps) {
    int i;
    float mean = 0.0f, var = 0.0f, inv_std;
    if (n <= 0) return;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    for (i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    inv_std = 1.0f / (float)sqrt((double)(var + eps));
    for (i = 0; i < n; i++) {
        float y = (x[i] - mean) * inv_std;
        if (w) y *= w[i];
        if (b) y += b[i];
        x[i] = y;
    }
}

static float gelu(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + (float)tanh((double)c));
}

static void dequantize_row(const Tensor *t, float *out, int row, int d);

static void add_bias(float *y, const Tensor *b, int n) {
    int i;
    float *tmp;
    if (!y || !b || !b->data || n <= 0) return;
    if (b->type == 0) {
        const float *src = (const float*)b->data;
        for (i = 0; i < n; i++) y[i] += src[i];
    } else {
        tmp = (float*)malloc((size_t)n * sizeof(float));
        if (!tmp) return;
        dequantize_row(b, tmp, 0, n);
        for (i = 0; i < n; i++) y[i] += tmp[i];
        free(tmp);
    }
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

static void rope_1d_half(float *vec, int n, int pos, const float *rope_cos, const float *rope_sin) {
    int half = n / 2;
    int k;
    const float *rc = rope_cos + pos * half;
    const float *rs = rope_sin + pos * half;
    for (k = 0; k < half; k++) {
        float c = rc[k];
        float s = rs[k];
        float x1 = vec[k];
        float x2 = vec[k + half];
        vec[k] = x1 * c - x2 * s;
        vec[k + half] = x2 * c + x1 * s;
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
    return x / (1.0f + (float)exp((double)(-x)));
}

static float softcap(float x, float cap) {
    if (cap <= 0.0f) return x;
    return cap * (float)tanh((double)(x / cap));
}

static int vector_has_nan(const float *v, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (!(v[i] == v[i])) return 1;
    }
    return 0;
}

static int vector_has_nonfinite(const float *v, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (v[i] != v[i] || v[i] > 1e30f || v[i] < -1e30f) return 1;
    }
    return 0;
}

static int first_nonfinite_index(const float *v, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (v[i] != v[i] || v[i] > 1e30f || v[i] < -1e30f) return i;
    }
    return -1;
}

static float clamp_nonfinite(float v) {
    if (v != v) return 0.0f;
    if (v > 1e30f) return 0.0f;
    if (v < -1e30f) return 0.0f;
    return v;
}

/* llama.cpp-style top-k / top-p / min-p + temperature sampling */
#define MAX_TOPK 256
static int sample_topk(float *logits, int n, float temp, int top_k, float top_p, float min_p, int ban_token) {
    int i, k;
    float maxv, sum, r, cdf;
    for (i = 0; i < n; i++) {
        if (token_is_blocked_for_generation(i)) logits[i] = -1e30f;
    }
    if (!g_allow_eog_sampling) {
        for (i = 0; i < g_eog_token_count; i++) {
            int tok = g_eog_tokens[i];
            if (tok >= 0 && tok < n) logits[tok] = -1e30f;
        }
    }
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
static float tensor_value_at(const Tensor *t, int row, int col, int row_width);
static void dequantize_row(const Tensor *t, float *out, int row, int d);
static void matvec(const Tensor *t, const float *x, float *y, int n, int d, float *dq_row);

static int tensor_row_count_for_width(const Tensor *t, int row_width) {
    int a, b;
    if (!t || t->n_dims == 0) return 0;
    a = (int)t->dims[0];
    b = (int)t->dims[1];
    if (t->n_dims == 1) {
        return row_width == a ? 1 : a;
    }
    if (a <= 0 || b <= 0 || row_width <= 0) return 0;
    if (a == row_width) return b;
    if (b == row_width) return a;
    return 0;
}

static int cache_tensor_f32_inplace(Tensor *t) {
    int n, d, i;
    float *buf;
    float *row;
    size_t bytes;

    if (!t || t->type == 0) return 1;
    d = (int)t->dims[0];
    n = (int)t->dims[1];
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

static int cache_vector_tensor_f32_inplace(Tensor *t) {
    int n;
    int i;
    float *buf;
    if (!t || t->type == 0) return 1;
    if (t->n_dims == 0) return 0;
    if (t->n_dims == 1) {
        n = (int)t->dims[0];
        if (n <= 0) return 0;
        buf = (float*)malloc((size_t)n * sizeof(float));
        if (!buf) return 0;
        if (t->type == 1) {
            const u16 *src = (const u16*)t->data;
            for (i = 0; i < n; i++) buf[i] = fp16_to_fp32(src[i]);
        } else {
            for (i = 0; i < n; i++) buf[i] = tensor_value_at(t, i, 0, 1);
        }
        t->data = buf;
        t->type = 0;
        return 1;
    }
    if (t->n_dims >= 2) {
        if (t->dims[1] == 1 && t->dims[0] > 0) {
            n = (int)t->dims[0];
            buf = (float*)malloc((size_t)n * sizeof(float));
            if (!buf) return 0;
            if (t->type == 1) {
                const u16 *src = (const u16*)t->data;
                for (i = 0; i < n; i++) buf[i] = fp16_to_fp32(src[i]);
            } else {
                for (i = 0; i < n; i++) buf[i] = tensor_value_at(t, i, 0, 1);
            }
            t->data = buf;
            t->type = 0;
            return 1;
        }
        if (t->dims[0] == 1 && t->dims[1] > 0) {
            n = (int)t->dims[1];
            buf = (float*)malloc((size_t)n * sizeof(float));
            if (!buf) return 0;
            if (t->type == 1) {
                const u16 *src = (const u16*)t->data;
                for (i = 0; i < n; i++) buf[i] = fp16_to_fp32(src[i]);
            } else {
                for (i = 0; i < n; i++) buf[i] = tensor_value_at(t, 0, i, n);
            }
            t->data = buf;
            t->type = 0;
            return 1;
        }
    }
    return 0;
}

static float *cache_tensor_logical_f32(const Tensor *t, int n, int d) {
    int i, j;
    float *buf;
    float *row;
    size_t bytes;
    int raw_n;

    if (!t || n <= 0 || d <= 0) return NULL;
    if (!((int)t->dims[0] == d && (int)t->dims[1] == n) &&
        !((int)t->dims[0] == n && (int)t->dims[1] == d)) {
        return NULL;
    }
    bytes = (size_t)n * (size_t)d * sizeof(float);
    buf = (float*)malloc(bytes);
    raw_n = (int)t->dims[0];
    if ((int)t->dims[1] > raw_n) raw_n = (int)t->dims[1];
    row = (float*)malloc((size_t)raw_n * sizeof(float));
    if (!buf || !row) {
        free(buf);
        free(row);
        return NULL;
    }
    if ((int)t->dims[0] == d && (int)t->dims[1] == n) {
        for (i = 0; i < n; i++) {
            dequantize_row(t, row, i, d);
            memcpy(buf + (size_t)i * (size_t)d, row, (size_t)d * sizeof(float));
        }
    } else {
        for (j = 0; j < d; j++) {
            dequantize_row(t, row, j, n);
            for (i = 0; i < n; i++) {
                buf[(size_t)i * (size_t)d + (size_t)j] = row[i];
            }
        }
    }
    free(row);
    return buf;
}

static int cache_tensor_f32_logical_inplace(Tensor *t, int n, int d) {
    float *buf;
    if (!t) return 0;
    if (t->type == 0 && (int)t->dims[0] == d && (int)t->dims[1] == n) return 1;
    buf = cache_tensor_logical_f32(t, n, d);
    if (!buf) return 0;
    t->data = buf;
    t->type = 0;
    t->dims[0] = (u64)d;
    t->dims[1] = (u64)n;
    return 1;
}

static int cache_tensor_f32_force_transpose_inplace(Tensor *t, int n, int d) {
    int i, j;
    float *buf;
    float *row;
    size_t bytes;

    if (!t || n <= 0 || d <= 0) return 0;
    bytes = (size_t)n * (size_t)d * sizeof(float);
    buf = (float*)malloc(bytes);
    row = (float*)malloc((size_t)n * sizeof(float));
    if (!buf || !row) {
        free(buf);
        free(row);
        return 0;
    }
    for (j = 0; j < d; j++) {
        dequantize_row(t, row, j, n);
        for (i = 0; i < n; i++) {
            buf[(size_t)i * (size_t)d + (size_t)j] = row[i];
        }
    }
    free(row);
    t->data = buf;
    t->type = 0;
    t->dims[0] = (u64)d;
    t->dims[1] = (u64)n;
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

static size_t tensor_f32_cache_bytes(const Tensor *t) {
    if (!t || t->dims[0] == 0 || t->dims[1] == 0) return 0;
    return (size_t)t->dims[0] * (size_t)t->dims[1] * sizeof(float);
}

static int try_cache_tensor_f32_inplace(Tensor *t, size_t *cache_used, size_t cache_budget) {
    size_t bytes;
    if (!t || t->type == 0 || !cache_used) return 0;
    bytes = tensor_f32_cache_bytes(t);
    if (bytes == 0 || bytes > cache_budget || *cache_used > cache_budget - bytes) return 0;
    if (!cache_tensor_f32_inplace(t)) return -1;
    *cache_used += bytes;
    return 1;
}

static int try_cache_tensor_f32_logical_inplace(Tensor *t, int n, int d, size_t *cache_used, size_t cache_budget) {
    size_t bytes;
    if (!t || t->type == 0 || !cache_used) return 0;
    bytes = (size_t)n * (size_t)d * sizeof(float);
    if (bytes == 0 || bytes > cache_budget || *cache_used > cache_budget - bytes) return 0;
    if (!cache_tensor_f32_logical_inplace(t, n, d)) return -1;
    *cache_used += bytes;
    return 1;
}

static int try_cache_tensor_f32_force_transpose_inplace(Tensor *t, int n, int d, size_t *cache_used, size_t cache_budget) {
    size_t bytes;
    if (!t || t->type == 0 || !cache_used) return 0;
    bytes = (size_t)n * (size_t)d * sizeof(float);
    if (bytes == 0 || bytes > cache_budget || *cache_used > cache_budget - bytes) return 0;
    if (!cache_tensor_f32_force_transpose_inplace(t, n, d)) return -1;
    *cache_used += bytes;
    return 1;
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

#define Q6K_BLOCK_BYTES 210
#pragma pack(pop)

static int blocks_256(int d);

static u64 tensor_storage_bytes(const Tensor *t) {
    u64 rows, cols;
    u64 bytes_a, bytes_b;
    if (!t) return 0;
    rows = t->dims[1];
    cols = t->dims[0];
    if (t->n_dims == 1) cols = 1;
    if (rows == 0 || cols == 0) return 0;
    switch (t->type) {
        case 0: return rows * cols * 4ULL;
        case 1: return rows * cols * 2ULL;
        case 2:
            bytes_a = rows * (u64)(((int)cols + 31) / 32) * (u64)sizeof(BlockQ4);
            bytes_b = cols * (u64)(((int)rows + 31) / 32) * (u64)sizeof(BlockQ4);
            return bytes_a < bytes_b ? bytes_a : bytes_b;
        case 6:
            bytes_a = rows * (u64)(((int)cols + 31) / 32) * (u64)sizeof(BlockQ5_0);
            bytes_b = cols * (u64)(((int)rows + 31) / 32) * (u64)sizeof(BlockQ5_0);
            return bytes_a < bytes_b ? bytes_a : bytes_b;
        case 7:
            bytes_a = rows * (u64)(((int)cols + 31) / 32) * (u64)sizeof(BlockQ5_1);
            bytes_b = cols * (u64)(((int)rows + 31) / 32) * (u64)sizeof(BlockQ5_1);
            return bytes_a < bytes_b ? bytes_a : bytes_b;
        case 8:
            bytes_a = rows * (u64)(((int)cols + 31) / 32) * (u64)sizeof(BlockQ8);
            bytes_b = cols * (u64)(((int)rows + 31) / 32) * (u64)sizeof(BlockQ8);
            return bytes_a < bytes_b ? bytes_a : bytes_b;
        case 12:
            bytes_a = rows * (u64)blocks_256((int)cols) * (u64)sizeof(BlockQ4K);
            bytes_b = cols * (u64)blocks_256((int)rows) * (u64)sizeof(BlockQ4K);
            return bytes_a < bytes_b ? bytes_a : bytes_b;
        case 13:
            bytes_a = rows * (u64)blocks_256((int)cols) * (u64)sizeof(BlockQ5K);
            bytes_b = cols * (u64)blocks_256((int)rows) * (u64)sizeof(BlockQ5K);
            return bytes_a < bytes_b ? bytes_a : bytes_b;
        case 14:
            bytes_a = rows * (u64)blocks_256((int)cols) * (u64)Q6K_BLOCK_BYTES;
            bytes_b = cols * (u64)blocks_256((int)rows) * (u64)Q6K_BLOCK_BYTES;
            return bytes_a < bytes_b ? bytes_a : bytes_b;
        default: return 0;
    }
}

static void get_scale_min_k4(int j, const u8 *q, u8 *d, u8 *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static float tensor_value_at(const Tensor *t, int row, int col, int row_width) {
    if (!t || row < 0 || col < 0 || row_width <= 0) return 0.0f;
    if (t->type == 0) {
        const float *data = (const float*)t->data;
        return data[(size_t)row * (size_t)row_width + (size_t)col];
    } else if (t->type == 1) {
        const u16 *data = (const u16*)t->data;
        return fp16_to_fp32(data[(size_t)row * (size_t)row_width + (size_t)col]);
    } else if (t->type == 8) {
        const BlockQ8 *blk;
        int nb = row_width / 32;
        int b = col / 32;
        int o = col % 32;
        if (nb <= 0) return 0.0f;
        blk = (const BlockQ8*)t->data + row * nb + b;
        return fp16_to_fp32(blk->d) * (float)blk->qs[o];
    } else if (t->type == 12) {
        const BlockQ4K *blk;
        int nb = blocks_256(row_width);
        int b = col / 256;
        int o = col % 256;
        int chunk = o / 64;
        int within = o % 64;
        u8 sc, m;
        float d_all, min_all;
        const u8 *q;
        if (nb <= 0) return 0.0f;
        blk = (const BlockQ4K*)t->data + row * nb + b;
        get_scale_min_k4(chunk * 2 + (within >= 32 ? 1 : 0), blk->scales, &sc, &m);
        d_all = fp16_to_fp32(blk->d[0]);
        min_all = fp16_to_fp32(blk->d[1]);
        q = blk->qs + chunk * 32;
        if (within < 32) return clamp_nonfinite(d_all * (float)sc * (float)(q[within] & 0x0F) - min_all * (float)m);
        return clamp_nonfinite(d_all * (float)sc * (float)(q[within - 32] >> 4) - min_all * (float)m);
    } else if (t->type == 13) {
        const BlockQ5K *blk;
        int nb = blocks_256(row_width);
        int b = col / 256;
        int o = col % 256;
        int chunk = o / 64;
        int within = o % 64;
        u8 sc, m;
        float d_all, min_all;
        const u8 *ql;
        const u8 *qh;
        if (nb <= 0) return 0.0f;
        blk = (const BlockQ5K*)t->data + row * nb + b;
        get_scale_min_k4(chunk * 2 + (within >= 32 ? 1 : 0), blk->scales, &sc, &m);
        d_all = fp16_to_fp32(blk->d[0]);
        min_all = fp16_to_fp32(blk->d[1]);
        ql = blk->qs + chunk * 32;
        qh = blk->qh + chunk * 8;
        if (within < 32) {
            int bit = within;
            int hi = (qh[bit >> 3] >> (bit & 7)) & 1;
            int qv = (ql[within] & 0x0F) + (hi ? 16 : 0);
            return clamp_nonfinite(d_all * (float)sc * (float)qv - min_all * (float)m);
        } else {
            int idx = within - 32;
            int bit = idx + 32;
            int hi = (qh[bit >> 3] >> (bit & 7)) & 1;
            int qv = (ql[idx] >> 4) + (hi ? 16 : 0);
            return clamp_nonfinite(d_all * (float)sc * (float)qv - min_all * (float)m);
        }
    } else if (t->type == 14) {
        const u8 *blk;
        int nb = blocks_256(row_width);
        int b = col / 256;
        int o = col % 256;
        int half = o / 128;
        int within = o % 128;
        const u8 *ql;
        const u8 *qh;
        const s8 *sc;
        float d_all;
        int l, is, qv;
        if (nb <= 0) return 0.0f;
        blk = (const u8*)t->data + ((size_t)row * (size_t)nb + (size_t)b) * (size_t)Q6K_BLOCK_BYTES;
        ql = blk + half * 64;
        qh = blk + 128 + half * 32;
        sc = (const s8*)(blk + 192) + half * 8;
        {
            u16 d_fp16;
            memcpy(&d_fp16, blk + 208, sizeof(d_fp16));
            d_all = fp16_to_fp32(d_fp16);
        }
        if (within < 32) {
            l = within;
            is = l / 16;
            qv = (int)((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            return clamp_nonfinite(d_all * (float)sc[is + 0] * (float)qv);
        } else if (within < 64) {
            l = within - 32;
            is = l / 16;
            qv = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            return clamp_nonfinite(d_all * (float)sc[is + 2] * (float)qv);
        } else if (within < 96) {
            l = within - 64;
            is = l / 16;
            qv = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            return clamp_nonfinite(d_all * (float)sc[is + 4] * (float)qv);
        } else {
            l = within - 96;
            is = l / 16;
            qv = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            return clamp_nonfinite(d_all * (float)sc[is + 6] * (float)qv);
        }
    }
    return 0.0f;
}

static void load_transposed_embedding_fast(const Tensor *t, float *x, int token, int dim, int vocab_size) {
    int j;
    for (j = 0; j < dim; j++) x[j] = tensor_value_at(t, j, token, vocab_size);
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
                    sum += clamp_nonfinite(d1 * (float)(q[l] & 0xF) - m1) * xb[jj + l];
                }
                if (rem > 32) {
                    int rem2 = rem - 32;
                    if (rem2 > 32) rem2 = 32;
                    for (l = 0; l < rem2; l++) {
                        sum += clamp_nonfinite(d2 * (float)(q[l] >> 4) - m2) * xb[jj + 32 + l];
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
                    sum += clamp_nonfinite(d1 * (float)x0 - m1) * xb[jj + l + 0];
                    if (l + 32 < rem) {
                        sum += clamp_nonfinite(d2 * (float)x1 - m2) * xb[jj + l + 32];
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
    const u8 *wbytes = (const u8*)w;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            int block_base = b * 256;
            int limit = d - block_base;
            const u8 *blk = wbytes + ((size_t)i * (size_t)nb + (size_t)b) * (size_t)Q6K_BLOCK_BYTES;
            u16 d_fp16;
            float d_all;
            const u8 *ql = blk + 0;
            const u8 *qh = blk + 128;
            const s8 *sc = (const s8*)(blk + 192);
            const float *xb = x + block_base;
            memcpy(&d_fp16, blk + 208, sizeof(d_fp16));
            d_all = fp16_to_fp32(d_fp16);
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
                    if (l + 0 < rem) sum += clamp_nonfinite(d_all * (float)sc[is + 0] * (float)q1) * xb[l + 0];
                    if (l + 32 < rem) sum += clamp_nonfinite(d_all * (float)sc[is + 2] * (float)q2) * xb[l + 32];
                    if (l + 64 < rem) sum += clamp_nonfinite(d_all * (float)sc[is + 4] * (float)q3) * xb[l + 64];
                    if (l + 96 < rem) sum += clamp_nonfinite(d_all * (float)sc[is + 6] * (float)q4) * xb[l + 96];
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

static int is_legacy_windows(void) {
#ifdef _WIN32
    DWORD ver;
    DWORD major;
    DWORD minor;
    if (g_legacy_windows >= 0) return g_legacy_windows;
    ver = GetVersion();
    major = (DWORD)(LOBYTE(LOWORD(ver)));
    minor = (DWORD)(HIBYTE(LOWORD(ver)));
    g_legacy_windows = (major < 5 || (major == 5 && minor == 0)) ? 1 : 0;
    return g_legacy_windows;
#else
    return 0;
#endif
}

static size_t legacy_cache_budget_bytes(size_t default_budget) {
#ifdef _WIN32
    MEMORYSTATUS ms;
    size_t phys;
    size_t budget;

    if (!is_legacy_windows()) return default_budget;
    memset(&ms, 0, sizeof(ms));
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatus(&ms);
    phys = (size_t)ms.dwTotalPhys;
    if (phys == 0) {
        if (default_budget > (512U * 1024U * 1024U)) return 512U * 1024U * 1024U;
        return default_budget;
    }

    budget = phys / 2;
    if (budget < (256U * 1024U * 1024U)) budget = 256U * 1024U * 1024U;
    if (budget > (1280U * 1024U * 1024U)) budget = 1280U * 1024U * 1024U;
    if (budget > default_budget) budget = default_budget;
    return budget;
#else
    return default_budget;
#endif
}

static int default_thread_count(void) {
#ifdef _WIN32
    DWORD_PTR processMask = 0, systemMask = 0;
    int n = 0;
    int sys_n = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
        while (processMask) {
            n += (int)(processMask & 1);
            processMask >>= 1;
        }
    }
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        if (si.dwNumberOfProcessors > 0) {
            sys_n = (int)si.dwNumberOfProcessors;
        }
    }
    if (sys_n > n) n = sys_n;
    {
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        if (k32) {
            typedef WORD (WINAPI *PFN_GetActiveProcessorCount)(WORD);
            PFN_GetActiveProcessorCount pGetActiveProcessorCount;
            pGetActiveProcessorCount = (PFN_GetActiveProcessorCount)GetProcAddress(k32, "GetActiveProcessorCount");
            if (pGetActiveProcessorCount) {
                int active_n = (int)pGetActiveProcessorCount((WORD)-1);
                if (active_n > n) n = active_n;
            }
        }
    }
    if (n > 8) n = 8;
    if (n > 0) return n;
#endif
    return 1;
}

static int blocks_256(int d);

static int blocks_256(int d) {
    return (d + 255) / 256;
}

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
    if (is_legacy_windows()) {
        work = (u64)n * (u64)d;
        if (n < 96) return 0;
        return work >= 80000ULL;
    }
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
    if (thread_count > MATVEC_POOL_MAX_THREADS) thread_count = MATVEC_POOL_MAX_THREADS;
    if (thread_count > n) thread_count = n;
    if (is_legacy_windows()) {
        if (thread_count > 8) thread_count = 8;
        if (work < 250000ULL) thread_count = 1;
        else if (work < 600000ULL && thread_count > 4) thread_count = 4;
        if (thread_count < 1) thread_count = 1;
        return thread_count;
    }
    /* Keep tiny matrices serial. Use fewer threads for mid-sized work and
       allow the full pool once the matrix is large enough to amortize the
       synchronization cost. */
    if (work < 250000ULL) thread_count = 1;
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
                sub_t.dims[0] = (u64)job.d;
                sub_t.dims[1] = (u64)rows;
                sub_t.size_bytes = (u64)rows * (u64)job.row_bytes;
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

    if (is_legacy_windows()) {
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
    if (thread_count > MATVEC_POOL_MAX_THREADS) thread_count = MATVEC_POOL_MAX_THREADS;
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

    /* Only parallelize standard GGML row access: dims[0] = d, dims[1] = n */
    if (!((int)t->dims[0] == d && (int)t->dims[1] == n)) {
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
    int row_count;
    if (!t || !out || d <= 0) return;
    row_count = tensor_row_count_for_width(t, d);
    if (row < 0 || (row_count > 0 && row >= row_count)) {
        memset(out, 0, (size_t)d * sizeof(float));
        return;
    }
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
                    out[j + jj + l] = clamp_nonfinite(d1 * (float)(q[l] & 0xF) - m1);
                }
                if (rem > 32) {
                    int rem2 = rem - 32;
                    if (rem2 > 32) rem2 = 32;
                    for (l = 0; l < rem2; l++) {
                        out[j + jj + 32 + l] = clamp_nonfinite(d2 * (float)(q[l] >> 4) - m2);
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
                    out[j + jj + l + 0] = clamp_nonfinite(d1 * (float)x0 - m1);
                    if (l + 32 < rem) {
                        out[j + jj + l + 32] = clamp_nonfinite(d2 * (float)x1 - m2);
                    }
                }
                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
            blk++;
        }
    } else if (t->type == 14) {
        int nb = blocks_256(d);
        const u8 *row_base = (const u8*)t->data + (size_t)row * (size_t)nb * (size_t)Q6K_BLOCK_BYTES;
        int j;
        if (t->size_bytes && ((size_t)(row + 1) * (size_t)nb * (size_t)Q6K_BLOCK_BYTES > (size_t)t->size_bytes)) {
            memset(out, 0, (size_t)d * sizeof(float));
            return;
        }
        for (j = 0; j < d; j += 256) {
            int limit = d - j;
            if (limit > 256) limit = 256;
            const u8 *blk = row_base + (size_t)(j / 256) * (size_t)Q6K_BLOCK_BYTES;
            u16 d_fp16;
            float d_all;
            const u8 *ql = blk + 0;
            const u8 *qh = blk + 128;
            const s8 *sc = (const s8*)(blk + 192);
            float *y = out + j;
            int n2;
            memcpy(&d_fp16, blk + 208, sizeof(d_fp16));
            d_all = fp16_to_fp32(d_fp16);
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
                    if (l + 0 < rem)  y[l + 0]  = clamp_nonfinite(d_all * (float)sc[is + 0] * (float)q1);
                    if (l + 32 < rem) y[l + 32] = clamp_nonfinite(d_all * (float)sc[is + 2] * (float)q2);
                    if (l + 64 < rem) y[l + 64] = clamp_nonfinite(d_all * (float)sc[is + 4] * (float)q3);
                    if (l + 96 < rem) y[l + 96] = clamp_nonfinite(d_all * (float)sc[is + 6] * (float)q4);
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
    int use_transposed = 0;
    /* GGML/GGUF 2D tensors use dims[0] = input width (d), dims[1] = output rows (n).
       Only use the transpose fallback when the tensor is clearly stored as [n, d]. */
    if (t && t->force_transpose) use_transposed = 1;
    else if ((int)t->dims[0] == n && (int)t->dims[1] == d && n != d) use_transposed = 1;
    if (use_transposed) {
        float *dq_row = (float*)malloc((size_t)n * sizeof(float));
        int j, v;
        if (!dq_row) {
            memset(y, 0, (size_t)n * sizeof(float));
            return;
        }
        memset(y, 0, (size_t)n * sizeof(float));
        for (j = 0; j < d; j++) {
            float xj = x[j];
            dequantize_row(t, dq_row, j, n);
            for (v = 0; v + 3 < n; v += 4) {
                y[v]     += xj * dq_row[v];
                y[v + 1] += xj * dq_row[v + 1];
                y[v + 2] += xj * dq_row[v + 2];
                y[v + 3] += xj * dq_row[v + 3];
            }
            for (; v < n; v++) y[v] += xj * dq_row[v];
        }
        free(dq_row);
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
    int use_transposed = 0;
    if (t && t->force_transpose) use_transposed = 1;
    else if ((int)t->dims[0] == n && (int)t->dims[1] == d && n != d) use_transposed = 1;
    if (should_parallelize_matvec(n, d)) {
        if ((int)t->dims[0] == d && (int)t->dims[1] == n) {
            if (use_transposed) {
                matvec_transposed_parallel(t, x, y, n, d);
                return;
            }
            matvec_parallel(t, x, y, n, d);
            return;
        }
        if (use_transposed) {
            matvec_transposed_parallel(t, x, y, n, d);
            return;
        }
    }
    if (use_transposed) {
        matvec_no_parallel(t, x, y, n, d, dq_row);
        return;
    }
    if ((int)t->dims[0] == d && (int)t->dims[1] == n) {
        matvec_no_parallel(t, x, y, n, d, dq_row);
        return;
    }
    matvec_no_parallel(t, x, y, n, d, dq_row);
}
/* --- Model builder --- */

static int build_model(GGUFContext *ctx, Model *m, HParams *hp, int n_heads_user, int n_kv_user, int hidden_user, int seq_user) {
    int i;
    Tensor *t;
    size_t cache_budget = (sizeof(void*) <= 4) ? (1536U * 1024U * 1024U) : (2048U * 1024U * 1024U);
    size_t cache_used = 0;
    int prefer_output_cache = 0;
    int legacy_windows = is_legacy_windows();

    if (legacy_windows) {
        cache_budget = legacy_cache_budget_bytes(cache_budget);
    }

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
    /* GGML 2D tensors are stored with dims[0] as row width and dims[1] as row count.
       token_embd.weight is therefore already logical [vocab rows][dim cols] when
       dims[0] = dim and dims[1] = vocab. Only treat it as transposed when the larger
       vocab axis is in dims[0]. */
    if (t->dims[0] <= t->dims[1]) {
        m->dim = (int)t->dims[0];
        m->vocab_size = (int)t->dims[1];
        m->tok_embd_transposed = 0;
    } else {
        m->vocab_size = (int)t->dims[0];
        m->dim = (int)t->dims[1];
        m->tok_embd_transposed = 1;
        printf("Note: token_embd.weight is transposed ([%u,%u])\n", (unsigned int)t->dims[0], (unsigned int)t->dims[1]);
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

    /* Architecture detection */
    m->arch = detect_architecture_name(hp->architecture);

    if (m->arch == ARCH_QWEN2 || m->arch == ARCH_QWEN3 || m->arch == ARCH_QWEN25) {
        prefer_output_cache = 1;
        if (cache_budget < (1280U * 1024U * 1024U)) {
            cache_budget = 1280U * 1024U * 1024U;
        }
        if (legacy_windows && cache_budget > (256U * 1024U * 1024U)) {
            cache_budget = 256U * 1024U * 1024U;
        }
    }

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

    {
        if (m->arch == ARCH_GPT2) {
            Tensor *q0 = find_tensor_f(ctx, "blk.%d.attn_qkv.weight", 0);
            if (!q0) q0 = find_tensor(ctx, "model.layers.0.attn.c_attn.weight");
            m->q_dim = tensor_width(q0, m->dim);
            m->kv_dim = m->dim;
        } else {
            Tensor *q0 = find_tensor_f(ctx, "blk.%d.attn_q.weight", 0);
            Tensor *k0 = find_tensor_f(ctx, "blk.%d.attn_k.weight", 0);
            Tensor *v0 = find_tensor_f(ctx, "blk.%d.attn_v.weight", 0);
            if (!q0) q0 = find_tensor(ctx, "model.layers.0.self_attn.q_proj.weight");
            if (!k0) k0 = find_tensor(ctx, "model.layers.0.self_attn.k_proj.weight");
            if (!v0) v0 = find_tensor(ctx, "model.layers.0.self_attn.v_proj.weight");
            m->q_dim = tensor_width(q0, m->dim);
            m->kv_dim = tensor_width(k0 ? k0 : v0, m->dim);
            if (m->kv_dim <= 0) m->kv_dim = m->dim;
        }
    }

    if (hp->attention_head_count > 0) m->n_heads = (int)hp->attention_head_count;
    else if (n_heads_user > 0) m->n_heads = n_heads_user;
    else if (m->arch == ARCH_GPT2 && m->dim > 0) {
        m->n_heads = 12;
    } else if (m->q_dim > 0 && m->kv_dim > 0) {
        int hd = gcd_int(m->q_dim, m->kv_dim);
        if (hd > 0) {
            m->head_dim = hd;
            m->n_heads = m->q_dim / hd;
        }
    }
    if (m->n_heads == 0) {
        fprintf(stderr, "Need --n_heads (not in GGUF metadata)\n");
        return 1;
    }
    if (m->arch == ARCH_GPT2) {
        m->head_dim = m->dim / m->n_heads;
    } else if (m->head_dim == 0 && hp->attention_key_length > 0) m->head_dim = (int)hp->attention_key_length;
    if (m->arch != ARCH_GPT2 && m->head_dim == 0 && m->q_dim > 0 && m->n_heads > 0 && (m->q_dim % m->n_heads) == 0) {
        m->head_dim = m->q_dim / m->n_heads;
    }
    if (m->arch != ARCH_GPT2 && m->head_dim == 0 && m->q_dim > 0 && m->kv_dim > 0) {
        int hd = gcd_int(m->q_dim, m->kv_dim);
        if (hd > 0) m->head_dim = hd;
    }
    if (m->head_dim == 0) m->head_dim = m->dim / m->n_heads;
    m->rope_dim = (hp->rope_dimension_count > 0 && (int)hp->rope_dimension_count <= m->head_dim)
                ? (int)hp->rope_dimension_count
                : m->head_dim;
    m->rsqrt_head_dim = 1.0f / (float)sqrt((double)m->head_dim);

    if (m->arch == ARCH_GPT2) m->n_kv_heads = m->n_heads;
    else if (hp->attention_head_count_kv > 0) m->n_kv_heads = (int)hp->attention_head_count_kv;
    else if (n_kv_user > 0) m->n_kv_heads = n_kv_user;
    else if (m->kv_dim > 0 && m->head_dim > 0) m->n_kv_heads = m->kv_dim / m->head_dim;
    if (m->n_kv_heads == 0) m->n_kv_heads = m->n_heads;

    if (hidden_user > 0) m->hidden_dim = hidden_user;
    else if (hp->feed_forward_length > 0) m->hidden_dim = (int)hp->feed_forward_length;
    else {
        if (m->arch == ARCH_GPT2) {
            t = find_tensor_f(ctx, "blk.%d.ffn_up.weight", 0);
            if (!t) t = find_tensor(ctx, "model.layers.0.mlp.c_fc.weight");
        } else {
            t = find_tensor_f(ctx, "blk.%d.ffn_gate.weight", 0);
            if (!t) t = find_tensor_f(ctx, "model.layers.%d.mlp.gate_proj.weight", 0);
        }
        if (t) m->hidden_dim = (int)t->dims[0];
    }
    if (m->hidden_dim == 0) { fprintf(stderr, "Need --hidden (cannot determine)\n"); return 1; }

    if (hp->attention_sliding_window > 0) m->attention_sliding_window = (int)hp->attention_sliding_window;
    else m->attention_sliding_window = 1024;

    if (!legacy_windows && m->tok_embd_transposed && m->arch != ARCH_GPT2 && m->arch != ARCH_GEMMA3 && !prefer_output_cache) {
        size_t cache_bytes = (size_t)m->vocab_size * (size_t)m->dim * sizeof(float);
        if (cache_bytes <= cache_budget - cache_used) {
            if (!g_clean_output) printf("Caching transposed token embeddings in F32 for speed...\n");
            m->cached_embd = cache_transposed_tensor_f32(m->tok_embd, m->dim, m->vocab_size);
            if (!m->cached_embd) {
                fprintf(stderr, "Failed to allocate embedding cache (%u MB)\n",
                        (unsigned int)(cache_bytes / (1024U * 1024U)));
                return 1;
            }
            cache_used += cache_bytes;
            if (m->output == m->tok_embd) {
                m->cached_output = m->cached_embd;
            }
            if (!g_clean_output) printf("Embedding cache ready.\n");
        } else {
            if (!g_clean_output) {
                printf("Skipping embedding cache (%u MB needed, %u MB budget left)\n",
                       (unsigned int)(cache_bytes / (1024U * 1024U)),
                       (unsigned int)((cache_budget - cache_used) / (1024U * 1024U)));
            }
        }
    }

    if (!legacy_windows &&
        (m->arch == ARCH_QWEN2 || m->arch == ARCH_QWEN3 || m->arch == ARCH_QWEN25) && !m->cached_embd) {
        size_t cache_bytes = (size_t)m->vocab_size * (size_t)m->dim * sizeof(float);
        if (cache_bytes <= cache_budget - cache_used) {
            if (!g_clean_output) printf("Caching Qwen token embeddings in logical F32...\n");
            m->cached_embd = cache_tensor_logical_f32(m->tok_embd, m->vocab_size, m->dim);
            if (!m->cached_embd) {
                fprintf(stderr, "Failed to allocate Qwen embedding cache (%u MB)\n",
                        (unsigned int)(cache_bytes / (1024U * 1024U)));
                return 1;
            }
            m->tok_embd->data = m->cached_embd;
            m->tok_embd->type = 0;
            m->tok_embd->dims[0] = (u64)m->dim;
            m->tok_embd->dims[1] = (u64)m->vocab_size;
            cache_used += cache_bytes;
            if (m->output == m->tok_embd) {
                m->cached_output = m->cached_embd;
            }
            if (!g_clean_output) printf("Qwen embedding cache ready.\n");
        } else {
            if (!g_clean_output) {
                printf("Skipping Qwen embedding cache (%u MB needed, %u MB budget left)\n",
                       (unsigned int)(cache_bytes / (1024U * 1024U)),
                       (unsigned int)((cache_budget - cache_used) / (1024U * 1024U)));
            }
        }
    }

    if (!legacy_windows &&
        (m->arch == ARCH_QWEN2 || m->arch == ARCH_QWEN3 || m->arch == ARCH_QWEN25) &&
        !m->cached_output && m->output && m->output != m->tok_embd && m->output->type != 0) {
        size_t cache_bytes = (size_t)m->vocab_size * (size_t)m->dim * sizeof(float);
        if (cache_bytes <= cache_budget - cache_used) {
            if (!g_clean_output) printf("Caching Qwen output head in logical F32...\n");
            m->cached_output = cache_tensor_logical_f32(m->output, m->vocab_size, m->dim);
            if (!m->cached_output) {
                fprintf(stderr, "Failed to allocate Qwen output cache (%u MB)\n",
                        (unsigned int)(cache_bytes / (1024U * 1024U)));
                return 1;
            }
            m->output->data = m->cached_output;
            m->output->type = 0;
            m->output->dims[0] = (u64)m->dim;
            m->output->dims[1] = (u64)m->vocab_size;
            cache_used += cache_bytes;
            if (!g_clean_output) printf("Qwen output cache ready.\n");
        } else {
            if (!g_clean_output) {
                printf("Skipping Qwen output cache (%u MB needed, %u MB budget left)\n",
                       (unsigned int)(cache_bytes / (1024U * 1024U)),
                       (unsigned int)((cache_budget - cache_used) / (1024U * 1024U)));
            }
        }
    }

    if (m->arch == ARCH_GEMMA3 &&
        !m->cached_embd &&
        m->output == m->tok_embd &&
        m->tok_embd &&
        m->tok_embd->type != 0) {
        size_t cache_bytes = tensor_f32_cache_bytes(m->tok_embd);
        if (cache_bytes <= cache_budget - cache_used) {
            if (!g_clean_output) printf("Caching Gemma3 token/output matrix in F32 for speed...\n");
            if (!cache_tensor_f32_inplace(m->tok_embd)) {
                fprintf(stderr, "Warning: failed to cache Gemma3 token/output matrix (%u MB)\n",
                        (unsigned int)(cache_bytes / (1024U * 1024U)));
            } else {
                m->cached_embd = (float*)m->tok_embd->data;
                m->cached_output = m->cached_embd;
                cache_used += cache_bytes;
                if (!g_clean_output) printf("Gemma3 token/output cache ready.\n");
            }
        } else {
            if (!g_clean_output) {
                printf("Skipping Gemma3 token/output cache (%u MB needed, %u MB budget left)\n",
                       (unsigned int)(cache_bytes / (1024U * 1024U)),
                       (unsigned int)((cache_budget - cache_used) / (1024U * 1024U)));
            }
        }
    }

    if (seq_user > 0) m->max_seq_len = seq_user;
    else if (hp->context_length > 0) m->max_seq_len = (int)hp->context_length;
    else m->max_seq_len = 2048;

    {
        int safe_seq = effective_seq_limit(m->max_seq_len, m->n_layers, m->n_kv_heads, m->head_dim);
        if (safe_seq < m->max_seq_len) {
            if (!g_clean_output) {
                printf("WARNING: capping seq_len %d -> %d to keep KV cache under 256MB\n",
                       m->max_seq_len, safe_seq);
            }
            m->max_seq_len = safe_seq;
        }
    }

    m->norm_eps = hp->attention_layer_norm_rms_epsilon;
    if (m->norm_eps == 0.0f) m->norm_eps = 1e-5f;
    m->rope_theta = hp->rope_freq_base;
    if (m->rope_theta == 0.0f) m->rope_theta = 10000.0f;
    m->attn_logit_softcapping = hp->attn_logit_softcapping;
    m->final_logit_softcapping = hp->final_logit_softcapping;

    if (m->arch != ARCH_GPT2) {
        int half = m->rope_dim / 2;
        int p, k;
        float local_theta = m->rope_theta;
        float global_theta = m->rope_theta;
        float *cos_tab = (float*)malloc((size_t)m->max_seq_len * half * sizeof(float));
        float *sin_tab = (float*)malloc((size_t)m->max_seq_len * half * sizeof(float));
        if (!cos_tab || !sin_tab) {
            fprintf(stderr, "Failed to allocate RoPE tables\n");
            return 1;
        }
        for (p = 0; p < m->max_seq_len; p++) {
            for (k = 0; k < half; k++) {
                float freq = 1.0f / (float)pow((double)local_theta, (double)(k * 2) / (double)m->rope_dim);
                float val = (float)p * freq;
                cos_tab[p * half + k] = (float)cos((double)val);
                sin_tab[p * half + k] = (float)sin((double)val);
            }
        }
        m->rope_cos = cos_tab;
        m->rope_sin = sin_tab;
        if (m->arch == ARCH_GEMMA3 && global_theta != local_theta) {
            float *gcos = (float*)malloc((size_t)m->max_seq_len * half * sizeof(float));
            float *gsin = (float*)malloc((size_t)m->max_seq_len * half * sizeof(float));
            if (!gcos || !gsin) {
                fprintf(stderr, "Failed to allocate Gemma 3 global RoPE tables\n");
                return 1;
            }
            for (p = 0; p < m->max_seq_len; p++) {
                for (k = 0; k < half; k++) {
                    float freq = 1.0f / (float)pow((double)global_theta, (double)(k * 2) / (double)m->rope_dim);
                    float val = (float)p * freq;
                    gcos[p * half + k] = (float)cos((double)val);
                    gsin[p * half + k] = (float)sin((double)val);
                }
            }
            m->rope_cos_global = gcos;
            m->rope_sin_global = gsin;
        }
    }

    m->attn_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_norm_bias = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_q_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_k_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_q = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_q_bias = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_k = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_v = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_o = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->attn_o_bias = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_norm_bias = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->post_attn_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->post_ffn_norm = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_gate = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_gate_bias = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_up = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_down = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    m->ffn_down_bias = (Tensor**)malloc(sizeof(Tensor*) * m->n_layers);
    if (!m->attn_norm || !m->attn_norm_bias || !m->attn_q_norm || !m->attn_k_norm || !m->attn_q || !m->attn_q_bias || !m->attn_k || !m->attn_v || !m->attn_o || !m->attn_o_bias ||
        !m->ffn_norm || !m->ffn_norm_bias || !m->post_attn_norm || !m->post_ffn_norm || !m->ffn_gate || !m->ffn_gate_bias || !m->ffn_up || !m->ffn_down || !m->ffn_down_bias) {
        fprintf(stderr, "Failed to allocate layer tensor pointers (out of memory?)\n");
        return 1;
    }

    if (m->arch == ARCH_GPT2) {
        int l;
        int pos_rows;
        m->pos_embd = find_tensor(ctx, "position_embd.weight");
        if (!m->pos_embd) m->pos_embd = find_tensor(ctx, "model.wpe.weight");
        if (!m->pos_embd) {
            fprintf(stderr, "Missing GPT2 position embeddings\n");
            list_tensors(ctx);
            return 1;
        }
        if (!m->output_norm) {
            fprintf(stderr, "Missing GPT2 output norm\n");
            return 1;
        }
        m->output_norm_bias = find_tensor(ctx, "output_norm.bias");
        if (!m->output_norm_bias) m->output_norm_bias = find_tensor(ctx, "ln_f.bias");

        if (!g_clean_output) printf("Caching GPT2 embeddings in logical row-major F32...\n");
        m->cached_embd = cache_tensor_logical_f32(m->tok_embd, m->vocab_size, m->dim);
        if (!m->cached_embd) {
            fprintf(stderr, "Failed to cache GPT2 token embeddings\n");
            return 1;
        }
        pos_rows = (int)((m->pos_embd->dims[0] == (u64)m->dim) ? m->pos_embd->dims[1] : m->pos_embd->dims[0]);
        m->cached_pos_embd = cache_tensor_logical_f32(m->pos_embd, pos_rows, m->dim);
        if (!m->cached_pos_embd) {
            fprintf(stderr, "Failed to cache GPT2 position embeddings\n");
            return 1;
        }
        if (m->output && !cache_tensor_f32_logical_inplace(m->output, m->vocab_size, m->dim)) {
            fprintf(stderr, "Failed to cache GPT2 output head\n");
            return 1;
        }
        if (!g_clean_output) printf("GPT2 embedding/output cache ready.\n");

        for (l = 0; l < m->n_layers; l++) {
            char name[128];
            m->attn_norm[l] = find_tensor_f(ctx, "blk.%d.attn_norm.weight", l);
            m->attn_norm_bias[l] = find_tensor_f(ctx, "blk.%d.attn_norm.bias", l);
            m->attn_q[l] = find_tensor_f(ctx, "blk.%d.attn_qkv.weight", l);
            m->attn_q_bias[l] = find_tensor_f(ctx, "blk.%d.attn_qkv.bias", l);
            m->attn_o[l] = find_tensor_f(ctx, "blk.%d.attn_output.weight", l);
            m->attn_o_bias[l] = find_tensor_f(ctx, "blk.%d.attn_output.bias", l);
            m->ffn_norm[l] = find_tensor_f(ctx, "blk.%d.ffn_norm.weight", l);
            m->ffn_norm_bias[l] = find_tensor_f(ctx, "blk.%d.ffn_norm.bias", l);
            m->ffn_gate[l] = find_tensor_f(ctx, "blk.%d.ffn_up.weight", l);
            m->ffn_gate_bias[l] = find_tensor_f(ctx, "blk.%d.ffn_up.bias", l);
            m->ffn_down[l] = find_tensor_f(ctx, "blk.%d.ffn_down.weight", l);
            m->ffn_down_bias[l] = find_tensor_f(ctx, "blk.%d.ffn_down.bias", l);
            if (!m->attn_norm[l]) { sprintf(name, "h.%d.ln_1.weight", l); m->attn_norm[l] = find_tensor(ctx, name); }
            if (!m->attn_norm_bias[l]) { sprintf(name, "h.%d.ln_1.bias", l); m->attn_norm_bias[l] = find_tensor(ctx, name); }
            if (!m->attn_q[l]) { sprintf(name, "h.%d.attn.c_attn.weight", l); m->attn_q[l] = find_tensor(ctx, name); }
            if (!m->attn_q_bias[l]) { sprintf(name, "h.%d.attn.c_attn.bias", l); m->attn_q_bias[l] = find_tensor(ctx, name); }
            if (!m->attn_o[l]) { sprintf(name, "h.%d.attn.c_proj.weight", l); m->attn_o[l] = find_tensor(ctx, name); }
            if (!m->attn_o_bias[l]) { sprintf(name, "h.%d.attn.c_proj.bias", l); m->attn_o_bias[l] = find_tensor(ctx, name); }
            if (!m->ffn_norm[l]) { sprintf(name, "h.%d.ln_2.weight", l); m->ffn_norm[l] = find_tensor(ctx, name); }
            if (!m->ffn_norm_bias[l]) { sprintf(name, "h.%d.ln_2.bias", l); m->ffn_norm_bias[l] = find_tensor(ctx, name); }
            if (!m->ffn_gate[l]) { sprintf(name, "h.%d.mlp.c_fc.weight", l); m->ffn_gate[l] = find_tensor(ctx, name); }
            if (!m->ffn_gate_bias[l]) { sprintf(name, "h.%d.mlp.c_fc.bias", l); m->ffn_gate_bias[l] = find_tensor(ctx, name); }
            if (!m->ffn_down[l]) { sprintf(name, "h.%d.mlp.c_proj.weight", l); m->ffn_down[l] = find_tensor(ctx, name); }
            if (!m->ffn_down_bias[l]) { sprintf(name, "h.%d.mlp.c_proj.bias", l); m->ffn_down_bias[l] = find_tensor(ctx, name); }
            if (!m->attn_norm[l] || !m->attn_q[l] || !m->attn_o[l] || !m->ffn_norm[l] || !m->ffn_gate[l] || !m->ffn_down[l]) {
                fprintf(stderr, "Missing GPT2 tensors for layer %d\n", l);
                return 1;
            }
            if (!cache_tensor_f32_logical_inplace(m->attn_q[l], m->q_dim, m->dim)) {
                fprintf(stderr, "Failed to cache %s\n", m->attn_q[l]->name);
                return 1;
            }
            if (!cache_tensor_f32_logical_inplace(m->attn_o[l], m->dim, m->dim)) {
                fprintf(stderr, "Failed to cache %s\n", m->attn_o[l]->name);
                return 1;
            }
            if (!cache_tensor_f32_logical_inplace(m->ffn_gate[l], m->hidden_dim, m->dim)) {
                fprintf(stderr, "Failed to cache %s\n", m->ffn_gate[l]->name);
                return 1;
            }
            if (!cache_tensor_f32_logical_inplace(m->ffn_down[l], m->dim, m->hidden_dim)) {
                fprintf(stderr, "Failed to cache %s\n", m->ffn_down[l]->name);
                return 1;
            }
        }
        if (!g_clean_output) printf("Caching hot projection weights in F32 for speed...\n");
        for (l = 0; l < m->n_layers; l++) {
            (void)l;
        }
        if (!g_clean_output) {
            printf("Hot weights cache pass complete.\n");
            printf("Model: dim=%d layers=%d heads=%d head_dim=%d kv_heads=%d hidden=%d vocab=%d seq=%d arch=%s\n",
                   m->dim, m->n_layers, m->n_heads, m->head_dim, m->n_kv_heads, m->hidden_dim, m->vocab_size, m->max_seq_len,
                   hp->architecture);
        }
        return 0;
    }

    for (i = 0; i < m->n_layers; i++) {
        char name[128];
        m->attn_norm[i] = find_tensor_f(ctx, "blk.%d.attn_norm.weight", i);
        m->attn_q_norm[i] = find_tensor_f(ctx, "blk.%d.attn_q_norm.weight", i);
        m->attn_k_norm[i] = find_tensor_f(ctx, "blk.%d.attn_k_norm.weight", i);
        m->attn_q[i]    = find_tensor_f(ctx, "blk.%d.attn_q.weight", i);
        m->attn_k[i]    = find_tensor_f(ctx, "blk.%d.attn_k.weight", i);
        m->attn_v[i]    = find_tensor_f(ctx, "blk.%d.attn_v.weight", i);
        m->attn_o[i]    = find_tensor_f(ctx, "blk.%d.attn_output.weight", i);
        m->ffn_norm[i]  = find_tensor_f(ctx, "blk.%d.ffn_norm.weight", i);
        m->post_attn_norm[i] = find_tensor_f(ctx, "blk.%d.post_attention_norm.weight", i);
        m->post_ffn_norm[i]  = find_tensor_f(ctx, "blk.%d.post_ffw_norm.weight", i);
        m->ffn_gate[i]  = find_tensor_f(ctx, "blk.%d.ffn_gate.weight", i);
        m->ffn_up[i]    = find_tensor_f(ctx, "blk.%d.ffn_up.weight", i);
        m->ffn_down[i]  = find_tensor_f(ctx, "blk.%d.ffn_down.weight", i);

        if (!m->attn_norm[i]) { sprintf(name, "model.layers.%d.input_layernorm.weight", i); m->attn_norm[i] = find_tensor(ctx, name); }
        if (!m->attn_q[i]) { sprintf(name, "model.layers.%d.self_attn.q_proj.weight", i); m->attn_q[i] = find_tensor(ctx, name); }
        if (!m->attn_k[i]) { sprintf(name, "model.layers.%d.self_attn.k_proj.weight", i); m->attn_k[i] = find_tensor(ctx, name); }
        if (!m->attn_v[i]) { sprintf(name, "model.layers.%d.self_attn.v_proj.weight", i); m->attn_v[i] = find_tensor(ctx, name); }
        if (!m->attn_o[i]) { sprintf(name, "model.layers.%d.self_attn.o_proj.weight", i); m->attn_o[i] = find_tensor(ctx, name); }
        if (!m->ffn_norm[i]) { sprintf(name, "model.layers.%d.post_attention_layernorm.weight", i); m->ffn_norm[i] = find_tensor(ctx, name); }
        if (!m->post_attn_norm[i]) { sprintf(name, "model.layers.%d.post_attention_norm.weight", i); m->post_attn_norm[i] = find_tensor(ctx, name); }
        if (!m->post_ffn_norm[i]) { sprintf(name, "model.layers.%d.post_ffw_norm.weight", i); m->post_ffn_norm[i] = find_tensor(ctx, name); }
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

    if (m->output_norm && m->output_norm->type != 0) {
        if (!cache_vector_tensor_f32_inplace(m->output_norm)) {
            fprintf(stderr, "Failed to cache %s\n", m->output_norm->name);
            return 1;
        }
    }
    for (i = 0; i < m->n_layers; i++) {
        if (m->attn_norm[i] && m->attn_norm[i]->type != 0 && !cache_vector_tensor_f32_inplace(m->attn_norm[i])) {
            fprintf(stderr, "Failed to cache %s\n", m->attn_norm[i]->name);
            return 1;
        }
        if (m->ffn_norm[i] && m->ffn_norm[i]->type != 0 && !cache_vector_tensor_f32_inplace(m->ffn_norm[i])) {
            fprintf(stderr, "Failed to cache %s\n", m->ffn_norm[i]->name);
            return 1;
        }
        if (m->post_attn_norm[i] && m->post_attn_norm[i]->type != 0 && !cache_vector_tensor_f32_inplace(m->post_attn_norm[i])) {
            fprintf(stderr, "Failed to cache %s\n", m->post_attn_norm[i]->name);
            return 1;
        }
        if (m->post_ffn_norm[i] && m->post_ffn_norm[i]->type != 0 && !cache_vector_tensor_f32_inplace(m->post_ffn_norm[i])) {
            fprintf(stderr, "Failed to cache %s\n", m->post_ffn_norm[i]->name);
            return 1;
        }
        if (m->attn_q_norm[i] && m->attn_q_norm[i]->type != 0 && !cache_vector_tensor_f32_inplace(m->attn_q_norm[i])) {
            fprintf(stderr, "Failed to cache %s\n", m->attn_q_norm[i]->name);
            return 1;
        }
        if (m->attn_k_norm[i] && m->attn_k_norm[i]->type != 0 && !cache_vector_tensor_f32_inplace(m->attn_k_norm[i])) {
            fprintf(stderr, "Failed to cache %s\n", m->attn_k_norm[i]->name);
            return 1;
        }
    }

    if (!legacy_windows &&
        !prefer_output_cache &&
        !m->cached_output && m->output && m->output != m->tok_embd &&
        m->output->dims[1] > m->output->dims[0] && m->output->type != 0) {
        size_t cache_bytes = (size_t)m->output->dims[0] * (size_t)m->output->dims[1] * sizeof(float);
        if (cache_bytes <= cache_budget - cache_used) {
            if (!g_clean_output) printf("Caching transposed output head in F32 for speed...\n");
            m->cached_output = cache_transposed_tensor_f32(m->output, (int)m->output->dims[0], (int)m->output->dims[1]);
            if (!m->cached_output) {
                fprintf(stderr, "Warning: failed to cache output head (%u MB)\n",
                        (unsigned int)(cache_bytes / (1024U * 1024U)));
            } else {
                cache_used += cache_bytes;
                if (!g_clean_output) printf("Output cache ready.\n");
            }
        } else {
            if (!g_clean_output) {
                printf("Skipping output cache (%u MB needed, %u MB budget left)\n",
                       (unsigned int)(cache_bytes / (1024U * 1024U)),
                       (unsigned int)((cache_budget - cache_used) / (1024U * 1024U)));
            }
        }
    }

    {
        int l;
        int cached_count = 0;
        int skipped_count = 0;
        int failed_count = 0;
        size_t hot_bytes_needed = 0;
        int force_transpose_square_kv =
            (m->arch == ARCH_QWEN2 || m->arch == ARCH_QWEN3 || m->arch == ARCH_QWEN25) &&
            m->attn_q[0] &&
            (int)m->attn_q[0]->dims[0] == m->dim &&
            (int)m->attn_q[0]->dims[1] == m->q_dim &&
            m->q_dim != m->dim;

        for (l = 0; l < m->n_layers; l++) {
            if (m->attn_q[l] && m->attn_q[l]->type != 0) hot_bytes_needed += tensor_f32_cache_bytes(m->attn_q[l]);
            if (m->attn_k[l] && m->attn_k[l]->type != 0) hot_bytes_needed += tensor_f32_cache_bytes(m->attn_k[l]);
            if (m->attn_v[l] && m->attn_v[l]->type != 0) hot_bytes_needed += tensor_f32_cache_bytes(m->attn_v[l]);
            if (m->attn_o[l] && m->attn_o[l]->type != 0) hot_bytes_needed += tensor_f32_cache_bytes(m->attn_o[l]);
            if (m->ffn_gate[l] && m->ffn_gate[l]->type != 0) hot_bytes_needed += tensor_f32_cache_bytes(m->ffn_gate[l]);
            if (m->ffn_up[l] && m->ffn_up[l]->type != 0) hot_bytes_needed += tensor_f32_cache_bytes(m->ffn_up[l]);
            if (m->ffn_down[l] && m->ffn_down[l]->type != 0) hot_bytes_needed += tensor_f32_cache_bytes(m->ffn_down[l]);
        }

        if (hot_bytes_needed > cache_budget - cache_used) {
            if (!g_clean_output) {
                printf("Skipping hot-weight cache (%u MB needed, %u MB budget left)\n",
                       (unsigned int)(hot_bytes_needed / (1024U * 1024U)),
                       (unsigned int)((cache_budget - cache_used) / (1024U * 1024U)));
            }
            goto hot_cache_done;
        }

        if (!g_clean_output) {
            printf("Caching hot projection weights in F32 for speed (budget %u MB)...\n",
                   (unsigned int)(cache_budget / (1024U * 1024U)));
        }
        if (force_transpose_square_kv) {
            for (l = 0; l < m->n_layers; l++) {
                int rc;
                if (m->attn_q[l] && m->attn_q[l]->type != 0) {
                    rc = try_cache_tensor_f32_logical_inplace(m->attn_q[l], m->q_dim, m->dim, &cache_used, cache_budget);
                    if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_q[l]->name); } else skipped_count++;
                }
                if (m->attn_k[l] && m->attn_k[l]->type != 0 &&
                    (int)m->attn_k[l]->dims[0] == m->dim &&
                    (int)m->attn_k[l]->dims[1] == m->dim) {
                    rc = try_cache_tensor_f32_force_transpose_inplace(m->attn_k[l], m->kv_dim, m->dim, &cache_used, cache_budget);
                    if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_k[l]->name); } else skipped_count++;
                }
                if (m->attn_v[l] && m->attn_v[l]->type != 0 &&
                    (int)m->attn_v[l]->dims[0] == m->dim &&
                    (int)m->attn_v[l]->dims[1] == m->dim) {
                    rc = try_cache_tensor_f32_force_transpose_inplace(m->attn_v[l], m->kv_dim, m->dim, &cache_used, cache_budget);
                    if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_v[l]->name); } else skipped_count++;
                }
                if (m->attn_o[l] && m->attn_o[l]->type != 0) {
                    rc = try_cache_tensor_f32_logical_inplace(m->attn_o[l], m->dim, m->q_dim, &cache_used, cache_budget);
                    if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_o[l]->name); } else skipped_count++;
                }
            }
        }
        for (l = 0; l < m->n_layers; l++) {
            int rc;
            if (m->attn_q[l] && m->attn_q[l]->type != 0) {
                rc = try_cache_tensor_f32_inplace(m->attn_q[l], &cache_used, cache_budget);
                if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_q[l]->name); } else skipped_count++;
            }
            if (m->attn_k[l] && m->attn_k[l]->type != 0) {
                rc = try_cache_tensor_f32_inplace(m->attn_k[l], &cache_used, cache_budget);
                if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_k[l]->name); } else skipped_count++;
            }
            if (m->attn_v[l] && m->attn_v[l]->type != 0) {
                rc = try_cache_tensor_f32_inplace(m->attn_v[l], &cache_used, cache_budget);
                if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_v[l]->name); } else skipped_count++;
            }
            if (m->attn_o[l] && m->attn_o[l]->type != 0) {
                rc = try_cache_tensor_f32_inplace(m->attn_o[l], &cache_used, cache_budget);
                if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->attn_o[l]->name); } else skipped_count++;
            }
            if (m->ffn_gate[l] && m->ffn_gate[l]->type != 0) {
                rc = try_cache_tensor_f32_inplace(m->ffn_gate[l], &cache_used, cache_budget);
                if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->ffn_gate[l]->name); } else skipped_count++;
            }
            if (m->ffn_up[l] && m->ffn_up[l]->type != 0) {
                rc = try_cache_tensor_f32_inplace(m->ffn_up[l], &cache_used, cache_budget);
                if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->ffn_up[l]->name); } else skipped_count++;
            }
            if (m->ffn_down[l] && m->ffn_down[l]->type != 0) {
                rc = try_cache_tensor_f32_inplace(m->ffn_down[l], &cache_used, cache_budget);
                if (rc > 0) cached_count++; else if (rc < 0) { failed_count++; fprintf(stderr, "Warning: failed to cache %s\n", m->ffn_down[l]->name); } else skipped_count++;
            }
        }
        if (!g_clean_output) {
            printf("Hot weights cache pass complete: cached=%d skipped=%d failed=%d used=%u/%u MB\n",
                   cached_count,
                   skipped_count,
                   failed_count,
                   (unsigned int)(cache_used / (1024U * 1024U)),
                   (unsigned int)(cache_budget / (1024U * 1024U)));
        }
hot_cache_done:
        ;
    }

    if (!legacy_windows &&
        prefer_output_cache &&
        !m->cached_output && m->output && m->output != m->tok_embd &&
        m->output->dims[1] > m->output->dims[0] && m->output->type != 0) {
        size_t cache_bytes = (size_t)m->output->dims[0] * (size_t)m->output->dims[1] * sizeof(float);
        if (cache_bytes <= cache_budget - cache_used) {
            if (!g_clean_output) printf("Caching transposed output head in F32 after hot-weight cache...\n");
            m->cached_output = cache_transposed_tensor_f32(m->output, (int)m->output->dims[0], (int)m->output->dims[1]);
            if (!m->cached_output) {
                fprintf(stderr, "Warning: failed to cache output head (%u MB)\n",
                        (unsigned int)(cache_bytes / (1024U * 1024U)));
            } else {
                cache_used += cache_bytes;
                if (!g_clean_output) printf("Output cache ready.\n");
            }
        } else {
            if (!g_clean_output) {
                printf("Skipping output cache after hot-weight cache (%u MB needed, %u MB budget left)\n",
                       (unsigned int)(cache_bytes / (1024U * 1024U)),
                       (unsigned int)((cache_budget - cache_used) / (1024U * 1024U)));
            }
        }
    }

    /* Diagnostics: verify block sizes and tensor offsets */
    if (!g_clean_output) {
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
        printf("Model: dim=%d layers=%d heads=%d head_dim=%d rope_dim=%d kv_heads=%d hidden=%d vocab=%d seq=%d arch=%s\n",
               m->dim, m->n_layers, m->n_heads, m->head_dim, m->rope_dim, m->n_kv_heads, m->hidden_dim, m->vocab_size, m->max_seq_len,
               hp->architecture);
    }
    if (legacy_windows) {
        printf("Legacy Windows mode: conservative matvec/cache path enabled (cache budget %u MB)\n",
               (unsigned int)(cache_budget / (1024U * 1024U)));
    }
    return 0;
}

static void free_model(Model *m) {
    if (m->cached_output && m->cached_output != m->cached_embd) { free(m->cached_output); m->cached_output = NULL; }
    if (m->cached_embd) { free(m->cached_embd); m->cached_embd = NULL; }
    if (m->cached_pos_embd) { free(m->cached_pos_embd); m->cached_pos_embd = NULL; }
    if (m->rope_cos)   { free(m->rope_cos);   m->rope_cos = NULL; }
    if (m->rope_sin)   { free(m->rope_sin);   m->rope_sin = NULL; }
    if (m->rope_cos_global) { free(m->rope_cos_global); m->rope_cos_global = NULL; }
    if (m->rope_sin_global) { free(m->rope_sin_global); m->rope_sin_global = NULL; }
    free(m->attn_norm); free(m->attn_norm_bias); free(m->attn_q_norm); free(m->attn_k_norm); free(m->attn_q); free(m->attn_q_bias); free(m->attn_k); free(m->attn_v); free(m->attn_o); free(m->attn_o_bias);
    free(m->ffn_norm); free(m->ffn_norm_bias); free(m->post_attn_norm); free(m->post_ffn_norm); free(m->ffn_gate); free(m->ffn_gate_bias); free(m->ffn_up); free(m->ffn_down); free(m->ffn_down_bias);
}

static int alloc_runstate(Model *m, RunState *s, GGUFContext *ctx) {
    int dim = m->dim;
    int hidden_dim = m->hidden_dim;
    int vocab_size = m->vocab_size;
    int n_layers = m->n_layers;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int q_dim = m->q_dim > 0 ? m->q_dim : dim;
    int kv_dim = m->kv_dim > 0 ? m->kv_dim : (n_kv_heads * head_dim);
    int max_seq_len = m->max_seq_len;
    int kv_size = n_layers * n_kv_heads * max_seq_len * head_dim;
    int max_d = dim;
    if (q_dim > max_d) max_d = q_dim;
    if (kv_dim > max_d) max_d = kv_dim;
    if (hidden_dim > max_d) max_d = hidden_dim;
    if (vocab_size > max_d) max_d = vocab_size;

    if (!g_clean_output) {
        printf("Allocating run state...\n");
        printf("  x: %u bytes\n", (unsigned int)(dim * sizeof(float)));
        printf("  q/attn_out: %u bytes each\n", (unsigned int)(q_dim * sizeof(float)));
        printf("  k/v total: %u bytes each\n", (unsigned int)(kv_dim * sizeof(float)));
        printf("  ffn_gate/up/hidden: %u bytes each\n", (unsigned int)(hidden_dim * sizeof(float)));
        printf("  logits: %u bytes\n", (unsigned int)(vocab_size * sizeof(float)));
        printf("  k_cache+v_cache: %u bytes\n", (unsigned int)(kv_size * sizeof(float) * 2));
        printf("  dq_row: %u bytes\n", (unsigned int)(max_d * sizeof(float)));
        printf("  Total state: ~%u KB\n", (unsigned int)((dim + q_dim*2 + kv_dim*2 + hidden_dim*3 + vocab_size + kv_size*2 + max_seq_len + max_d) * sizeof(float) / 1024));
    }

    s->x = (float*)malloc(dim * sizeof(float));
    s->q = (float*)malloc(q_dim * sizeof(float));
    s->k = (float*)malloc(kv_dim * sizeof(float));
    s->v = (float*)malloc(kv_dim * sizeof(float));
    s->attn_out = (float*)malloc(q_dim * sizeof(float));
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

static int forward(Model *m, RunState *s, int token, int pos, float temp, int top_k, float top_p, float min_p, int *recent, int recent_n, int ban_token, int compute_logits) {
    int dim = m->dim;
    int hidden_dim = m->hidden_dim;
    int n_layers = m->n_layers;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int q_dim = m->q_dim > 0 ? m->q_dim : dim;
    int kv_dim = m->kv_dim > 0 ? m->kv_dim : n_kv_heads * head_dim;
    int max_seq_len = m->max_seq_len;
    int vocab_size = m->vocab_size;
    int q_per_kv = n_heads / n_kv_heads;
    int is_gemma3 = (m->arch == ARCH_GEMMA3);
    int is_gemma_family = (m->arch == ARCH_GEMMA || m->arch == ARCH_GEMMA2 || m->arch == ARCH_GEMMA3);
    int use_gemma_softcaps = (m->arch == ARCH_GEMMA2);
    float attn_softcap = m->attn_logit_softcapping;
    float final_softcap = m->final_logit_softcapping;
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

    if (use_gemma_softcaps && attn_softcap <= 0.0f) attn_softcap = 50.0f;

    if (m->arch == ARCH_GPT2) {
        int qkv_dim = q_dim;
        int attn_dim = dim;
        int mlp_hidden = hidden_dim;
        int h2, t2, i2;
        float *qkv = q;
        float *k0 = qkv + attn_dim;
        float *v0 = qkv + attn_dim * 2;

        if (m->cached_embd) {
            memcpy(x, m->cached_embd + (size_t)token * (size_t)dim, (size_t)dim * sizeof(float));
        } else if (m->tok_embd && (int)m->tok_embd->dims[0] == dim && (int)m->tok_embd->dims[1] == vocab_size) {
            dequantize_row(m->tok_embd, x, token, dim);
        } else if (m->tok_embd_transposed) {
            load_transposed_embedding_fast(m->tok_embd, x, token, dim, vocab_size);
        } else {
            dequantize_row(m->tok_embd, x, token, dim);
        }
        if (m->cached_pos_embd) {
            for (i2 = 0; i2 < dim; i2++) x[i2] += m->cached_pos_embd[(size_t)pos * (size_t)dim + (size_t)i2];
        } else if (m->pos_embd && (int)m->pos_embd->dims[0] == dim) {
            dequantize_row(m->pos_embd, s->dq_row, pos, dim);
            for (i2 = 0; i2 < dim; i2++) x[i2] += s->dq_row[i2];
        } else if (m->pos_embd && m->pos_embd->type == 0 && m->pos_embd->dims[1] == (u64)dim && m->pos_embd->dims[0] > (u64)pos) {
            int seq = (int)m->pos_embd->dims[0];
            for (i2 = 0; i2 < dim; i2++) {
                dequantize_row(m->pos_embd, s->dq_row, i2, seq);
                x[i2] += s->dq_row[pos];
            }
        } else if (m->pos_embd) {
            if ((int)m->pos_embd->dims[1] == dim) {
                int seq = (int)m->pos_embd->dims[0];
                for (i2 = 0; i2 < dim; i2++) {
                    dequantize_row(m->pos_embd, s->dq_row, i2, seq);
                    x[i2] += s->dq_row[pos];
                }
            } else {
                dequantize_row(m->pos_embd, s->dq_row, pos, dim);
                for (i2 = 0; i2 < dim; i2++) x[i2] += s->dq_row[i2];
            }
        }

        for (l = 0; l < n_layers; l++) {
            const float *ln_w = m->attn_norm[l] ? (const float*)m->attn_norm[l]->data : NULL;
            const float *ln_b = m->attn_norm_bias[l] ? (const float*)m->attn_norm_bias[l]->data : NULL;
            const float *ln2_w = m->ffn_norm[l] ? (const float*)m->ffn_norm[l]->data : NULL;
            const float *ln2_b = m->ffn_norm_bias[l] ? (const float*)m->ffn_norm_bias[l]->data : NULL;
            int head_dim = m->head_dim;
            int h;
            int attn_start = 0;
            int q_per_kv = 1;
            (void)q_per_kv;
            for (i2 = 0; i2 < dim; i2++) tmp[i2] = x[i2];
            layernorm(x, ln_w, ln_b, dim, m->norm_eps);

            matvec(m->attn_q[l], x, qkv, qkv_dim, dim, s->dq_row);
            add_bias(qkv, m->attn_q_bias[l], qkv_dim);

            memset(attn_out, 0, (size_t)dim * sizeof(float));
            for (h = 0; h < n_heads; h++) {
                float *q_head = qkv + h * head_dim;
                float *k_head = k0 + h * head_dim;
                float *v_head = v0 + h * head_dim;
                float *kc = k_cache + ((l * n_heads + h) * max_seq_len + pos) * head_dim;
                float *vc = v_cache + ((l * n_heads + h) * max_seq_len + pos) * head_dim;
                float *out_head = attn_out + h * head_dim;
                float max_score = -1e30f;
                float sum_score = 0.0f;
                for (t2 = 0; t2 <= pos; t2++) {
                    float *kt = k_cache + ((l * n_heads + h) * max_seq_len + t2) * head_dim;
                    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, score;
                    for (i2 = 0; i2 + 3 < head_dim; i2 += 4) {
                        s0 += q_head[i2]   * kt[i2];
                        s1 += q_head[i2+1] * kt[i2+1];
                        s2 += q_head[i2+2] * kt[i2+2];
                        s3 += q_head[i2+3] * kt[i2+3];
                    }
                    score = s0 + s1 + s2 + s3;
                    for (; i2 < head_dim; i2++) score += q_head[i2] * kt[i2];
                    score *= m->rsqrt_head_dim;
                    attn_scores[t2] = score;
                    if (score > max_score) max_score = score;
            }
            for (t2 = 0; t2 <= pos; t2++) {
                attn_scores[t2] = (float)exp((double)(attn_scores[t2] - max_score));
                sum_score += attn_scores[t2];
            }
                for (t2 = 0; t2 <= pos; t2++) attn_scores[t2] /= sum_score;
                for (i2 = 0; i2 < head_dim; i2++) out_head[i2] = 0.0f;
                for (t2 = 0; t2 <= pos; t2++) {
                    float score = attn_scores[t2];
                    float *vt = v_cache + ((l * n_heads + h) * max_seq_len + t2) * head_dim;
                    for (i2 = 0; i2 + 3 < head_dim; i2 += 4) {
                        out_head[i2]   += score * vt[i2];
                        out_head[i2+1] += score * vt[i2+1];
                        out_head[i2+2] += score * vt[i2+2];
                        out_head[i2+3] += score * vt[i2+3];
                    }
                    for (; i2 < head_dim; i2++) out_head[i2] += score * vt[i2];
                }
                for (i2 = 0; i2 < head_dim; i2++) {
                    kc[i2] = k_head[i2];
                    vc[i2] = v_head[i2];
                }
            }

            matvec(m->attn_o[l], attn_out, x, dim, dim, s->dq_row);
            add_bias(x, m->attn_o_bias[l], dim);
            for (i2 = 0; i2 < dim; i2++) x[i2] += tmp[i2];

            for (i2 = 0; i2 < dim; i2++) tmp[i2] = x[i2];
            layernorm(x, ln2_w, ln2_b, dim, m->norm_eps);

            matvec(m->ffn_gate[l], x, s->ffn_gate, mlp_hidden, dim, s->dq_row);
            add_bias(s->ffn_gate, m->ffn_gate_bias[l], mlp_hidden);
            for (i2 = 0; i2 < mlp_hidden; i2++) s->ffn_hidden[i2] = gelu(s->ffn_gate[i2]);
            matvec(m->ffn_down[l], s->ffn_hidden, x, dim, mlp_hidden, s->dq_row);
            add_bias(x, m->ffn_down_bias[l], dim);
            for (i2 = 0; i2 < dim; i2++) x[i2] += tmp[i2];
        }

        if (!compute_logits) return 0;
        layernorm(x, m->output_norm ? (const float*)m->output_norm->data : NULL,
                  m->output_norm_bias ? (const float*)m->output_norm_bias->data : NULL,
                  dim, m->norm_eps);
        if (m->cached_output) {
            Tensor fake;
            fake = *m->output;
            fake.type = 0;
            fake.dims[0] = (u64)dim;
            fake.dims[1] = (u64)vocab_size;
            fake.data = m->cached_output;
            matvec(&fake, x, s->logits, vocab_size, dim, s->dq_row);
        } else if (m->cached_embd && m->output == m->tok_embd) {
            Tensor fake;
            fake = *m->output;
            fake.type = 0;
            fake.dims[0] = (u64)dim;
            fake.dims[1] = (u64)vocab_size;
            fake.data = m->cached_embd;
            matvec(&fake, x, s->logits, vocab_size, dim, s->dq_row);
        } else {
            matvec(m->output, x, s->logits, vocab_size, dim, s->dq_row);
        }
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

    /* Token embedding row lookup */
    if (m->cached_embd) {
        memcpy(x, m->cached_embd + (size_t)token * (size_t)dim, (size_t)dim * sizeof(float));
    } else if (!m->tok_embd_transposed) {
        dequantize_row(m->tok_embd, x, token, dim);
    } else {
        load_transposed_embedding_fast(m->tok_embd, x, token, dim, vocab_size);
    }
    if (is_gemma_family) {
        float emb_scale = (float)sqrt((double)dim);
        for (i = 0; i < dim; i++) x[i] *= emb_scale;
    }

    for (l = 0; l < n_layers; l++) {
        int is_global_layer = is_gemma3 && ((l % 6) == 0);
        int attn_start = 0;
        const float *rope_cos = m->rope_cos;
        const float *rope_sin = m->rope_sin;
        /* Attention */
        for (i = 0; i < dim; i++) tmp[i] = x[i];
        rmsnorm(x, (float*)m->attn_norm[l]->data, dim, m->norm_eps);

        matvec(m->attn_q[l], x, q, q_dim, dim, s->dq_row);
        matvec(m->attn_k[l], x, k, kv_dim, dim, s->dq_row);
        matvec(m->attn_v[l], x, v, kv_dim, dim, s->dq_row);

        if (m->attn_q_norm[l] && m->attn_q_norm[l]->data) {
            for (h = 0; h < n_heads; h++) rmsnorm(q + h * head_dim, (float*)m->attn_q_norm[l]->data, head_dim, m->norm_eps);
        }
        if (m->attn_k_norm[l] && m->attn_k_norm[l]->data) {
            for (h_kv = 0; h_kv < n_kv_heads; h_kv++) rmsnorm(k + h_kv * head_dim, (float*)m->attn_k_norm[l]->data, head_dim, m->norm_eps);
        }

        if (is_gemma3) {
            if (is_global_layer && m->rope_cos_global && m->rope_sin_global) {
                rope_cos = m->rope_cos_global;
                rope_sin = m->rope_sin_global;
            } else if (!is_global_layer) {
                int window = m->attention_sliding_window > 0 ? m->attention_sliding_window : 1024;
                attn_start = pos > window - 1 ? pos - (window - 1) : 0;
            }
        }

        if (is_gemma_family) {
            for (h = 0; h < n_heads; h++) rope_1d_half(q + h * head_dim, m->rope_dim, pos, rope_cos, rope_sin);
            for (h_kv = 0; h_kv < n_kv_heads; h_kv++) rope_1d_half(k + h_kv * head_dim, m->rope_dim, pos, rope_cos, rope_sin);
        } else {
            for (h = 0; h < n_heads; h++) rope_1d(q + h * head_dim, m->rope_dim, pos, rope_cos, rope_sin);
            for (h_kv = 0; h_kv < n_kv_heads; h_kv++) rope_1d(k + h_kv * head_dim, m->rope_dim, pos, rope_cos, rope_sin);
        }

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
            for (t = attn_start; t <= pos; t++) {
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
                if (attn_softcap > 0.0f) score = softcap(score, attn_softcap);
                attn_scores[t] = score;
                if (score > max_score) max_score = score;
            }
            for (t = attn_start; t <= pos; t++) {
                attn_scores[t] = (float)exp((double)(attn_scores[t] - max_score));
                sum_score += attn_scores[t];
            }
            for (t = attn_start; t <= pos; t++) attn_scores[t] /= sum_score;
            for (i = 0; i < head_dim; i++) out_head[i] = 0.0f;
            for (t = attn_start; t <= pos; t++) {
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
        matvec(m->attn_o[l], attn_out, x, dim, q_dim, s->dq_row);
        if (is_gemma_family && m->post_attn_norm[l] && m->post_attn_norm[l]->data) {
            rmsnorm(x, (float*)m->post_attn_norm[l]->data, dim, m->norm_eps);
        }
        for (i = 0; i < dim; i++) x[i] += tmp[i];

        /* FFN */
        for (i = 0; i < dim; i++) tmp[i] = x[i];
        rmsnorm(x, (float*)m->ffn_norm[l]->data, dim, m->norm_eps);

        matvec(m->ffn_gate[l], x, s->ffn_gate, hidden_dim, dim, s->dq_row);
        matvec(m->ffn_up[l], x, s->ffn_up, hidden_dim, dim, s->dq_row);
        if (is_gemma_family) {
            for (i = 0; i < hidden_dim; i++) s->ffn_hidden[i] = gelu(s->ffn_gate[i]) * s->ffn_up[i];
        } else {
            for (i = 0; i < hidden_dim; i++) s->ffn_hidden[i] = silu(s->ffn_gate[i]) * s->ffn_up[i];
        }

        matvec(m->ffn_down[l], s->ffn_hidden, x, dim, hidden_dim, s->dq_row);
        if (is_gemma_family && m->post_ffn_norm[l] && m->post_ffn_norm[l]->data) {
            rmsnorm(x, (float*)m->post_ffn_norm[l]->data, dim, m->norm_eps);
        }
        for (i = 0; i < dim; i++) x[i] += tmp[i];
    }

    if (!compute_logits) return 0;

    rmsnorm(x, (float*)m->output_norm->data, dim, m->norm_eps);
    if (m->cached_output) {
        Tensor fake;
        fake = *m->output;
        fake.type = 0;
        fake.dims[0] = (u64)dim;
        fake.dims[1] = (u64)vocab_size;
        fake.data = m->cached_output;
        matvec(&fake, x, s->logits, vocab_size, dim, s->dq_row);
    } else if (m->cached_embd && m->output == m->tok_embd) {
        Tensor fake;
        fake = *m->output;
        fake.type = 0;
        fake.dims[0] = (u64)dim;
        fake.dims[1] = (u64)vocab_size;
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
    if (final_softcap > 0.0f) {
        for (i = 0; i < vocab_size; i++) s->logits[i] = softcap(s->logits[i], final_softcap);
    } else if (use_gemma_softcaps) {
        for (i = 0; i < vocab_size; i++) s->logits[i] = softcap(s->logits[i], 30.0f);
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

static void reset_byte_token_map(void) {
    int i;
    for (i = 0; i < 256; i++) g_byte_token[i] = -1;
}

static int load_tokenizer(const char *path) {
    FILE *f;
    size_t sz;
    u8 *buf;
    u32 count;
    int i;
    size_t pos;
    int parsed;
    if (g_tok_buf) { free(g_tok_buf); g_tok_buf = NULL; free(g_vocab); g_vocab = NULL; }
    if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; }
    g_sorted_init = 0;
    free_gpt2_tokenizer_tables();
    reset_byte_token_map();
    g_gpt2_add_space_prefix = 0;
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
    rebuild_byte_token_map();
    if (!g_clean_output) printf("Loaded tokenizer: %d tokens\n", g_vocab_n);
    return 1;
}

static int probe_tensor_section(const u8 *base, size_t size, u64 offset, u64 tensor_count) {
    const u8 *p = base + offset;
    const u8 *end = base + size;
    u64 check = tensor_count < 16 ? tensor_count : 16;
    u64 i;
    for (i = 0; i < check; i++) {
        u64 name_len;
        u32 n_dims;
        if (p + 8 > end) return 0;
        memcpy(&name_len, p, 8); p += 8;
        if (name_len == 0 || name_len > 4096) return 0;
        if (p + name_len > end) return 0;
        p += name_len;
        if (p + 4 > end) return 0;
        memcpy(&n_dims, p, 4); p += 4;
        if (n_dims > 4) return 0;
        if (p + (size_t)n_dims * 8 + 12 > end) return 0;
        p += (size_t)n_dims * 8;
        p += 12;
    }
    return 1;
}

static int load_tokenizer_auto(const char *model_path, const char *tok_path, GGUFContext *ctx) {
    char found[768];

    if (tok_path && tok_path[0] && strcmp(tok_path, "auto") != 0) {
        if (load_tokenizer(tok_path)) {
            if (!g_clean_output) printf("Tokenizer source: %s\n", tok_path);
            return 1;
        }
        if (!g_clean_output) printf("Warning: failed to load tokenizer file %s\n", tok_path);
    }

    if (load_tokenizer_from_gguf(ctx)) {
        if (!g_clean_output) printf("Tokenizer source: embedded GGUF metadata\n");
        return 1;
    }

    if (discover_tokenizer_path(model_path, found, sizeof(found))) {
        if (load_tokenizer(found)) {
            if (!g_clean_output) printf("Tokenizer source: %s\n", found);
            return 1;
        }
        if (!g_clean_output) printf("Warning: failed to load discovered tokenizer file %s\n", found);
    }

    if (!g_clean_output) printf("Warning: no tokenizer file found; falling back to raw byte tokens\n");
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
        if (!g_clean_output) printf("Tokenizer sorted: %d tokens by length\n", g_vocab_n);
}

static int tokenize(const char *text, int *tokens, int max_tokens, int vocab_size) {
    if (g_tokenizer_is_gpt2) {
        return tokenize_gpt2(text, tokens, max_tokens);
    }
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
            unsigned char c = (unsigned char)text[pos];
            if (g_byte_token[c] >= 0) {
                tokens[n++] = g_byte_token[c];
            } else {
                tokens[n++] = (int)c;
            }
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

static int append_sentencepiece_text_tokens(int *dst, int cap, int n, const char *text, int vocab_size) {
    size_t len;
    size_t out_len = 0;
    char *buf;
    size_t i;

    if (!dst || !text || cap <= 0 || n >= cap) return n;
    len = strlen(text);
    if (len == 0) return n;

    /* Gemma-style tokenizers often want whitespace to behave like SentencePiece.
       We only normalize the user prompt, not the wrapper tokens. */
    buf = (char*)malloc(len * 4 + 4);
    if (!buf) return append_text_tokens(dst, cap, n, text, vocab_size);

    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == ' ') {
            buf[out_len++] = (char)0xE2;
            buf[out_len++] = (char)0x96;
            buf[out_len++] = (char)0x81;
        } else {
            buf[out_len++] = (char)c;
        }
    }
    buf[out_len] = '\0';
    n = append_text_tokens(dst, cap, n, buf, vocab_size);
    free(buf);
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
    if (g_tokenizer_is_gpt2) {
        detokenize_gpt2(token);
        return;
    }
    if (g_vocab && token >= 0 && token < g_vocab_n && g_vocab[token].len > 0) {
        int byte = parse_byte_fallback_token(g_vocab[token].text, g_vocab[token].len);
        if (byte >= 0 && byte < 256) {
            putchar((char)byte);
        } else {
            print_token_text_clean(g_vocab[token].text, g_vocab[token].len);
        }
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
    if (g_clean_output) return;
    printf("Special tokens: ");
    for (i = 0; i < 9; i++) {
        ids[i] = find_special_token(names[i]);
        if (ids[i] >= 0) printf("%s=%d ", names[i], ids[i]);
    }
    if (find_special_token("<end_of_turn>") >= 0) printf("<end_of_turn>=%d ", find_special_token("<end_of_turn>"));
    if (find_special_token("\n") >= 0) printf("\\n=%d ", find_special_token("\n"));
    printf("\n");
}

static void init_eog_tokens(void) {
    const char *names[6] = { "<|endoftext|>", "<|im_end|>", "<reponame>", "</s>", "<eos>", "<end_of_turn>" };
    int i;
    g_eog_token_count = 0;
    for (i = 0; i < 6; i++) {
        int tok = find_special_token(names[i]);
        int j;
        if (tok < 0) continue;
        for (j = 0; j < g_eog_token_count; j++) {
            if (g_eog_tokens[j] == tok) break;
        }
        if (j == g_eog_token_count && g_eog_token_count < 8) {
            g_eog_tokens[g_eog_token_count++] = tok;
        }
    }
}

#define TOKEN_TYPE_NORMAL 1
#define TOKEN_TYPE_UNKNOWN 2
#define TOKEN_TYPE_CONTROL 3
#define TOKEN_TYPE_USER_DEFINED 4
#define TOKEN_TYPE_UNUSED 5

static int is_eog_token_id(int tok) {
    int i;
    for (i = 0; i < g_eog_token_count; i++) {
        if (g_eog_tokens[i] == tok) return 1;
    }
    return 0;
}

static int token_is_blocked_for_generation(int tok) {
    int type;
    if (!g_vocab || tok < 0 || tok >= g_vocab_n) return 0;
    if (g_eog_token_count > 0 && is_eog_token_id(tok)) return 0;
    if (g_vocab[tok].len <= 0) return 1;
    if (!g_token_types || tok >= g_vocab_n) return 0;
    type = g_token_types[tok];
    if (type == TOKEN_TYPE_UNKNOWN || type == TOKEN_TYPE_CONTROL) {
        return 1;
    }
    return 0;
}

static int is_eog_token(int tok, int eos_token) {
    int i;
    if (tok < 0) return 0;
    if (eos_token >= 0 && tok == eos_token) return 1;
    for (i = 0; i < g_eog_token_count; i++) {
        if (g_eog_tokens[i] == tok) return 1;
    }
    return 0;
}

static void print_banner(void) {
    printf("                                _____\n");
    printf(" ___ _ __   ___ _ __ _ __ ___  / ____|\n");
    printf("/ __| '_ \\ / _ \\ '__| '_ ` _ \\| |\n");
    printf("\\__ \\ |_) |  __/ |  | | | | | | |____\n");
    printf("|___/ .__/ \\___|_|  |_| |_| |_|\\_____|\n");
    printf("    |_|\n");
    printf("\n");
}

static void sanitize_repl_input(char *s) {
    unsigned char *p = (unsigned char*)s;
    size_t i, j, len;
    if (!s || !s[0]) return;
    if (p[0] == 0xEF && p[1] == 0xBB && p[2] == 0xBF) {
        memmove(s, s + 3, strlen(s + 3) + 1);
        p = (unsigned char*)s;
    }
    len = strlen(s);
    i = 0;
    while (i < len) {
        unsigned char c = p[i];
        if (c >= 32 || c == '\t') break;
        i++;
    }
    if (i > 0) {
        memmove(s, s + i, len - i + 1);
        len -= i;
    }
    for (i = 0, j = 0; i < len; i++) {
        unsigned char c = p[i];
        if (c == 0) break;
        if (c < 32 && c != '\t') continue;
        p[j++] = c;
    }
    p[j] = '\0';
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
    hSession = pOpen(L"spermC/2.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
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

static int run_local_request(Model *m, GGUFContext *ctx, HParams *hp,
                             const char *model_path, const char *prompt,
                             int n_generate, float temp, int top_k, float top_p, float min_p,
                             int eos_token, int no_eos_stop, int do_chat, int do_raw_tokens,
                             int gpt2_instruct_mode, int smollm2_mode, int qwen_chatml_mode,
                             int tinyllama_mode, int gemma3_mode, int slopllm_mode,
                             int interactive_mode, int n_generate_user) {
    RunState s;
    int prompt_len = 0;
    int *prompt_tokens;
    int next_token = 0;
    int stop_on_eos = 0;
    int max_generate;
    int pos;
    int i;
    int tokens_printed = 0;
    double t_request_start;
    double t_first_token = 0.0;
    double t_end;

    if (!prompt) prompt = "";
    max_generate = n_generate;
    if (interactive_mode && !n_generate_user) max_generate = 256;
    if (max_generate < 1) max_generate = 1;
    stop_on_eos = interactive_mode && !no_eos_stop;
    g_allow_eog_sampling = stop_on_eos ? 1 : 0;
    memset(&s, 0, sizeof(s));
    if (alloc_runstate(m, &s, ctx) != 0) {
        g_allow_eog_sampling = 0;
        return 1;
    }

    prompt_tokens = (int*)malloc(m->max_seq_len * sizeof(int));
    if (!prompt_tokens) {
        fprintf(stderr, "Failed to allocate prompt buffer\n");
        free_runstate(&s);
        g_allow_eog_sampling = 0;
        return 1;
    }

    /* Build prompt tokens */
    {
        int tok_im_start = find_special_token("<|im_start|>");
        int tok_im_end = find_special_token("<|im_end|>");
        int tok_bos = find_special_token("<bos>");
        int tok_endoftext = find_special_token("<|endoftext|>");
        int tok_user = find_special_token("<|user|>");
        int tok_assistant = find_special_token("<|assistant|>");
        int auto_bos_id = hp->tokenizer_bos_token_id > 0 ? (int)hp->tokenizer_bos_token_id : -1;
        if (tok_bos < 0) tok_bos = find_special_token("<s>");
        if (auto_bos_id < 0) auto_bos_id = tok_bos;
        if (auto_bos_id < 0) auto_bos_id = tok_endoftext;
        int prompt_has_template =
            contains_nocase(prompt, "<start_of_turn>") ||
            contains_nocase(prompt, "<|im_start|>") ||
            contains_nocase(prompt, "<|user|>") ||
            contains_nocase(prompt, "<|assistant|>") ||
            contains_nocase(prompt, "[INST]") ||
            contains_nocase(prompt, "user:");
        int n = 0;
        int wrapped_prompt = 0;

        if (gpt2_instruct_mode && !do_chat && !prompt_has_template) {
            wrapped_prompt = 1;
            if (!g_clean_output) printf("Chat tokens: gpt2-instruct user/assistant style\n");
            if (tok_user >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_user;
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, prompt, m->vocab_size);
            if (tok_assistant >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_assistant;
        } else if (!do_chat && prompt_has_template) {
            if (!g_clean_output) printf("Prompt already looks templated; using raw prompt text\n");
            prompt_len = tokenize(prompt, prompt_tokens, m->max_seq_len, m->vocab_size);
        } else if (qwen_chatml_mode) {
            wrapped_prompt = 1;
            if (!g_clean_output) printf("Chat tokens: qwen ChatML special-token style\n");
            if (tok_im_start >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_im_start;
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "user\n", m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, prompt, m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "\n", m->vocab_size);
            if (tok_im_end >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_im_end;
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "\n", m->vocab_size);
            if (tok_im_start >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_im_start;
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "assistant\n", m->vocab_size);
        } else if (m->arch == ARCH_GPT2 || (!do_chat && !smollm2_mode && !qwen_chatml_mode && !tinyllama_mode && !gemma3_mode && !slopllm_mode)) {
            prompt_len = tokenize(prompt, prompt_tokens, m->max_seq_len, m->vocab_size);
        } else if (smollm2_mode) {
            char *chat_buf;
            size_t chat_cap;
            wrapped_prompt = 1;
            if (!g_clean_output) printf("Chat tokens: smollm2 ChatML style\n");
            chat_cap = strlen(prompt) + 256;
            chat_buf = (char*)malloc(chat_cap);
            if (!chat_buf) {
                fprintf(stderr, "Failed to allocate SmolLM2 prompt buffer\n");
                free(prompt_tokens);
                free_runstate(&s);
                g_allow_eog_sampling = 0;
                return 1;
            }
            sprintf(chat_buf,
                    "<|im_start|>system\n"
                    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    "%s"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n",
                    prompt);
            prompt_len = tokenize(chat_buf, prompt_tokens, m->max_seq_len, m->vocab_size);
            free(chat_buf);
            n = prompt_len;
        } else if (gemma3_mode) {
            wrapped_prompt = 1;
            if (!g_clean_output) printf("Chat tokens: gemma3 start_of_turn style (no system, with BOS)\n");
            if (tok_bos >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_bos;
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "<start_of_turn>user\n", m->vocab_size);
            n = append_sentencepiece_text_tokens(prompt_tokens, m->max_seq_len, n, prompt, m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "<end_of_turn>\n", m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "<start_of_turn>model\n", m->vocab_size);
        } else if (tinyllama_mode) {
            wrapped_prompt = 1;
            if (!g_clean_output) printf("Chat tokens: tinyllama Llama-2 style\n");
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "<s>[INST] ", m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, prompt, m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, " [/INST]", m->vocab_size);
        } else if (slopllm_mode) {
            wrapped_prompt = 1;
            if (!g_clean_output) printf("Chat tokens: slopllm user/assistant style\n");
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, "user: ", m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, prompt, m->vocab_size);
            n = append_text_tokens(prompt_tokens, m->max_seq_len, n, " assistant:", m->vocab_size);
        } else {
            wrapped_prompt = 1;
            {
                int tmp_tok[256];
                int tmp_n;
                if (tok_im_start < 0) tok_im_start = find_special_token("<s>");
                if (tok_im_end < 0) tok_im_end = find_special_token("</s>");
                if (!g_clean_output) printf("Chat tokens: im_start=%d im_end=%d\n", tok_im_start, tok_im_end);
                if (tok_im_start >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_im_start;
                tmp_n = tokenize("user\n", tmp_tok, 256, m->vocab_size);
                for (i = 0; i < tmp_n && n < m->max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];
                tmp_n = tokenize(prompt, tmp_tok, 256, m->vocab_size);
                for (i = 0; i < tmp_n && n < m->max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];
                tmp_n = tokenize("\n", tmp_tok, 256, m->vocab_size);
                for (i = 0; i < tmp_n && n < m->max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];
                if (tok_im_end >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_im_end;
                tmp_n = tokenize("\n", tmp_tok, 256, m->vocab_size);
                for (i = 0; i < tmp_n && n < m->max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];
                if (tok_im_start >= 0 && n < m->max_seq_len) prompt_tokens[n++] = tok_im_start;
                tmp_n = tokenize("assistant\n", tmp_tok, 256, m->vocab_size);
                for (i = 0; i < tmp_n && n < m->max_seq_len; i++) prompt_tokens[n++] = tmp_tok[i];
            }
        }

        if (wrapped_prompt) prompt_len = n;
        if (hp->tokenizer_add_bos_token && auto_bos_id >= 0 && prompt_len > 0 && prompt_len < m->max_seq_len) {
            if (prompt_tokens[0] != auto_bos_id) {
                memmove(prompt_tokens + 1, prompt_tokens, (size_t)prompt_len * sizeof(int));
                prompt_tokens[0] = auto_bos_id;
                prompt_len++;
            }
        }
    }

    if (prompt_len < 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        free(prompt_tokens);
        free_runstate(&s);
        g_allow_eog_sampling = 0;
        return 1;
    }

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

    t_request_start = now_seconds();
    if (prompt_len > 1 && !g_clean_output) {
        printf("Prefill %d tokens...", prompt_len);
        fflush(stdout);
    }
    for (pos = 0; pos < prompt_len - 1 && prompt_len > 0; pos++) {
        forward(m, &s, prompt_tokens[pos], pos, 0.0f, 0, 0.0f, 0.0f, NULL, 0, -1, 0);
        if (prompt_len > 1 && !g_clean_output && (pos % 2 == 1 || pos == prompt_len - 2)) {
            printf(".");
            fflush(stdout);
        }
    }

    if (prompt_len > 1 && !g_clean_output) printf("\n");
    if (prompt_len > 0) {
        int ban_token = (!interactive_mode && !no_eos_stop && max_generate > 1) ? eos_token : -1;
        next_token = forward(m, &s, prompt_tokens[prompt_len - 1], pos, temp, top_k, top_p, min_p, NULL, 0, ban_token, 1);
        pos = prompt_len;
    } else {
        int ban_token = (!interactive_mode && !no_eos_stop && max_generate > 1) ? eos_token : -1;
        next_token = forward(m, &s, 1, 0, temp, top_k, top_p, min_p, NULL, 0, ban_token, 1);
        pos = 1;
    }

    {
        int *last_tokens = (int*)malloc(m->max_seq_len * sizeof(int));
        int n_last = 0;
        if (!last_tokens) {
            fprintf(stderr, "Failed to allocate repetition buffer\n");
            free(prompt_tokens);
            free_runstate(&s);
            g_allow_eog_sampling = 0;
            return 1;
        }
        if (!(stop_on_eos && is_eog_token(next_token, eos_token))) {
            if (tokens_printed == 0) t_first_token = now_seconds();
            if (do_raw_tokens) {
                printf("[%d]", next_token);
            } else if (next_token != 0) {
                detokenize(next_token, m->vocab_size);
            } else {
                printf("<|endoftext|>");
            }
            fflush(stdout);
            if (n_last < m->max_seq_len) last_tokens[n_last++] = next_token;
            tokens_printed++;
        }
        for (i = 1; i < max_generate; i++) {
            int ban_token = (!interactive_mode && !no_eos_stop && i + 1 < max_generate) ? eos_token : -1;
            if (pos >= m->max_seq_len) break;
            next_token = forward(m, &s, next_token, pos, temp, top_k, top_p, min_p, last_tokens, n_last, ban_token, 1);
            pos++;
            if (stop_on_eos && is_eog_token(next_token, eos_token)) break;
            if (do_raw_tokens) {
                printf("[%d]", next_token);
            } else if (next_token != 0) {
                detokenize(next_token, m->vocab_size);
            } else {
                printf("<|endoftext|>");
            }
            if (tokens_printed == 0) t_first_token = now_seconds();
            if (n_last < m->max_seq_len) last_tokens[n_last++] = next_token;
            tokens_printed++;
            fflush(stdout);
        }
        free(last_tokens);
    }
    printf("\n");
    t_end = now_seconds();
    if (!g_clean_output) {
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
    }

    free(prompt_tokens);
    free_runstate(&s);
    g_allow_eog_sampling = 0;
    return 0;
}

/* --- Main --- */

int main(int argc, char **argv) {
    GGUFContext *ctx;
    Model m;
    HParams hp;
    Arch parsed_arch;
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
    float min_p = 0.05f;
    int top_k = 40;
    int eos_token = 2;
    int no_eos_stop = 0;
    int do_chat = 0;
    int use_api = 0;
    int do_info = 0;
    int do_list = 0;
    int do_raw_tokens = 0;
    int do_repl = 0;
    int do_interactive = 0;
    int tinyllama_mode = 0;
    int gemma3_mode = 0;
    int gpt2_instruct_mode = 0;
    int smollm2_mode = 0;
    int qwen_chatml_mode = 0;
    int slopllm_mode = 0;
    int seq_user = 0;
    int n_heads_user = 0;
    int n_kv_user = 0;
    int hidden_user = 0;
    int temp_user = 0;
    int top_k_user = 0;
    int top_p_user = 0;
    int min_p_user = 0;
    int n_generate_user = 0;
    int eos_user = 0;
    unsigned int seed = 0;
    int arg;
    if (argc < 2) {
        printf("Usage: spermC.exe <model.gguf> [options]\n");
        printf("Options:\n");
        printf("  --info          Dump model info and exit\n");
        printf("  --list          List all tensors and exit\n");
        printf("  --clean         Suppress non-essential output\n");
        printf("  --no-eos-stop   Do not stop on EOS token\n");
        printf("  --raw-tokens    Print token IDs instead of text\n");
        printf("  --chat          Wrap prompt in chat template\n");
        printf("  --repl          Keep model loaded and read prompts from stdin\n");
        printf("  --keep-loaded   Alias for --repl\n");
        printf("  --interactive   Llama.cpp-style interactive chat; stops on EOS\n");
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
        printf("  --min-p <p>     Min-p sampling (default 0.05)\n");
        printf("  --eos <id>      EOS token ID (default 2)\n");
        printf("  --repeat-penalty <p>  Repetition penalty (default 1.0)\n");
        printf("  --threads <n|auto>   Parallel threads for large matmul (WinNT only, default auto)\n");
        return 1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    memset(&m, 0, sizeof(Model));
    m.repeat_penalty = 1.0f;
    reset_byte_token_map();

    model_path = argv[1];
    tinyllama_mode = 0;
    for (arg = 2; arg < argc; arg++) {
        if (strcmp(argv[arg], "--info") == 0) { do_info = 1; }
        else if (strcmp(argv[arg], "--clean") == 0) { g_clean_output = 1; }
        else if (strcmp(argv[arg], "--no-eos-stop") == 0) { no_eos_stop = 1; }
        else if (strcmp(argv[arg], "--raw-tokens") == 0) { do_raw_tokens = 1; }
        else if (strcmp(argv[arg], "--chat") == 0) { do_chat = 1; }
        else if (strcmp(argv[arg], "--repl") == 0 || strcmp(argv[arg], "--keep-loaded") == 0) { do_repl = 1; }
        else if (strcmp(argv[arg], "--interactive") == 0) { do_repl = 1; do_interactive = 1; }
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
        else if (strcmp(argv[arg], "-n") == 0 && arg + 1 < argc) { n_generate = atoi(argv[arg+1]); n_generate_user = 1; arg++; }
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
        else if (strcmp(argv[arg], "--eos") == 0 && arg + 1 < argc) { eos_token = atoi(argv[arg+1]); eos_user = 1; arg++; }
        else if (strcmp(argv[arg], "--repeat-penalty") == 0 && arg + 1 < argc) { m.repeat_penalty = (float)atof(argv[arg+1]); arg++; }
    }

    if (do_interactive) {
        g_clean_output = 1;
        g_interactive_ui = 1;
    }

    if (g_n_threads <= 0) g_n_threads = default_thread_count();
    if (g_n_threads < 1) g_n_threads = 1;

    if (seed == 0) seed = (unsigned int)time(NULL);
    srand(seed);

    if (!g_clean_output) {
        printf("spermC v2 (multi-arch + sorted tokenizer)\n");
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
        printf("  Sliding window: %u\n", (unsigned int)ctx->hp.attention_sliding_window);
        printf("  Rope dimension count: %u\n", (unsigned int)ctx->hp.rope_dimension_count);
        printf("  Rope freq base: %.0f\n", (double)ctx->hp.rope_freq_base);
        printf("  Attn logit softcap: %.6g\n", (double)ctx->hp.attn_logit_softcapping);
        printf("  Final logit softcap: %.6g\n", (double)ctx->hp.final_logit_softcapping);
        if (ctx->hp.chat_template[0]) {
            printf("  Chat template: %s\n", ctx->hp.chat_template);
        }
        gguf_free(ctx);
        return 0;
    }

    hp = ctx->hp;
    parsed_arch = detect_architecture_name(hp.architecture);
    gemma3_mode = (parsed_arch == ARCH_GEMMA3) ||
                  template_mentions(&hp, "start_of_turn");
    {
        int gemma3_270m_mode = (parsed_arch == ARCH_GEMMA3) &&
                               hp.embedding_length == 640 &&
                               hp.block_count == 18 &&
                               hp.attention_head_count == 4;
        if (gemma3_270m_mode || parsed_arch == ARCH_GEMMA3) {
            if (!temp_user) temp = 1.0f;
            if (!top_k_user) top_k = 64;
            if (!top_p_user) top_p = 0.95f;
            if (!min_p_user) min_p = 0.0f;
            if (!g_clean_output) {
                printf("Gemma3 sampler defaults: temp=%.1f top_k=%d top_p=%.2f min_p=%.1f\n",
                       temp, top_k, top_p, min_p);
            }
        }
    }
    smollm2_mode = (template_mentions(&hp, "<|im_start|>") ||
                    template_mentions(&hp, "smollm")) &&
                   !(parsed_arch == ARCH_QWEN2 ||
                     parsed_arch == ARCH_QWEN3 ||
                     parsed_arch == ARCH_QWEN25 ||
                     contains_nocase(hp.tokenizer_pre, "qwen"));
    qwen_chatml_mode = (parsed_arch == ARCH_QWEN2 ||
                        parsed_arch == ARCH_QWEN3 ||
                        parsed_arch == ARCH_QWEN25 ||
                        contains_nocase(hp.tokenizer_pre, "qwen")) &&
                       template_mentions(&hp, "<|im_start|>") &&
                       (do_chat ||
                        contains_nocase(model_path, "instruct") ||
                        contains_nocase(model_path, "-it") ||
                        contains_nocase(hp.chat_template, "assistant") ||
                        contains_nocase(hp.chat_template, "start a conversation") ||
                        contains_nocase(hp.chat_template, "add_generation_prompt"));
    slopllm_mode = template_mentions(&hp, "SlopLLM") ||
                   template_mentions(&hp, "SlopAI");
    tinyllama_mode = template_mentions(&hp, "[INST]") ||
                     template_mentions(&hp, "tinyllama");

    load_tokenizer_auto(model_path, tok_path, ctx);

    {
        int est_dim = (int)hp.embedding_length;
        int est_layers = (int)hp.block_count;
        int est_hidden = (int)hp.feed_forward_length;
        int est_seq = seq_user ? seq_user : ((int)hp.context_length ? (int)hp.context_length : 512);
        unsigned int file_mb = (unsigned int)(ctx->size / (1024*1024));
        unsigned int cache_kb = (unsigned int)((u64)est_layers * (2 * est_dim * est_dim + est_dim * est_hidden) * 4 / 1024);
        unsigned int state_kb = (unsigned int)((u64)est_seq * est_dim * 2 * 4 / 1024);
        if (!g_clean_output) {
            printf("Estimated memory: file=%uMB + matmul_workspace=%uKB + run_state=%uKB\n",
                   file_mb, cache_kb, state_kb);
            printf("Total RAM needed: ~%u MB\n", file_mb + cache_kb/1024 + state_kb/1024);
        }
    }

    if (build_model(ctx, &m, &hp, n_heads_user, n_kv_user, hidden_user, seq_user) != 0) {
        gguf_free(ctx);
        return 1;
    }

    /* Scan special tokens */
    scan_special_tokens();
    init_eog_tokens();
    gpt2_instruct_mode =
        m.arch == ARCH_GPT2 &&
        find_special_token("<|user|>") >= 0 &&
        find_special_token("<|assistant|>") >= 0;
    if (!gpt2_instruct_mode &&
        m.arch == ARCH_GPT2 &&
        contains_nocase(model_path, "gpt2-instruct")) {
        gpt2_instruct_mode = 1;
    }

    if (!eos_user) {
        int tok_eos = hp.tokenizer_eos_token_id > 0 ? (int)hp.tokenizer_eos_token_id : -1;
        if (tok_eos < 0) tok_eos = find_special_token("</s>");
        if (tok_eos < 0) tok_eos = find_special_token("<eos>");
        if (tok_eos < 0) tok_eos = find_special_token("<|endoftext|>");
        if (tok_eos >= 0) eos_token = tok_eos;
    }

    if (do_repl) {
        char repl_buf[8192];
        char file_buf[8192];
        if (do_interactive) {
            print_banner();
            printf("interactive mode\n");
            printf("type /exit to quit\n\n");
        } else if (!g_clean_output) {
            printf("REPL mode: model stays loaded. Type /exit to quit.\n");
        }
        if (prompt_file) {
            if (read_file_text(prompt_file, file_buf, sizeof(file_buf))) {
                if (run_local_request(&m, ctx, &hp, model_path, file_buf, n_generate, temp, top_k, top_p, min_p,
                                      eos_token, no_eos_stop, do_chat, do_raw_tokens,
                                      gpt2_instruct_mode, smollm2_mode, qwen_chatml_mode,
                                      tinyllama_mode, gemma3_mode, slopllm_mode,
                                      do_interactive, n_generate_user) != 0) {
                    free_model(&m);
                    gguf_free(ctx);
                    if (g_tok_buf) { free(g_tok_buf); free(g_vocab); }
                    if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; g_sorted_init = 0; }
                    return 1;
                }
            }
        } else if (prompt && prompt[0]) {
            if (run_local_request(&m, ctx, &hp, model_path, prompt, n_generate, temp, top_k, top_p, min_p,
                                  eos_token, no_eos_stop, do_chat, do_raw_tokens,
                                  gpt2_instruct_mode, smollm2_mode, qwen_chatml_mode,
                                  tinyllama_mode, gemma3_mode, slopllm_mode,
                                  do_interactive, n_generate_user) != 0) {
                free_model(&m);
                gguf_free(ctx);
                if (g_tok_buf) { free(g_tok_buf); free(g_vocab); }
                if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; g_sorted_init = 0; }
                return 1;
            }
        }
        for (;;) {
            size_t len;
            printf(">>> ");
            if (!fgets(repl_buf, sizeof(repl_buf), stdin)) break;
            len = strlen(repl_buf);
            while (len > 0 && (repl_buf[len - 1] == '\n' || repl_buf[len - 1] == '\r')) {
                repl_buf[--len] = '\0';
            }
            sanitize_repl_input(repl_buf);
            if (strcmp(repl_buf, "/exit") == 0 || strcmp(repl_buf, "/quit") == 0) break;
            if (repl_buf[0] == '\0') continue;
            if (run_local_request(&m, ctx, &hp, model_path, repl_buf, n_generate, temp, top_k, top_p, min_p,
                                  eos_token, no_eos_stop, do_chat, do_raw_tokens,
                                  gpt2_instruct_mode, smollm2_mode, qwen_chatml_mode,
                                  tinyllama_mode, gemma3_mode, slopllm_mode,
                                  do_interactive, n_generate_user) != 0) {
                free_model(&m);
                gguf_free(ctx);
                if (g_tok_buf) { free(g_tok_buf); free(g_vocab); }
                if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; g_sorted_init = 0; }
                return 1;
            }
        }
    } else {
        char file_buf[8192];
        const char *request_prompt = prompt;
        if (prompt_file && read_file_text(prompt_file, file_buf, sizeof(file_buf))) {
            request_prompt = file_buf;
        }
        if (run_local_request(&m, ctx, &hp, model_path, request_prompt, n_generate, temp, top_k, top_p, min_p,
                              eos_token, no_eos_stop, do_chat, do_raw_tokens,
                              gpt2_instruct_mode, smollm2_mode, qwen_chatml_mode,
                              tinyllama_mode, gemma3_mode, slopllm_mode,
                              0, n_generate_user) != 0) {
            free_model(&m);
            gguf_free(ctx);
            if (g_tok_buf) { free(g_tok_buf); free(g_vocab); }
            if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; g_sorted_init = 0; }
            return 1;
        }
    }

    free_model(&m);
    gguf_free(ctx);
    if (g_tok_buf) { free(g_tok_buf); free(g_vocab); }
    if (g_sorted_vocab) { free(g_sorted_vocab); g_sorted_vocab = NULL; g_sorted_init = 0; }
    return 0;
}

