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
#include <time.h>

/* --- Types --- */

typedef unsigned __int64 u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed char s8;

typedef struct { float d; u8 qs[16]; } BlockQ4;
typedef struct { float d; s8 qs[32]; } BlockQ8;
typedef struct { u16 d; u8 qh[4]; u8 qs[16]; } BlockQ5_0;

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
} HParams;

typedef struct {
    u8 *base;
    size_t size;
    u64 n_tensors;
    Tensor *tensors;
    u64 data_offset;
    HParams hp;
} GGUFContext;

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, head_dim;
    float norm_eps, rope_theta, rsqrt_head_dim;
    int tok_embd_transposed; /* 1 if token_embd.weight is [dim, vocab] instead of [vocab, dim] */
    float *cached_embd;      /* dequantized embeddings [vocab_size][dim], NULL if not cached */
    float *rope_cos;         /* precomputed RoPE cos table [max_seq_len][head_dim/2] */
    float *rope_sin;         /* precomputed RoPE sin table [max_seq_len][head_dim/2] */
    float **cached_attn_k;   /* per-layer dequantized attn_k, NULL if not cached */
    float **cached_attn_v;   /* per-layer dequantized attn_v, NULL if not cached */
    float **cached_ffn_down; /* per-layer dequantized ffn_down, NULL if not cached */
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

static void parse_metadata(u8 **p, u64 n_kv, HParams *hp) {
    u64 i;
    u32 vtype;
    u64 keylen;
    char key[256];
    memset(hp, 0, sizeof(HParams));
    hp->alignment = 32;
    hp->attention_layer_norm_rms_epsilon = 1e-6f;
    hp->rope_freq_base = 10000.0f;
    for (i = 0; i < n_kv; i++) {
        keylen = read_u64(p);
        if (keylen > 255) keylen = 255;
        memcpy(key, *p, (size_t)keylen); key[keylen] = '\0'; *p += keylen;
        vtype = read_u32(p);
        if (strcmp(key, "general.alignment") == 0 && vtype == 4) {
            hp->alignment = read_u32(p);
        } else if (strcmp(key, "llama.context_length") == 0 && vtype == 4) {
            hp->context_length = read_u32(p);
        } else if (strcmp(key, "llama.embedding_length") == 0 && vtype == 4) {
            hp->embedding_length = read_u32(p);
        } else if (strcmp(key, "llama.feed_forward_length") == 0 && vtype == 4) {
            hp->feed_forward_length = read_u32(p);
        } else if (strcmp(key, "llama.block_count") == 0 && vtype == 4) {
            hp->block_count = read_u32(p);
        } else if (strcmp(key, "llama.attention.head_count") == 0 && vtype == 4) {
            hp->attention_head_count = read_u32(p);
        } else if (strcmp(key, "llama.attention.head_count_kv") == 0 && vtype == 4) {
            hp->attention_head_count_kv = read_u32(p);
        } else if (strcmp(key, "llama.attention.layer_norm_rms_epsilon") == 0 && vtype == 6) {
            float val; memcpy(&val, *p, 4); *p += 4; hp->attention_layer_norm_rms_epsilon = val;
        } else if (strcmp(key, "llama.rope.dimension_count") == 0 && vtype == 4) {
            hp->rope_dimension_count = read_u32(p);
        } else if (strcmp(key, "llama.rope.freq_base") == 0 && vtype == 6) {
            float val; memcpy(&val, *p, 4); *p += 4; hp->rope_freq_base = val;
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

    f = fopen(path, "rb");
    if (!f) { printf("Error: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    file_size = (u64)ftell(f);
    fseek(f, 0, SEEK_SET);

    ctx = (GGUFContext*)malloc(sizeof(GGUFContext));
    ctx->size = (size_t)file_size;
    ctx->base = (u8*)malloc((size_t)file_size);
    read_bytes = fread(ctx->base, 1, (size_t)file_size, f);
    fclose(f);
    if ((u64)read_bytes != file_size) { printf("Error: incomplete read\n"); free(ctx->base); free(ctx); return NULL; }

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
            "token_embd.weight", "tok_embeddings.weight", "model.embed_tokens.weight",
            "token_embeddings.weight", "embedding.weight", "embed.weight", NULL
        };
        u8 *search_end = ctx->base + ctx->size;
        u8 *s;
        u8 *found = NULL;
        if (search_end > ctx->base + 200000) search_end = ctx->base + 200000;
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
    free(ctx->base);
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

static int sample(float *probs, int n, float temp) {
    int i;
    int best;
    float r, cdf;
    float maxv;
    if (temp == 0.0f) {
        best = 0;
        for (i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
        return best;
    }
    for (i = 0; i < n; i++) probs[i] /= temp;
    maxv = probs[0];
    for (i = 1; i < n; i++) if (probs[i] > maxv) maxv = probs[i];
    cdf = 0.0f;
    for (i = 0; i < n; i++) { probs[i] = fast_exp_table(probs[i] - maxv); cdf += probs[i]; }
    for (i = 0; i < n; i++) probs[i] /= cdf;
    r = (float)rand() / (float)RAND_MAX;
    cdf = 0.0f;
    for (i = 0; i < n; i++) {
        cdf += probs[i];
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

static void matvec_q4(const BlockQ4 *w, const float *x, float *y, int n, int d) {
    int nb = d / 32;
    int i, b, j;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ4 *blk = &w[i * nb + b];
            float ds = blk->d;
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
            float ds = blk->d;
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

typedef struct { u16 d[2]; u8 scales[12]; u8 qs[128]; } BlockQ4K;
typedef struct { u16 d[2]; u8 scales[12]; u8 qh[32]; u8 qs[128]; } BlockQ5K;
typedef struct { u8 ql[128]; u8 qh[64]; s8 scales[16]; u16 d; } BlockQ6K;

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

static void matvec_q4k(const BlockQ4K *w, const float *x, float *y, int n, int d) {
    int nb = d / 256;
    int i, b, jj, l;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ4K *blk = &w[i * nb + b];
            float d_all = fp16_to_fp32(blk->d[0]);
            float min_all = fp16_to_fp32(blk->d[1]);
            const u8 *q = blk->qs;
            int is = 0;
            u8 sc, m;
            float d1, m1, d2, m2;
            const float *xb = x + b * 256;
            for (jj = 0; jj < 256; jj += 64) {
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                for (l = 0; l < 32; l++) {
                    sum += (d1 * (q[l] & 0xF) - m1) * xb[jj + l];
                }
                for (l = 0; l < 32; l++) {
                    sum += (d2 * (q[l] >> 4) - m2) * xb[jj + 32 + l];
                }
                q += 32;
                is += 2;
            }
        }
        y[i] = sum;
    }
}

static void matvec_q5k(const BlockQ5K *w, const float *x, float *y, int n, int d) {
    int nb = d / 256;
    int i, b, jj, l;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ5K *blk = &w[i * nb + b];
            float d_all = fp16_to_fp32(blk->d[0]);
            float min_all = fp16_to_fp32(blk->d[1]);
            const u8 *ql = blk->qs;
            const u8 *qh = blk->qh;
            int is = 0;
            u8 sc, m;
            u8 u1 = 1, u2 = 2;
            float d1, m1, d2, m2;
            const float *xb = x + b * 256;
            for (jj = 0; jj < 256; jj += 64) {
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                for (l = 0; l < 32; l++) {
                    int x0 = (ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0);
                    int x1 = (ql[l] >> 4) + (qh[l] & u2 ? 16 : 0);
                    sum += (d1 * (float)x0 - m1) * xb[jj + l + 0]
                         + (d2 * (float)x1 - m2) * xb[jj + l + 32];
                }
                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        y[i] = sum;
    }
}

static void matvec_q6k(const BlockQ6K *w, const float *x, float *y, int n, int d) {
    int nb = d / 256;
    int i, b, n2, l;
    for (i = 0; i < n; i++) {
        float sum = 0.0f;
        for (b = 0; b < nb; b++) {
            const BlockQ6K *blk = &w[i * nb + b];
            float d_all = fp16_to_fp32(blk->d);
            const u8 *ql = blk->ql;
            const u8 *qh = blk->qh;
            const s8 *sc = blk->scales;
            const float *xb = x + b * 256;
            for (n2 = 0; n2 < 256; n2 += 128) {
                for (l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    sum += d_all * (float)sc[is + 0] * (float)q1 * xb[l + 0];
                    sum += d_all * (float)sc[is + 2] * (float)q2 * xb[l + 32];
                    sum += d_all * (float)sc[is + 4] * (float)q3 * xb[l + 64];
                    sum += d_all * (float)sc[is + 6] * (float)q4 * xb[l + 96];
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
            float ds = blk->d;
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
            float ds = blk->d;
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
    } else if (t->type == 12) {
        int nb = d / 256;
        const BlockQ4K *blk = (const BlockQ4K*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 256) {
            float d_all = fp16_to_fp32(blk->d[0]);
            float min_all = fp16_to_fp32(blk->d[1]);
            const u8 *q = blk->qs;
            int is = 0;
            u8 sc, m;
            int jj;
            float d1, m1, d2, m2;
            int l;
            for (jj = 0; jj < 256; jj += 64) {
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                for (l = 0; l < 32; l++) {
                    out[j + jj + l] = d1 * (q[l] & 0xF) - m1;
                }
                for (l = 0; l < 32; l++) {
                    out[j + jj + 32 + l] = d2 * (q[l] >> 4) - m2;
                }
                q += 32;
                is += 2;
            }
            blk++;
        }
    } else if (t->type == 13) {
        int nb = d / 256;
        const BlockQ5K *blk = (const BlockQ5K*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 256) {
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
            for (jj = 0; jj < 256; jj += 64) {
                get_scale_min_k4(is + 0, blk->scales, &sc, &m);
                d1 = d_all * sc;
                m1 = min_all * m;
                get_scale_min_k4(is + 1, blk->scales, &sc, &m);
                d2 = d_all * sc;
                m2 = min_all * m;
                for (l = 0; l < 32; l++) {
                    int x0 = (ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0);
                    int x1 = (ql[l] >> 4) + (qh[l] & u2 ? 16 : 0);
                    out[j + jj + l + 0] = d1 * (float)x0 - m1;
                    out[j + jj + l + 32] = d2 * (float)x1 - m2;
                }
                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
            blk++;
        }
    } else if (t->type == 14) {
        int nb = d / 256;
        const BlockQ6K *blk = (const BlockQ6K*)t->data + row * nb;
        int j;
        for (j = 0; j < d; j += 256) {
            const float d_all = fp16_to_fp32(blk->d);
            const u8 *ql = blk->ql;
            const u8 *qh = blk->qh;
            const s8 *sc = blk->scales;
            float *y = out + j;
            int n2;
            for (n2 = 0; n2 < 256; n2 += 128) {
                int l;
                for (l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l + 0]  = d_all * (float)sc[is + 0] * (float)q1;
                    y[l + 32] = d_all * (float)sc[is + 2] * (float)q2;
                    y[l + 64] = d_all * (float)sc[is + 4] * (float)q3;
                    y[l + 96] = d_all * (float)sc[is + 6] * (float)q4;
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

static float *cache_tensor_f32(const Tensor *t, int n, int d) {
    /* Pre-dequantize a [n, d] quantized tensor to contiguous f32 buffer.
       Returns NULL on failure or if already f32. Caller must free(). */
    float *cache;
    int i;
    float *tmp;
    if (!t || t->type == 0) return NULL;
    cache = (float*)malloc((size_t)n * d * sizeof(float));
    if (!cache) {
        fprintf(stderr, "Warning: failed to allocate f32 cache for %s\n", t->name);
        return NULL;
    }
    tmp = (float*)malloc(d * sizeof(float));
    if (!tmp) {
        free(cache);
        fprintf(stderr, "Warning: failed to allocate tmp row for %s\n", t->name);
        return NULL;
    }
    printf("Dequantizing %s to f32 cache (%u KB)...\n", t->name,
           (unsigned int)((size_t)n * d * sizeof(float) / 1024));
    for (i = 0; i < n; i++) {
        dequantize_row(t, tmp, i, d);
        memcpy(cache + (size_t)i * d, tmp, d * sizeof(float));
    }
    free(tmp);
    return cache;
}

static void matvec(const Tensor *t, const float *x, float *y, int n, int d, float *dq_row) {
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

/* --- Model builder --- */

static int build_model(GGUFContext *ctx, Model *m, HParams *hp, int n_heads_user, int n_kv_user, int hidden_user, int seq_user) {
    int i;
    Tensor *t;
    memset(m, 0, sizeof(Model));

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

    /* Pre-dequantize embeddings to f32 cache for fast lookup */
    if (m->tok_embd->type != 0) {
        int vs = m->vocab_size;
        int dm = m->dim;
        int j, tk;
        float *tmp = (float*)malloc(vs * sizeof(float));
        m->cached_embd = (float*)malloc((size_t)vs * dm * sizeof(float));
        if (!tmp || !m->cached_embd) {
            fprintf(stderr, "Warning: failed to allocate embedding cache, using slow dequant path\n");
            if (tmp) free(tmp);
            if (m->cached_embd) { free(m->cached_embd); m->cached_embd = NULL; }
        } else {
            printf("Dequantizing embeddings to f32 cache (%u KB)...\n", (unsigned int)((size_t)vs * dm * sizeof(float) / 1024));
            if (m->tok_embd_transposed) {
                for (j = 0; j < dm; j++) {
                    dequantize_row(m->tok_embd, tmp, j, vs);
                    for (tk = 0; tk < vs; tk++) {
                        m->cached_embd[tk * dm + j] = tmp[tk];
                    }
                }
            } else {
                for (tk = 0; tk < vs; tk++) {
                    dequantize_row(m->tok_embd, tmp, tk, dm);
                    memcpy(m->cached_embd + tk * dm, tmp, dm * sizeof(float));
                }
            }
            printf("Embedding cache ready.\n");
            free(tmp);
        }
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

    if (seq_user > 0) m->max_seq_len = seq_user;
    else if (hp->context_length > 0) m->max_seq_len = (int)hp->context_length;
    else m->max_seq_len = 2048;

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
    }

    /* Pre-dequantize heaviest weights to f32 for fast matmul */
    {
        int l;
        m->cached_attn_k = (float**)malloc(sizeof(float*) * m->n_layers);
        m->cached_attn_v = (float**)malloc(sizeof(float*) * m->n_layers);
        m->cached_ffn_down = (float**)malloc(sizeof(float*) * m->n_layers);
        for (l = 0; l < m->n_layers; l++) {
            m->cached_attn_k[l] = NULL;
            m->cached_attn_v[l] = NULL;
            m->cached_ffn_down[l] = NULL;
        }
        for (l = 0; l < m->n_layers; l++) {
            int nk = m->n_kv_heads * m->head_dim;
            m->cached_attn_k[l] = cache_tensor_f32(m->attn_k[l], nk, m->dim);
            m->cached_attn_v[l] = cache_tensor_f32(m->attn_v[l], nk, m->dim);
            m->cached_ffn_down[l] = cache_tensor_f32(m->ffn_down[l], m->dim, m->hidden_dim);
        }
    }

    printf("Model: dim=%d layers=%d heads=%d kv_heads=%d hidden=%d vocab=%d seq=%d\n",
           m->dim, m->n_layers, m->n_heads, m->n_kv_heads, m->hidden_dim, m->vocab_size, m->max_seq_len);
    return 0;
}

static void free_model(Model *m) {
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

static int forward(Model *m, RunState *s, int token, int pos, float temp) {
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
        memcpy(x, m->cached_embd + (size_t)token * dim, dim * sizeof(float));
    } else if (m->tok_embd_transposed) {
        int j;
        for (j = 0; j < dim; j++) {
            dequantize_row(m->tok_embd, s->dq_row, j, vocab_size);
            x[j] = s->dq_row[token];
        }
    } else {
        dequantize_row(m->tok_embd, x, token, dim);
    }

    printf("[tok %d] ", pos);
    fflush(stdout);
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
        if (l % 5 == 4) { printf("."); fflush(stdout); }
    }

    rmsnorm(x, (float*)m->output_norm->data, dim, m->norm_eps);
    if (m->cached_embd && m->output == m->tok_embd) {
        /* Fast path: cached f32 embeddings for output projection */
        matvec_f32(m->cached_embd, x, s->logits, vocab_size, dim);
    } else {
        matvec(m->output, x, s->logits, vocab_size, dim, s->dq_row);
    }
    return sample(s->logits, vocab_size, temp);
}

/* --- Tokenizer --- */

static int load_tokenizer(const char *path) {
    FILE *f;
    size_t sz;
    u8 *buf;
    u32 count;
    int i;
    size_t pos;
    if (g_tok_buf) { free(g_tok_buf); g_tok_buf = NULL; free(g_vocab); g_vocab = NULL; }
    f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    buf = (u8*)malloc(sz);
    fread(buf, 1, sz, f); fclose(f);
    memcpy(&count, buf, 4);
    g_vocab = (TokenEntry*)malloc(sizeof(TokenEntry) * count);
    g_vocab_n = (int)count;
    pos = 4;
    for (i = 0; i < (int)count && pos < sz; i++) {
        int tlen = (int)buf[pos++];
        g_vocab[i].len = tlen;
        g_vocab[i].text = (char*)(buf + pos);
        pos += tlen;
    }
    g_tok_buf = buf;
    printf("Loaded tokenizer: %d tokens\n", g_vocab_n);
    return 1;
}

static int tokenize(const char *text, int *tokens, int max_tokens, int vocab_size) {
    int len = (int)strlen(text);
    int pos = 0;
    int n = 0;
    int i;
    while (pos < len && n < max_tokens) {
        int best = -1;
        int best_len = 0;
        if (g_vocab) {
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
            if (vocab_size == 256) tokens[n++] = (unsigned char)text[pos];
            else tokens[n++] = (int)(unsigned char)text[pos];
            pos++;
        }
    }
    return n;
}

static void detokenize(int token) {
    if (g_vocab && token >= 0 && token < g_vocab_n && g_vocab[token].len > 0) {
        fwrite(g_vocab[token].text, 1, g_vocab[token].len, stdout);
    } else {
        if (token >= 32 && token < 127) printf("%c", (char)token);
        else printf("<%d>", token);
    }
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
    int n_generate = 50;
    float temp = 0.8f;
    unsigned int seed = 0;
    int n_heads_user = 0;
    int n_kv_user = 0;
    int hidden_user = 0;
    int seq_user = 0;
    int do_list = 0;
    int arg;
    int pos;
    int prompt_len;
    int *prompt_tokens;
    int next_token = 0;
    int i;

    if (argc < 2) {
        printf("Usage: gguf_infer.exe <model.gguf> [options]\n");
        printf("Options:\n");
        printf("  --list          List all tensors and exit\n");
        printf("  -n <num>        Max tokens to generate (default 50)\n");
        printf("  -t <temp>       Temperature, 0=argmax (default 0.8)\n");
        printf("  -s <seed>       Random seed (default time)\n");
        printf("  --prompt <s>    Prompt string\n");
        printf("  --tok <file>    tokenizer.bin (optional)\n");
        printf("  --n_heads <n>   Override head count\n");
        printf("  --n_kv <n>      Override KV head count\n");
        printf("  --hidden <n>    Override hidden dim\n");
        printf("  --seq <n>       Override max sequence length\n");
        return 1;
    }

    model_path = argv[1];
    for (arg = 2; arg < argc; arg++) {
        if (strcmp(argv[arg], "-n") == 0 && arg + 1 < argc) { n_generate = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "-t") == 0 && arg + 1 < argc) { temp = (float)atof(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "-s") == 0 && arg + 1 < argc) { seed = (unsigned int)atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--prompt") == 0 && arg + 1 < argc) { prompt = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--tok") == 0 && arg + 1 < argc) { tok_path = argv[arg+1]; arg++; }
        else if (strcmp(argv[arg], "--n_heads") == 0 && arg + 1 < argc) { n_heads_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--n_kv") == 0 && arg + 1 < argc) { n_kv_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--hidden") == 0 && arg + 1 < argc) { hidden_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--seq") == 0 && arg + 1 < argc) { seq_user = atoi(argv[arg+1]); arg++; }
        else if (strcmp(argv[arg], "--list") == 0) { do_list = 1; }
    }

    if (seed == 0) seed = (unsigned int)time(NULL);
    srand(seed);

    ctx = gguf_load(model_path);
    if (!ctx) return 1;

    if (do_list) {
        list_tensors(ctx);
        gguf_free(ctx);
        return 0;
    }

    hp = ctx->hp;

    if (tok_path && tok_path[0]) load_tokenizer(tok_path);

    if (build_model(ctx, &m, &hp, n_heads_user, n_kv_user, hidden_user, seq_user) != 0) {
        gguf_free(ctx);
        return 1;
    }

    if (alloc_runstate(&m, &s, ctx) != 0) {
        free_model(&m);
        gguf_free(ctx);
        return 1;
    }

    prompt_tokens = (int*)malloc(m.max_seq_len * sizeof(int));
    prompt_len = tokenize(prompt, prompt_tokens, m.max_seq_len, m.vocab_size);

    printf("Prompt: '%s' (%d tokens)\n", prompt, prompt_len);
    printf("Max tokens: %d | temp: %.2f | seed: %u | seq: %d\n", n_generate, temp, seed, m.max_seq_len);
    printf("---\n");

    /* Prefill: process all prompt tokens silently to fill KV cache */
    {
        clock_t prefill_t0 = clock();
        for (pos = 0; pos < prompt_len; pos++) {
            next_token = forward(&m, &s, prompt_tokens[pos], pos, 0.0f);
        }
        {
            clock_t prefill_t1 = clock();
            double prefill_sec = (double)(prefill_t1 - prefill_t0) / (double)CLOCKS_PER_SEC;
            printf("[Prefill] %d prompt tokens in %.2f sec (%.2f sec/token)\n",
                   prompt_len, prefill_sec,
                   prompt_len > 0 ? prefill_sec / prompt_len : 0.0);
            printf("---\n");
        }
    }

    /* Generation: stream tokens with live TPS */
    {
        clock_t gen_t0 = clock();
        clock_t now;
        int last_i = 0;
        for (i = 0; i < n_generate; i++) {
            if (pos >= m.max_seq_len) break;
            next_token = forward(&m, &s, next_token, pos, temp);
            detokenize(next_token);
            fflush(stdout);
            pos++;

            /* Stop on EOS: token 0 or last vocab token (e.g. 49151) */
            if (next_token == 0 || next_token == m.vocab_size - 1) {
                printf(" [EOS:%d]", next_token);
                fflush(stdout);
                i++;
                break;
            }

            /* Report TPS every 5 tokens or on last token */
            now = clock();
            if (i - last_i >= 4 || i == n_generate - 1 || pos >= m.max_seq_len) {
                double gen_elapsed = (double)(now - gen_t0) / (double)CLOCKS_PER_SEC;
                int gen_count = i + 1;
                double tps = gen_elapsed > 0 ? (double)gen_count / gen_elapsed : 0.0;
                double tpk = gen_count > 0 ? gen_elapsed / gen_count : 0.0;
                printf(" [T+%d | %.2fs | %.2f TPS | %.2fs/T]",
                       gen_count, gen_elapsed, tps, tpk);
                fflush(stdout);
                last_i = i;
            }
        }
        {
            clock_t gen_t1 = clock();
            double gen_sec = (double)(gen_t1 - gen_t0) / (double)CLOCKS_PER_SEC;
            int gen_count = i;
            double tps = gen_sec > 0 ? (double)gen_count / gen_sec : 0.0;
            printf("\n---\n");
            printf("[Done] Generated %d tokens in %.2f sec | Avg %.2f TPS | %.2f sec/token | %.1f tokens/min\n",
                   gen_count, gen_sec, tps,
                   gen_count > 0 ? gen_sec / gen_count : 0.0,
                   gen_sec > 0 ? (60.0 * gen_count) / gen_sec : 0.0);
        }
    }

    free(prompt_tokens);
    free_runstate(&s);
    free_model(&m);
    gguf_free(ctx);
    if (g_tok_buf) { free(g_tok_buf); free(g_vocab); }
    return 0;
}
