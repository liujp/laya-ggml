/*
 * laya-infer.c — Lightweight Laya decision model inference with ggml.
 *
 * Architecture: ModernBERT encoder (22 layers, GeGLU FFN, RoPE, pre-norm)
 *             + type_emb + 2-layer decision head + scorer + act_head
 *
 * Usage:
 *   echo '{"input_ids":[...],"marker_pos":[...],"qtype":0}' | ./laya-infer model.gguf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

/* Wall-clock helper for the per-call timing printed to stderr. */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#define MAX_TOKENS   1024
#define MAX_MARKERS  64
#define MAX_NAME_LEN 128
#define ALIGN32(x)   (((x) + 31) & ~31)
#define MAX_GRAPH_NODES 4096
#define MAX_BATCH    64    /* slots per request in {"items":[...]} form */
#define MAX_PACKED_MARKERS 1024  /* sum of markers across a packed request */
#define MAX_LINE     (1 << 20)

/* Several slots packed into one sequence. They are concatenated (not padded to a
 * common length) and kept apart by a block-diagonal attention mask, so the model
 * still sees a plain 2-D sequence and the graph structure is unchanged -- only
 * the inputs and the mask differ. One forward pass then answers every slot. */
typedef struct {
    int n_slots;
    int kmax;                    /* markers per slot after padding */
    int seg_len[MAX_BATCH];      /* tokens per slot */
    int seg_qtype[MAX_BATCH];    /* qtype per slot */
    int seg_k[MAX_BATCH];        /* real markers per slot (rest is padding) */
} packed_t;

/* ---------------------------------------------------------------- model */

struct laya_enc_layer {
    struct ggml_tensor * attn_norm_w;
    struct ggml_tensor * Wqkv, * Wo;
    struct ggml_tensor * mlp_norm_w;
    struct ggml_tensor * Wi, * Wo_mlp;
};

struct laya_head_layer {
    struct ggml_tensor * norm1_w, * norm1_b;
    struct ggml_tensor * in_proj_w, * in_proj_b;
    struct ggml_tensor * out_proj_w, * out_proj_b;
    struct ggml_tensor * norm2_w, * norm2_b;
    struct ggml_tensor * linear1_w, * linear1_b;
    struct ggml_tensor * linear2_w, * linear2_b;
};

struct laya_model {
    int n_layers, n_heads, hidden, intermediate, vocab, max_pos, head_layers;
    int local_attention;      /* sliding window size (0 = global) */
    int global_attn_every_n;  /* every N layers is global, rest is sliding */
    float rope_theta;
    float temperatures[3];

    struct ggml_tensor * tok_emb, * emb_norm_w, * final_norm_w, * type_emb;
    struct laya_enc_layer * enc;
    struct laya_head_layer * head;
    struct ggml_tensor * sc_ln_w, * sc_ln_b, * sc_w1, * sc_b1, * sc_w2, * sc_b2;
    struct ggml_tensor * ah_w1, * ah_b1, * ah_w2, * ah_b2;

    /* Single context holds weight tensors (metadata only; data is mmap/malloc'd) */
    struct ggml_context * ctx_w;
    void * weight_data;

    /* Backend: CPU or CUDA */
    ggml_backend_t backend;
    ggml_backend_buffer_t weight_buf;

    /* Cached compute graph. Rebuilding the graph costs ~6 ms of CPU time and
     * forces a fresh CUDA graph capture (~50 ms) on the first run. Keeping the
     * built graph around and only swapping the input data lets repeat calls go
     * straight to a replay. Invalidated whenever (n_tok, n_markers, qtype). */
    struct {
        bool ready;
        int L, K, qtype;
        struct ggml_context * ctx;
        ggml_backend_sched_t sched;
        struct ggml_cgraph * gf;
        ggml_backend_t inp_backend;      /* CPU backend owning the input buffers */
        ggml_backend_buffer_t inp_buf;   /* ids / marker positions / pos ids */
        ggml_backend_buffer_t aux_buf;   /* per-token qtype ids + logit padding mask */
        ggml_backend_buffer_t mask_buf;  /* sliding-window / block-diagonal masks */
        struct ggml_tensor * inp, * inp_m, * pos, * sw_mask, * blk_mask;
        ggml_backend_t backends[2];
        int n_backends;
        /* Packed (multi-slot) request description, for splitting the output */
        bool packed;
        int n_slots, kmax;
        int seg_k[MAX_BATCH];
    } r;
};

/* ---------------------------------------------------------------- GGUF reader */

static uint64_t read_u64(FILE * f) { uint64_t v; fread(&v, 8, 1, f); return v; }
static uint32_t read_u32(FILE * f) { uint32_t v; fread(&v, 4, 1, f); return v; }
static float    read_f32(FILE * f) { float v;    fread(&v, 4, 1, f); return v; }

static char * read_string(FILE * f) {
    uint64_t len = read_u64(f);
    char * s = (char *)malloc(len + 1);
    fread(s, 1, len, f);
    s[len] = '\0';
    return s;
}

static void skip_kv_value(FILE * f, uint32_t vtype) {
    switch (vtype) {
        case 0: case 1: fseek(f, 1, SEEK_CUR); break;
        case 2: case 3: fseek(f, 2, SEEK_CUR); break;
        case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break;
        case 7: fseek(f, 8, SEEK_CUR); break;
        case 8: { char * s = read_string(f); free(s); break; }
        case 9: {
            uint32_t atype = read_u32(f);
            uint64_t alen = read_u64(f);
            for (uint64_t i = 0; i < alen; i++) skip_kv_value(f, atype);
            break;
        }
        default: fprintf(stderr, "unknown kv type %u\n", vtype); exit(1);
    }
}

static void read_kv_float_array(FILE * f, float * out, int n) {
    uint32_t vtype = read_u32(f);
    if (vtype != 9) { skip_kv_value(f, vtype); return; }
    uint32_t atype = read_u32(f);
    uint64_t alen = read_u64(f);
    (void)atype;
    for (uint64_t i = 0; i < alen && (int)i < n; i++) out[i] = read_f32(f);
}

struct tensor_info {
    char name[MAX_NAME_LEN];
    uint32_t ndim;
    uint64_t shape[4];
    uint32_t dtype;
    uint64_t offset;
};

/* Compute the byte size of a tensor given its type and element count */
static size_t tensor_nbytes(uint32_t dtype, uint64_t n_elements) {
    switch (dtype) {
        case 0: return n_elements * 4;  /* F32 */
        case 1: return n_elements * 2;  /* F16 */
        default: return n_elements * 4;
    }
}

static struct laya_model * load_model(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }

    if (read_u32(f) != 0x46554747) { fprintf(stderr, "not GGUF\n"); fclose(f); return NULL; }
    read_u32(f); /* version */
    uint64_t n_tensors = read_u64(f);
    uint64_t n_kv = read_u64(f);

    struct laya_model * m = (struct laya_model *)calloc(1, sizeof(*m));
    m->n_layers = 22; m->n_heads = 12; m->hidden = 768; m->intermediate = 1152;
    m->vocab = 256000; m->max_pos = 8192; m->head_layers = 2;
    m->local_attention = 128; m->global_attn_every_n = 3; m->rope_theta = 160000.0f;
    m->temperatures[0] = m->temperatures[1] = m->temperatures[2] = 1.0f;

    /* Read metadata KV */
    for (uint64_t i = 0; i < n_kv; i++) {
        char * key = read_string(f);
        if      (!strcmp(key, "laya.hidden_size"))             { read_u32(f); m->hidden = read_u32(f); }
        else if (!strcmp(key, "laya.num_layers"))              { read_u32(f); m->n_layers = read_u32(f); }
        else if (!strcmp(key, "laya.num_heads"))               { read_u32(f); m->n_heads = read_u32(f); }
        else if (!strcmp(key, "laya.intermediate_size"))       { read_u32(f); m->intermediate = read_u32(f); }
        else if (!strcmp(key, "laya.vocab_size"))              { read_u32(f); m->vocab = read_u32(f); }
        else if (!strcmp(key, "laya.max_position_embeddings")) { read_u32(f); m->max_pos = read_u32(f); }
        else if (!strcmp(key, "laya.head_layers"))             { read_u32(f); m->head_layers = read_u32(f); }
        else if (!strcmp(key, "laya.temperatures"))            read_kv_float_array(f, m->temperatures, 3);
        else { uint32_t vt = read_u32(f); skip_kv_value(f, vt); }
        free(key);
    }

    fprintf(stderr, "[laya] %d layers, %dd, %d heads, vocab %d\n",
            m->n_layers, m->hidden, m->n_heads, m->vocab);

    /* Read tensor infos */
    struct tensor_info * tinfo = calloc(n_tensors, sizeof(struct tensor_info));
    size_t total_data = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        char * name = read_string(f);
        strncpy(tinfo[i].name, name, MAX_NAME_LEN - 1);
        free(name);
        tinfo[i].ndim = read_u32(f);
        uint64_t nel = 1;
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) {
            tinfo[i].shape[d] = read_u64(f);
            nel *= tinfo[i].shape[d];
        }
        tinfo[i].dtype = read_u32(f);
        tinfo[i].offset = read_u64(f);
        size_t nb = tensor_nbytes(tinfo[i].dtype, nel);
        size_t end = tinfo[i].offset + nb;
        if (end > total_data) total_data = end;
    }

    long data_start = ALIGN32(ftell(f));

    /* Read ALL tensor data into a temporary CPU buffer */
    total_data = ALIGN32(total_data + 64);
    void * raw_data = malloc(total_data);
    if (!raw_data) { fprintf(stderr, "OOM: need %zu bytes\n", total_data); exit(1); }
    fseek(f, data_start, SEEK_SET);
    fread(raw_data, 1, total_data, f);
    fclose(f);

    /* Initialize backend: prefer CUDA, fall back to CPU */
    m->backend = NULL;
#ifdef GGML_USE_CUDA
    if (!getenv("LAYA_CPU")) {
        m->backend = ggml_backend_cuda_init(0);
        if (m->backend) fprintf(stderr, "[laya] CUDA backend initialized\n");
    }
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        fprintf(stderr, "[laya] CPU backend initialized\n");
    }

    /* Create tensor metadata (no_alloc: backend will allocate) */
    size_t ctx_size = ggml_tensor_overhead() * (n_tensors + 16) + 1024 * 1024;
    struct ggml_init_params wp = { .mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = true };
    m->ctx_w = ggml_init(wp);

    m->enc = calloc(m->n_layers, sizeof(struct laya_enc_layer));
    m->head = calloc(m->head_layers, sizeof(struct laya_head_layer));

    /* Phase 1: Create tensor metadata and map to model struct */
    for (uint64_t i = 0; i < n_tensors; i++) {
        struct tensor_info * ti = &tinfo[i];
        /* Quantized weights are expanded to f32 on load. The graph mixes
         * activations with f32 norms and biases, and ggml rejects mixed-type
         * elementwise ops -- an f16 model used to abort inside ggml_mul. This
         * keeps one dtype throughout: the file stays small, memory does not. */
        enum ggml_type gtype = (enum ggml_type)ti->dtype;
        enum ggml_type wtype = (gtype == GGML_TYPE_F16 || gtype == GGML_TYPE_Q8_0)
                               ? GGML_TYPE_F32 : gtype;

        struct ggml_tensor * t;
        if (ti->ndim == 1)
            t = ggml_new_tensor_1d(m->ctx_w, wtype, ti->shape[0]);
        else
            t = ggml_new_tensor_2d(m->ctx_w, wtype, ti->shape[0], ti->shape[1]);
        ggml_set_name(t, ti->name);

        /* Map to model struct */
        int layer;
        if (!strcmp(ti->name, "encoder.embeddings.tok_embeddings.weight")) m->tok_emb = t;
        else if (!strcmp(ti->name, "encoder.embeddings.norm.weight"))      m->emb_norm_w = t;
        else if (!strcmp(ti->name, "encoder.final_norm.weight"))           m->final_norm_w = t;
        else if (!strcmp(ti->name, "type_emb.weight"))                     m->type_emb = t;
        else if (!strcmp(ti->name, "scorer.0.weight"))  m->sc_ln_w = t;
        else if (!strcmp(ti->name, "scorer.0.bias"))    m->sc_ln_b = t;
        else if (!strcmp(ti->name, "scorer.1.weight"))  m->sc_w1 = t;
        else if (!strcmp(ti->name, "scorer.1.bias"))    m->sc_b1 = t;
        else if (!strcmp(ti->name, "scorer.3.weight"))  m->sc_w2 = t;
        else if (!strcmp(ti->name, "scorer.3.bias"))    m->sc_b2 = t;
        else if (!strcmp(ti->name, "act_head.0.weight")) m->ah_w1 = t;
        else if (!strcmp(ti->name, "act_head.0.bias"))   m->ah_b1 = t;
        else if (!strcmp(ti->name, "act_head.2.weight")) m->ah_w2 = t;
        else if (!strcmp(ti->name, "act_head.2.bias"))   m->ah_b2 = t;
        else if (sscanf(ti->name, "encoder.layers.%d.", &layer) == 1 && layer < m->n_layers) {
            struct laya_enc_layer * el = &m->enc[layer];
            if      (strstr(ti->name, "attn_norm.weight"))  el->attn_norm_w = t;
            else if (strstr(ti->name, "attn.Wqkv.weight"))  el->Wqkv = t;
            else if (strstr(ti->name, "attn.Wo.weight"))     el->Wo = t;
            else if (strstr(ti->name, "mlp_norm.weight"))    el->mlp_norm_w = t;
            else if (strstr(ti->name, "mlp.Wi.weight"))      el->Wi = t;
            else if (strstr(ti->name, "mlp.Wo.weight"))      el->Wo_mlp = t;
        }
        else if (sscanf(ti->name, "head.layers.%d.", &layer) == 1 && layer < m->head_layers) {
            struct laya_head_layer * hl = &m->head[layer];
            if      (strstr(ti->name, "norm1.weight"))              hl->norm1_w = t;
            else if (strstr(ti->name, "norm1.bias"))                hl->norm1_b = t;
            else if (strstr(ti->name, "self_attn.in_proj_weight"))  hl->in_proj_w = t;
            else if (strstr(ti->name, "self_attn.in_proj_bias"))    hl->in_proj_b = t;
            else if (strstr(ti->name, "self_attn.out_proj.weight")) hl->out_proj_w = t;
            else if (strstr(ti->name, "self_attn.out_proj.bias"))   hl->out_proj_b = t;
            else if (strstr(ti->name, "norm2.weight"))              hl->norm2_w = t;
            else if (strstr(ti->name, "norm2.bias"))                hl->norm2_b = t;
            else if (strstr(ti->name, "linear1.weight"))            hl->linear1_w = t;
            else if (strstr(ti->name, "linear1.bias"))              hl->linear1_b = t;
            else if (strstr(ti->name, "linear2.weight"))            hl->linear2_w = t;
            else if (strstr(ti->name, "linear2.bias"))              hl->linear2_b = t;
        }
    }
    /* Verify critical tensors */
    if (!m->tok_emb || !m->emb_norm_w || !m->final_norm_w || !m->type_emb) {
        fprintf(stderr, "MISSING critical tensor\n"); exit(1);
    }
    for (int i = 0; i < m->n_layers; i++) {
        if (!m->enc[i].Wqkv || !m->enc[i].Wo || !m->enc[i].mlp_norm_w || !m->enc[i].Wi || !m->enc[i].Wo_mlp) {
            fprintf(stderr, "MISSING encoder layer %d tensor\n", i); exit(1);
        }
    }

    /* Phase 2: Allocate weight storage on the backend (CPU or GPU) */
    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_w, m->backend);
    if (!m->weight_buf) {
        fprintf(stderr, "[laya] backend alloc failed\n"); exit(1);
    }

    /* Phase 3: Upload raw data into backend-managed buffers.
     * tinfo still holds the file offsets for each tensor. We walk the context
     * tensors in creation order (same as tinfo order) and upload each one. */
    {
        struct ggml_tensor * t = ggml_get_first_tensor(m->ctx_w);
        uint64_t ti_idx = 0;
        while (t && ti_idx < n_tensors) {
            const char * src = (char *)raw_data + tinfo[ti_idx].offset;
            enum ggml_type src_type = (enum ggml_type)tinfo[ti_idx].dtype;
            int64_t n = ggml_nelements(t);

            if (src_type == GGML_TYPE_F16) {
                const ggml_fp16_t * h = (const ggml_fp16_t *)src;
                float * tmp = (float *)malloc((size_t)n * sizeof(float));
                for (int64_t j = 0; j < n; j++) tmp[j] = ggml_fp16_to_fp32(h[j]);
                ggml_backend_tensor_set(t, tmp, 0, (size_t)n * sizeof(float));
                free(tmp);
            } else if (src_type == GGML_TYPE_Q8_0) {
                /* Q8_0: 2-byte f16 scale then 32 int8 quants, per 32 elements. */
                float * tmp = (float *)malloc((size_t)n * sizeof(float));
                const uint8_t * p = (const uint8_t *)src;
                int64_t done = 0;
                while (done < n) {
                    ggml_fp16_t s;
                    memcpy(&s, p, sizeof(s));
                    float scale = ggml_fp16_to_fp32(s);
                    p += sizeof(s);
                    int64_t take = (n - done) < 32 ? (n - done) : 32;
                    for (int64_t j = 0; j < take; j++)
                        tmp[done + j] = scale * (float)((int8_t)p[j]);
                    p += 32;
                    done += take;
                }
                ggml_backend_tensor_set(t, tmp, 0, (size_t)n * sizeof(float));
                free(tmp);
            } else {
                ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
            }
            t = ggml_get_next_tensor(m->ctx_w, t);
            ti_idx++;
        }
    }

    bool is_gpu = !ggml_backend_is_cpu(m->backend);
    fprintf(stderr, "[laya] weights on %s (%.0f MB)\n",
            is_gpu ? "GPU" : "CPU",
            ggml_backend_buffer_get_size(m->weight_buf) / 1048576.0);

    free(tinfo);
    free(raw_data);
    m->weight_data = NULL;
    return m;
}

/* ---------------------------------------------------------------- graph ops */

static struct ggml_tensor * ln_nobias(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * w) {
    return ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w);
}

static struct ggml_tensor * ln(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * w, struct ggml_tensor * b) {
    x = ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w);
    return b ? ggml_add(ctx, x, b) : x;
}

static struct ggml_tensor * mha(struct ggml_context * ctx,
                                struct ggml_tensor * h,
                                struct ggml_tensor * Wqkv, struct ggml_tensor * Wqkv_b,
                                struct ggml_tensor * Wo,   struct ggml_tensor * Wo_b,
                                int n_heads, bool use_rope,
                                struct ggml_tensor * pos_ids, float rope_theta,
                                struct ggml_tensor * sw_mask) {
    int64_t H = h->ne[0], L = h->ne[1];
    int hd = (int)(H / n_heads);

    struct ggml_tensor * qkv = ggml_mul_mat(ctx, Wqkv, h);
    if (Wqkv_b) qkv = ggml_add(ctx, qkv, Wqkv_b);

    /* Split QKV. qkv is [3H, L]; the copy per branch is deliberate. Addressing
     * [hd, n_heads, L] as a strided view instead removes 66 kernels from the
     * graph but leaves rope/permute reading non-contiguous memory, which measured
     * ~65% slower on GPU -- the copies are cheaper than the strided kernels. */
    size_t es = ggml_element_size(qkv);
    struct ggml_tensor * q = ggml_reshape_3d(ctx,
        ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, L, qkv->nb[1], 0)), hd, n_heads, L);
    struct ggml_tensor * k = ggml_reshape_3d(ctx,
        ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, L, qkv->nb[1], H * es)), hd, n_heads, L);
    struct ggml_tensor * v = ggml_reshape_3d(ctx,
        ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, L, qkv->nb[1], 2 * H * es)), hd, n_heads, L);

    if (use_rope && pos_ids) {
        /* ModernBERT uses rotate_half (split-half) = GGML_ROPE_TYPE_NEOX (mode=2) */
        q = ggml_rope_ext(ctx, q, pos_ids, NULL, hd, 2, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos_ids, NULL, hd, 2, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    /* Permute [hd, n_heads, L] → swap ne[1] and ne[2] for batched matmul */
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    /* Fused attention. The manual version was mul_mat(k,q) -> scale -> add mask ->
     * softmax -> permute+cont(V) -> mul_mat(Vt,scores) -> permute+cont: about seven
     * kernels per layer, most of them tiny. ggml_flash_attn_ext does the whole
     * thing in one, takes the additive mask directly, and returns
     * [n_embd, n_head, n_batch] -- already the head-major layout the reshape below
     * wants, so the trailing permute disappears as well. */
    struct ggml_tensor * attn = ggml_flash_attn_ext(
        ctx, q, k, v, sw_mask, 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
    attn = ggml_cont(ctx, attn);              /* [hd, n_heads, L] */
    attn = ggml_reshape_2d(ctx, attn, H, L);  /* [H, L] */

    struct ggml_tensor * out = ggml_mul_mat(ctx, Wo, attn);
    if (Wo_b) out = ggml_add(ctx, out, Wo_b);
    return out;
}

static struct ggml_tensor * geglu_ffn(struct ggml_context * ctx, struct ggml_tensor * x,
                                      struct ggml_tensor * Wi, struct ggml_tensor * Wo,
                                      int intermediate) {
    struct ggml_tensor * gu = ggml_mul_mat(ctx, Wi, x);
    int64_t L = gu->ne[1];
    /* Both halves are copied before the element-wise work. Keeping them as views
     * and copying only the product removed a kernel but left ggml_mul reading
     * strided memory, which cost ~70% end to end on GPU. */
    struct ggml_tensor * gate = ggml_cont(ctx, ggml_view_2d(ctx, gu, intermediate, L, gu->nb[1], 0));
    struct ggml_tensor * up   = ggml_cont(ctx, ggml_view_2d(ctx, gu, intermediate, L, gu->nb[1],
                                                             intermediate * ggml_element_size(gu)));
    return ggml_mul_mat(ctx, Wo, ggml_mul(ctx, ggml_gelu(ctx, gate), up));
}

static struct ggml_tensor * gelu_ffn(struct ggml_context * ctx, struct ggml_tensor * x,
                                     struct ggml_tensor * W1, struct ggml_tensor * b1,
                                     struct ggml_tensor * W2, struct ggml_tensor * b2) {
    struct ggml_tensor * h = ggml_mul_mat(ctx, W1, x);
    if (b1) h = ggml_add(ctx, h, b1);
    h = ggml_gelu(ctx, h);
    h = ggml_mul_mat(ctx, W2, h);
    if (b2) h = ggml_add(ctx, h, b2);
    return h;
}

/* ---------------------------------------------------------------- build + compute */

/* Release everything the cached graph owns. */
static void runner_free(struct laya_model * m) {
    struct laya_model * r_dummy = m; /* keep accesses below grouped */
    (void)r_dummy;
    if (m->r.sched)    ggml_backend_sched_free(m->r.sched);
    if (m->r.mask_buf) ggml_backend_buffer_free(m->r.mask_buf);
    if (m->r.aux_buf)  ggml_backend_buffer_free(m->r.aux_buf);
    if (m->r.inp_buf)  ggml_backend_buffer_free(m->r.inp_buf);
    /* The fallback CPU backend is created only when the primary is not CPU */
    if (m->r.n_backends > 1) ggml_backend_free(m->r.backends[1]);
    if (m->r.inp_backend)    ggml_backend_free(m->r.inp_backend);
    if (m->r.ctx)            ggml_free(m->r.ctx);
    memset(&m->r, 0, sizeof(m->r));
}

/* Build the compute graph for one shape. Called once per shape; run_inference
 * then only swaps input data and replays.
 * pk == NULL: one slot. pk != NULL: several slots concatenated into one
 * sequence, separated by a block-diagonal mask. */
static bool runner_build(struct laya_model * m, int L, int K, int qtype,
                         const packed_t * pk) {
    int H = m->hidden;

    /* Compute context: tensor metadata only; sched allocates the actual memory */
    size_t ctx_mem = ggml_tensor_overhead() * MAX_GRAPH_NODES + 16 * 1024 * 1024;
    struct ggml_init_params cp = { .mem_size = ctx_mem, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(cp);
    if (!ctx) { fprintf(stderr, "OOM: compute context\n"); return false; }

    /* Input tensors hold data we supply, so they need a real buffer (the sched
     * rejects tensors with no buffer). Allocate one small CPU buffer for all of
     * them; on the GPU path the sched copies them over automatically. */
    ggml_backend_t inp_backend = ggml_backend_cpu_init();
    size_t inp_bytes = (size_t)(2 * L + K) * sizeof(int32_t) + 128;
    ggml_backend_buffer_t inp_buf = ggml_backend_alloc_buffer(inp_backend, inp_bytes);
    if (!inp_buf) { fprintf(stderr, "OOM: input buffer\n"); ggml_free(ctx); return false; }

    struct ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(inp, "inp_ids");
    ggml_set_input(inp);
    struct ggml_tensor * inp_m = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    ggml_set_name(inp_m, "inp_markers");
    ggml_set_input(inp_m);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(pos, "pos_ids");
    ggml_set_input(pos);

    /* Per-token qtype ids. A single slot broadcasts one type embedding; a packed
     * sequence needs each token to pick its own slot's embedding. */
    struct ggml_tensor * qtype_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(qtype_ids, "qtype_ids");
    ggml_set_input(qtype_ids);
    size_t kmax = pk ? (size_t)pk->kmax : (size_t)K;
    size_t nslots = pk ? (size_t)pk->n_slots : 1;
    /* Padding entries in the [kmax, n_slots] logit block get -inf so softmax
     * gives them zero mass. */
    struct ggml_tensor * logit_mask = NULL;
    ggml_backend_buffer_t aux_buf = NULL;
    if (pk) {
        logit_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)kmax, (int64_t)nslots);
        ggml_set_name(logit_mask, "logit_mask");
        ggml_set_input(logit_mask);
    }
    if (qtype_ids || logit_mask) {
        size_t aux_bytes = (size_t)L * sizeof(int32_t)
                         + (logit_mask ? kmax * nslots * sizeof(float) : 0) + 128;
        aux_buf = ggml_backend_alloc_buffer(inp_backend, aux_bytes);
        if (!aux_buf) { fprintf(stderr, "OOM: aux buffer\n"); ggml_free(ctx); return false; }
        qtype_ids->buffer = aux_buf;
        qtype_ids->data = ggml_backend_buffer_get_base(aux_buf);
        if (logit_mask) {
            logit_mask->buffer = aux_buf;
            logit_mask->data = (char *)ggml_backend_buffer_get_base(aux_buf)
                             + (size_t)L * sizeof(int32_t);
        }
    }

    /* 1. Embedding */
    struct ggml_tensor * h = ggml_get_rows(ctx, m->tok_emb, inp);

    /* Attention masks. Two of them when slots are packed:
     *   blk_mask - block diagonal only, for the global layers
     *   sw_mask  - block diagonal AND the sliding window, for the sliding layers
     * A single slot only ever needs the window half (and only past the window).
     * F16, not F32: ggml_flash_attn_ext asserts on the mask type. */
    struct ggml_tensor * sw_mask = NULL;
    struct ggml_tensor * blk_mask = NULL;
    bool need_window = m->local_attention > 0 && L > m->local_attention;
    if (pk) {
        blk_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
        ggml_set_name(blk_mask, "blk_mask");
        ggml_set_input(blk_mask);
        sw_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
        ggml_set_name(sw_mask, "sw_mask");
        ggml_set_input(sw_mask);
    } else if (need_window) {
        sw_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
        ggml_set_name(sw_mask, "sw_mask");
        ggml_set_input(sw_mask);
    }

    /* 2. Embedding norm */
    h = ln_nobias(ctx, h, m->emb_norm_w);

    /* 3. Encoder */
    int max_layers = (getenv("LAYA_LAYERS") ? atoi(getenv("LAYA_LAYERS")) : m->n_layers);
    for (int i = 0; i < max_layers; i++) {
        struct laya_enc_layer * el = &m->enc[i];
        struct ggml_tensor * r = h;
        if (i > 0 && el->attn_norm_w) h = ln_nobias(ctx, h, el->attn_norm_w);
        /* Every global_attn_every_n layers is global; the rest use the window mask.
         * Global layers still need the block mask, otherwise they would attend
         * across slot boundaries. */
        int is_global = (m->global_attn_every_n > 0) && ((i % m->global_attn_every_n) == 0);
        h = mha(ctx, h, el->Wqkv, NULL, el->Wo, NULL, m->n_heads, true, pos,
                m->rope_theta, is_global ? blk_mask : sw_mask);
        h = ggml_add(ctx, h, r);
        r = h;
        h = ln_nobias(ctx, h, el->mlp_norm_w);
        h = geglu_ffn(ctx, h, el->Wi, el->Wo_mlp, m->intermediate);
        h = ggml_add(ctx, h, r);
    }

    /* Debug: if LAYA_DEBUG, compute h and print */
    if (getenv("LAYA_DEBUG")) {
        ggml_set_name(h, "debug_h");
        struct ggml_cgraph * dbg = ggml_new_graph_custom(ctx, MAX_GRAPH_NODES, false);
        ggml_build_forward_expand(dbg, h);
        ggml_backend_sched_t ds = ggml_backend_sched_new(&m->backend, NULL, 1, MAX_GRAPH_NODES, false, true);
        if (ggml_backend_is_cpu(m->backend)) ggml_backend_cpu_set_n_threads(m->backend, 8);
        ggml_backend_sched_graph_compute(ds, dbg);
        struct ggml_tensor * dh = ggml_graph_get_tensor(dbg, "debug_h");
        float dd[5], dd2[5];
        ggml_backend_tensor_get(dh, dd, 0, 5*sizeof(float));
        ggml_backend_tensor_get(dh, dd2, (dh->ne[1]-1)*dh->ne[0]*sizeof(float), 5*sizeof(float));
        fprintf(stderr, "[debug] h ne=[%lld,%lld] after %d layers\n",
                (long long)dh->ne[0], (long long)dh->ne[1], max_layers);
        fprintf(stderr, "[debug] h[0,:5]: %f %f %f %f %f\n", dd[0],dd[1],dd[2],dd[3],dd[4]);
        fprintf(stderr, "[debug] h[-1,:5]: %f %f %f %f %f\n", dd2[0],dd2[1],dd2[2],dd2[3],dd2[4]);
        ggml_backend_sched_free(ds);
        ggml_free(ctx); printf("{\"debug\":true}\n"); return NULL;
    }

    /* 4. Final norm + type embedding.
     * type_emb is [H, n_qtypes] in memory, so a per-token gather picks the right
     * embedding for each position -- needed once slots with different qtypes
     * share one sequence (a single slot just repeats one id L times). */
    h = ln_nobias(ctx, h, m->final_norm_w);
    struct ggml_tensor * te2 = ggml_view_2d(ctx, m->type_emb, H, 3,
                                            H * ggml_element_size(m->type_emb), 0);
    struct ggml_tensor * tv = ggml_get_rows(ctx, te2, qtype_ids);
    h = ggml_add(ctx, h, tv);

    /* 5. Decision head */
    for (int i = 0; i < m->head_layers; i++) {
        struct laya_head_layer * hl = &m->head[i];
        struct ggml_tensor * r = h;
        h = ln(ctx, h, hl->norm1_w, hl->norm1_b);
        h = mha(ctx, h, hl->in_proj_w, hl->in_proj_b, hl->out_proj_w, hl->out_proj_b,
                m->n_heads, false, NULL, 10000.0f, blk_mask);
        h = ggml_add(ctx, h, r);
        r = h;
        h = ln(ctx, h, hl->norm2_w, hl->norm2_b);
        h = gelu_ffn(ctx, h, hl->linear1_w, hl->linear1_b, hl->linear2_w, hl->linear2_b);
        h = ggml_add(ctx, h, r);
    }

    /* 6. Gather marker positions: h[:, marker_pos[k]] for each k → [H, K].
     * This used to be K views plus K-1 concats with the offsets baked into the
     * graph, which tied the graph to one particular set of markers and made it
     * impossible to reuse. ggml_get_rows does the same gather as one op, driven
     * by the input tensor, so the graph only depends on K (not the values). */
    struct ggml_tensor * mk = ggml_get_rows(ctx, h, inp_m);

    /* 7. Scorer */
    struct ggml_tensor * sc = ln(ctx, mk, m->sc_ln_w, m->sc_ln_b);
    sc = ggml_add(ctx, ggml_mul_mat(ctx, m->sc_w1, sc), m->sc_b1);
    sc = ggml_gelu(ctx, sc);
    struct ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, m->sc_w2, sc), m->sc_b2);
    if (pk) {
        /* Markers are laid out slot-major with kmax entries each, so the flat
         * [kmax*n_slots] logit vector reshapes into [kmax, n_slots] and softmax
         * over ne[0] normalises each slot independently. Padding slots are pushed
         * to -inf first so they take none of the mass. */
        /* Mask only, no softmax: temperature is per-qtype and slots in one pack
         * can have different qtypes, so the normalisation has to happen per slot
         * at output time. */
        logits = ggml_reshape_2d(ctx, logits, (int64_t)kmax, (int64_t)nslots);
        logits = ggml_add(ctx, logits, logit_mask);
        logits = ggml_reshape_1d(ctx, logits, (int64_t)(kmax * nslots));
    } else {
        logits = ggml_reshape_1d(ctx, logits, K);
    }
    ggml_set_name(logits, "logits");

    /* Build graph */
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_GRAPH_NODES, false);
    ggml_build_forward_expand(gf, logits);
    fprintf(stderr, "[laya] graph: %d nodes\n", ggml_graph_n_nodes(gf));

    /* LAYA_OPS=1 prints an op histogram. The model is small enough that runtime
     * tracks kernel count far more closely than FLOPs, so this is what tells you
     * whether a fusion is worth doing. */
    if (getenv("LAYA_OPS")) {
        int counts[GGML_OP_COUNT];
        memset(counts, 0, sizeof(counts));
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) counts[ggml_graph_node(gf, i)->op]++;
        fprintf(stderr, "[laya] op histogram:\n");
        for (int o = 0; o < GGML_OP_COUNT; o++) {
            if (counts[o]) fprintf(stderr, "        %-16s %d\n", ggml_op_name((enum ggml_op)o), counts[o]);
        }
    }

    /* Attach the input tensors to their buffer. Token ids and marker positions
     * change every call and are set in run_inference; positions and the window
     * mask depend only on the shape, so fill them once here. */
    inp->buffer = inp_buf;
    inp->data = ggml_backend_buffer_get_base(inp_buf);
    inp_m->buffer = inp_buf;
    inp_m->data = (char *)ggml_backend_buffer_get_base(inp_buf) + L * sizeof(int32_t);
    pos->buffer = inp_buf;
    pos->data = (char *)ggml_backend_buffer_get_base(inp_buf) + (L + K) * sizeof(int32_t);
    {
        /* Positions restart at 0 for each slot: a concatenated slot must not see
         * itself as continuing on from the previous one's positions. */
        int32_t * p = (int32_t *)malloc(L * sizeof(int32_t));
        int * qt = (int32_t *)malloc(L * sizeof(int32_t));
        int at = 0;
        for (int s = 0; s < (pk ? pk->n_slots : 1); s++) {
            int len = pk ? pk->seg_len[s] : L;
            int qt_s = pk ? pk->seg_qtype[s] : qtype;
            for (int i = 0; i < len; i++) { p[at] = i; qt[at] = qt_s; at++; }
        }
        ggml_backend_tensor_set(pos, p, 0, L * sizeof(int32_t));
        ggml_backend_tensor_set(qtype_ids, qt, 0, L * sizeof(int32_t));
        free(p);
        free(qt);
    }

    if (logit_mask) {
        float * lm = (float *)malloc(kmax * nslots * sizeof(float));
        for (size_t s = 0; s < nslots; s++)
            for (size_t i = 0; i < kmax; i++)
                lm[s * kmax + i] = (i < (size_t)pk->seg_k[s]) ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(logit_mask, lm, 0, kmax * nslots * sizeof(float));
        free(lm);
    }

    /* Sliding-window / block-diagonal masks: 0 where attention is allowed,
     * -inf where it is not. */
    ggml_backend_buffer_t mask_buf = NULL;
    if (sw_mask || blk_mask) {
        size_t cells = (size_t)L * L;
        mask_buf = ggml_backend_alloc_buffer(inp_backend,
                                             cells * sizeof(ggml_fp16_t)
                                             * (blk_mask ? 2 : 1) + 128);
        if (!mask_buf) { fprintf(stderr, "OOM: mask buffer\n"); ggml_free(ctx); return false; }

        /* Segment id per token, so we can tell same-slot pairs apart. */
        int * seg = (int *)malloc(L * sizeof(int));
        if (pk) {
            int at = 0;
            for (int s = 0; s < pk->n_slots; s++)
                for (int i = 0; i < pk->seg_len[s]; i++) seg[at++] = s;
        }
        ggml_fp16_t * mb = (ggml_fp16_t *)malloc(cells * sizeof(ggml_fp16_t));

        if (blk_mask) {
            for (int q = 0; q < L; q++)
                for (int k = 0; k < L; k++)
                    mb[q * L + k] = ggml_fp32_to_fp16(seg[q] == seg[k] ? 0.0f : -INFINITY);
            blk_mask->buffer = mask_buf;
            blk_mask->data = ggml_backend_buffer_get_base(mask_buf);
            ggml_backend_tensor_set(blk_mask, mb, 0, cells * sizeof(ggml_fp16_t));
        }

        if (sw_mask) {
            int win = m->local_attention;
            size_t off = blk_mask ? cells * sizeof(ggml_fp16_t) : 0;
            for (int q = 0; q < L; q++)
                for (int k = 0; k < L; k++) {
                    bool ok = (!pk || seg[q] == seg[k]) && (!win || abs(q - k) < win);
                    mb[q * L + k] = ggml_fp32_to_fp16(ok ? 0.0f : -INFINITY);
                }
            sw_mask->buffer = mask_buf;
            sw_mask->data = (char *)ggml_backend_buffer_get_base(mask_buf) + off;
            ggml_backend_tensor_set(sw_mask, mb, 0, cells * sizeof(ggml_fp16_t));
        }
        free(mb);
        free(seg);
    }

    /* Backend scheduler: allocates intermediates, dispatches ops to the right
     * backend (GPU for matmul when weights live there, CPU for anything without a
     * GPU kernel) and manages cross-backend copies. */
    ggml_backend_t backends[2];
    int n_backends = 0;
    backends[n_backends++] = m->backend;
    if (!ggml_backend_is_cpu(m->backend)) {
        backends[n_backends++] = ggml_backend_cpu_init();  /* fallback */
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, n_backends, MAX_GRAPH_NODES, false, true);
    if (!sched) {
        fprintf(stderr, "[laya] sched creation failed\n");
        runner_free(m);
        ggml_free(ctx);
        return false;
    }

    int n_threads = 8;
    if (getenv("LAYA_THREADS")) n_threads = atoi(getenv("LAYA_THREADS"));
    for (int bi = 0; bi < n_backends; bi++) {
        if (ggml_backend_is_cpu(backends[bi])) {
            ggml_backend_cpu_set_n_threads(backends[bi], n_threads);
        }
    }

    /* Publish to the cache; run_inference takes it from here. */
    m->r.ctx = ctx;
    m->r.sched = sched;
    m->r.gf = gf;
    m->r.inp_backend = inp_backend;
    m->r.inp_buf = inp_buf;
    m->r.aux_buf = aux_buf;
    m->r.mask_buf = mask_buf;
    m->r.inp = inp;
    m->r.inp_m = inp_m;
    m->r.pos = pos;
    m->r.sw_mask = sw_mask;
    m->r.blk_mask = blk_mask;
    m->r.backends[0] = backends[0];
    m->r.backends[1] = (n_backends > 1) ? backends[1] : NULL;
    m->r.n_backends = n_backends;
    m->r.L = L;
    m->r.K = K;
    m->r.packed = pk != NULL;
    if (pk) {
        m->r.n_slots = pk->n_slots;
        m->r.kmax = pk->kmax;
        memcpy(m->r.seg_k, pk->seg_k, sizeof(int) * pk->n_slots);
    } else {
        m->r.n_slots = 1;
        m->r.kmax = K;
        m->r.seg_k[0] = K;
    }
    m->r.qtype = qtype;
    m->r.ready = true;
    return true;
}

/* One forward pass. Reuses the cached graph when the shape is unchanged, so
 * steady-state calls skip both graph building and CUDA graph capture. */
static float * run_inference(struct laya_model * m, int32_t * ids, int n_tok,
                             int32_t * marker_pos, int n_markers, int qtype,
                             const packed_t * pk) {
    int L = n_tok, K = n_markers;

    /* A packed request is keyed by its full segment layout, not just (L, K):
     * two packs with the same totals but different per-slot shapes need
     * different masks. */
    bool same = m->r.ready && m->r.L == L && m->r.K == K && m->r.qtype == qtype
                && m->r.packed == (pk != NULL);
    if (same && pk) {
        if (m->r.n_slots != pk->n_slots || m->r.kmax != pk->kmax) same = false;
        else same = memcmp(m->r.seg_k, pk->seg_k, sizeof(int) * pk->n_slots) == 0;
    }
    if (!same) {
        runner_free(m);
        if (!runner_build(m, L, K, qtype, pk)) return NULL;
    }

    /* Only the token ids and marker positions change between calls. */
    ggml_backend_tensor_set(m->r.inp, ids, 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(m->r.inp_m, marker_pos, 0, K * sizeof(int32_t));

    double t0 = now_ms();
    enum ggml_status status = ggml_backend_sched_graph_compute(m->r.sched, m->r.gf);
    double compute_ms = now_ms() - t0;
    if (getenv("LAYA_TIMING")) {
        fprintf(stderr, "[laya] compute: %.2f ms  (%s, graph %d nodes)\n",
                compute_ms, ggml_backend_is_cpu(m->backend) ? "cpu" : "gpu",
                ggml_graph_n_nodes(m->r.gf));
    }
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[laya] compute failed: %d\n", status);
        return NULL;
    }

    struct ggml_tensor * result = ggml_graph_get_tensor(m->r.gf, "logits");
    if (!result) { fprintf(stderr, "[laya] logits tensor not found\n"); return NULL; }

    float * out = (float *)malloc(K * sizeof(float));
    ggml_backend_tensor_get(result, out, 0, K * sizeof(float));
    return out;
}

/* ---------------------------------------------------------------- JSON I/O */

static int parse_int_array(const char * json, const char * key, int32_t * out, int max_n) {
    char pat[64];
    /* Try with and without space after colon */
    snprintf(pat, sizeof(pat), "\"%s\":[", key);
    const char * p = strstr(json, pat);
    if (!p) {
        snprintf(pat, sizeof(pat), "\"%s\": [", key);
        p = strstr(json, pat);
    }
    if (!p) return 0;
    p = strchr(p, '[') + 1;
    int n = 0;
    while (*p && *p != ']' && n < max_n) {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') break;
        out[n++] = (int32_t)strtol(p, (char **)&p, 10);
    }
    return n;
}

static int parse_int(const char * json, const char * key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char * p = strstr(json, pat);
    if (!p) return 0;
    p = strchr(p, ':') + 1;
    while (*p == ' ') p++;
    return (int)strtol(p, NULL, 10);
}

/* ---------------------------------------------------------------- output */

/* One result object, no trailing newline -- batch mode supplies the commas. */
static void print_result(struct laya_model * m, float * raw, int K, int qtype) {
    float temp = (qtype < 3) ? m->temperatures[qtype] : 1.0f;
    float probs[MAX_MARKERS], max_l = -1e9f;
    for (int i = 0; i < K; i++) {
        probs[i] = raw[i] / temp;
        if (probs[i] > max_l) max_l = probs[i];
    }
    float sum = 0;
    for (int i = 0; i < K; i++) { probs[i] = expf(probs[i] - max_l); sum += probs[i]; }
    for (int i = 0; i < K; i++) probs[i] /= sum;

    int choice = 0; float pmax = 0;
    for (int i = 0; i < K; i++) if (probs[i] > pmax) { pmax = probs[i]; choice = i; }

    float ent = 0;
    for (int i = 0; i < K; i++) if (probs[i] > 1e-9f) ent -= probs[i] * logf(probs[i]);
    float maxent = logf((float)K);
    float conf = maxent > 0 ? 1.0f - ent / maxent : 1.0f;

    printf("{\"logits\":[");
    for (int i = 0; i < K; i++) printf("%s%.4f", i ? "," : "", raw[i]);
    printf("],\"probs\":[");
    for (int i = 0; i < K; i++) printf("%s%.4f", i ? "," : "", probs[i]);
    printf("],\"choice\":%d,\"confidence\":%.4f}", choice, conf);
}

/* Parse a packed request. Returns false if the line has no "packed" object. */
static bool parse_packed(const char * json, packed_t * pk) {
    const char * p = strstr(json, "\"packed\"");
    if (!p) return false;
    pk->kmax = parse_int(json, "kmax");
    pk->n_slots = parse_int_array(json, "seg_len", pk->seg_len, MAX_BATCH);
    if (pk->n_slots <= 0 || pk->kmax <= 0) {
        fprintf(stderr, "packed needs seg_len and kmax\n");
        return false;
    }
    if (parse_int_array(json, "seg_qtype", pk->seg_qtype, MAX_BATCH) != pk->n_slots ||
        parse_int_array(json, "seg_k", pk->seg_k, MAX_BATCH) != pk->n_slots) {
        fprintf(stderr, "seg_qtype / seg_k must match seg_len length\n");
        return false;
    }
    return true;
}

/* One packed forward pass, split back into per-slot results.
 * Output shape matches a batched {"items":[...]} response. */
static bool handle_packed(struct laya_model * m, const char * json, const packed_t * pk,
                          bool quiet) {
    int32_t ids[MAX_TOKENS * 4];
    int32_t markers[MAX_PACKED_MARKERS];
    int n_tok = parse_int_array(json, "input_ids", ids, MAX_TOKENS * 4);
    int n_markers = parse_int_array(json, "marker_pos", markers, MAX_PACKED_MARKERS);

    int L_expected = 0, K_expected = 0;
    for (int s = 0; s < pk->n_slots; s++) { L_expected += pk->seg_len[s]; K_expected += pk->kmax; }
    if (n_tok != L_expected || n_markers != K_expected) {
        fprintf(stderr, "packed shape mismatch: got %d tokens/%d markers, "
                        "segments say %d/%d\n", n_tok, n_markers, L_expected, K_expected);
        return false;
    }
    if (!quiet) {
        fprintf(stderr, "[laya] packed: %d slots, %d tokens, kmax=%d\n",
                pk->n_slots, n_tok, pk->kmax);
    }

    float * raw = run_inference(m, ids, n_tok, markers, n_markers, 0, pk);
    if (!raw) return false;

    printf("{\"results\":[");
    for (int s = 0; s < pk->n_slots; s++) {
        if (s) printf(",");
        /* Slot s owns markers [s*kmax, s*kmax + seg_k[s]) */
        print_result(m, raw + s * pk->kmax, pk->seg_k[s], pk->seg_qtype[s]);
    }
    printf("]}");
    free(raw);
    return true;
}

/* Run a single request and print its result object. */
static bool handle_one(struct laya_model * m, const char * json, bool quiet) {
    int32_t ids[MAX_TOKENS], markers[MAX_MARKERS];
    int n_tok = parse_int_array(json, "input_ids", ids, MAX_TOKENS);
    int n_markers = parse_int_array(json, "marker_pos", markers, MAX_MARKERS);
    int qtype = parse_int(json, "qtype");
    if (n_tok == 0 || n_markers == 0) {
        fprintf(stderr, "need input_ids and marker_pos\n");
        return false;
    }
    if (!quiet) fprintf(stderr, "[laya] %d tokens, %d markers, qtype=%d\n", n_tok, n_markers, qtype);

    /* LAYA_REPEAT=N re-runs the same input N times after a single load; this is
     * how you measure steady-state throughput rather than cold-start latency. */
    int repeat = 1;
    if (getenv("LAYA_REPEAT")) repeat = atoi(getenv("LAYA_REPEAT"));
    if (repeat < 1) repeat = 1;

    float * raw = NULL;
    double t_first = 0, t_sum = 0, t_min = 1e18, t_max = 0;
    for (int rep = 0; rep < repeat; rep++) {
        double t0 = now_ms();
        float * r = run_inference(m, ids, n_tok, markers, n_markers, qtype, NULL);
        double dt = now_ms() - t0;
        if (!r) return false;
        if (rep == 0) t_first = dt;
        t_sum += dt;
        if (dt < t_min) t_min = dt;
        if (dt > t_max) t_max = dt;
        if (raw) free(raw);
        raw = r;
    }
    if (repeat > 1) {
        fprintf(stderr,
                "[laya] repeat=%d  first=%.2f ms  mean=%.2f ms  min=%.2f  max=%.2f  "
                "(excludes model load)\n", repeat, t_first, t_sum / repeat, t_min, t_max);
    }

    print_result(m, raw, n_markers, qtype);
    free(raw);
    return true;
}

/* Split {"items":[{...},{...}]} into individual object strings.
 * Returns the item count, or -1 when the line is not a batch request. */
static int split_items(const char * json, char ** out, int max_items) {
    const char * p = strstr(json, "\"items\"");
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    p++;
    int n = 0;
    while (*p && n < max_items) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p == ']' || *p == '\0' || *p != '{') break;
        const char * start = p;
        int depth = 0;
        while (*p) {
            if (*p == '"') {           /* skip strings so braces inside don't count */
                p++;
                while (*p && *p != '"') { if (*p == '\\') p++; p++; }
            } else if (*p == '{') depth++;
            else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
            p++;
        }
        size_t len = (size_t)(p - start);
        out[n] = (char *)malloc(len + 1);
        memcpy(out[n], start, len);
        out[n][len] = '\0';
        n++;
    }
    return n;
}

/* One input line: either a batch ({"items":[...]}) or a single request.
 * Exactly one line of output either way, so daemon mode stays line-synchronised. */
static bool handle_line(struct laya_model * m, char * line, bool quiet) {
    packed_t pk;
    if (parse_packed(line, &pk)) return handle_packed(m, line, &pk, quiet);

    char * items[MAX_BATCH];
    int n = split_items(line, items, MAX_BATCH);
    if (n < 0) return handle_one(m, line, quiet);
    if (n == 0) { fprintf(stderr, "empty items array\n"); return false; }

    printf("{\"results\":[");
    for (int i = 0; i < n; i++) {
        if (i) printf(",");
        bool ok = handle_one(m, items[i], quiet);
        free(items[i]);
        if (!ok) { printf("]}"); return false; }
    }
    printf("]}");
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> < input.json\n", argv[0]);
        return 1;
    }

    struct laya_model * m = load_model(argv[1]);
    if (!m) return 1;

    char * line = (char *)malloc(MAX_LINE);
    if (!line) { fprintf(stderr, "OOM: line buffer\n"); return 1; }

    bool daemon = getenv("LAYA_DAEMON") != NULL;

    if (!daemon) {
        size_t n = fread(line, 1, MAX_LINE - 1, stdin);
        line[n] = '\0';
        bool ok = handle_line(m, line, false);
        printf("\n");
        free(line);
        return ok ? 0 : 1;
    }

    /* Daemon: one request per input line, one result per output line, flushed
     * immediately. The model stays loaded and the cached graph stays warm, so
     * neither the 630 ms load nor the CUDA graph capture repeats per call. */
    fprintf(stderr, "[laya] daemon ready\n");
    while (fgets(line, MAX_LINE, stdin)) {
        bool ok = handle_line(m, line, true);
        printf("\n");
        fflush(stdout);
        if (!ok) fprintf(stderr, "[laya] request failed\n");
    }
    free(line);
    runner_free(m);
    return 0;
}
