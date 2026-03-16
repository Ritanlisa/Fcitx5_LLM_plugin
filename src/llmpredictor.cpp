#include "llmpredictor.h"

// llama.cpp public C header (available after installing llama.cpp).
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Internal implementation details – hidden from the header.
// ---------------------------------------------------------------------------
struct LLMPredictor::Impl {
    llama_model   *model = nullptr;
    llama_context *ctx   = nullptr;
    Config         cfg;
    bool           loaded = false;

    ~Impl() {
        if (ctx)   { llama_free(ctx);        ctx   = nullptr; }
        if (model) { llama_free_model(model); model = nullptr; }
    }
};

// ---------------------------------------------------------------------------
// Constructor – loads the GGUF model and creates an inference context.
// ---------------------------------------------------------------------------
LLMPredictor::LLMPredictor(const Config &config)
    : impl_(std::make_unique<Impl>())
{
    impl_->cfg = config;

    if (config.modelPath.empty()) {
        return; // not configured yet
    }

    // One-time backend initialisation (idempotent after first call).
    llama_backend_init();

    // ── Load model ────────────────────────────────────────────────────────
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.nGpuLayers; // -1 = all layers on GPU

    impl_->model = llama_load_model_from_file(config.modelPath.c_str(), mparams);
    if (!impl_->model) {
        return; // error already printed by llama.cpp
    }

    // ── Create inference context ──────────────────────────────────────────
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = static_cast<uint32_t>(config.nCtx);
    cparams.n_batch    = static_cast<uint32_t>(config.nCtx);
    cparams.n_threads  = static_cast<uint32_t>(config.nThreads);
    cparams.seed       = static_cast<uint32_t>(
                             config.seed >= 0 ? config.seed : LLAMA_DEFAULT_SEED);

    impl_->ctx = llama_new_context_with_params(impl_->model, cparams);
    if (!impl_->ctx) {
        llama_free_model(impl_->model);
        impl_->model = nullptr;
        return;
    }

    impl_->loaded = true;
}

LLMPredictor::~LLMPredictor() = default;

bool LLMPredictor::isLoaded() const noexcept {
    return impl_ && impl_->loaded;
}

// ---------------------------------------------------------------------------
// predict() – run a single forward pass and return the topK next tokens
//             sorted by descending softmax probability.
// ---------------------------------------------------------------------------
std::vector<TokenCandidate> LLMPredictor::predict(const std::string &context,
                                                   int topK)
{
    if (!isLoaded() || context.empty() || topK <= 0) {
        return {};
    }

    auto *model = impl_->model;
    auto *ctx   = impl_->ctx;

    // ── Tokenise the context string ───────────────────────────────────────
    // First call with a null buffer: llama_tokenize returns the negative of the
    // number of tokens needed, so we negate it to get the required count.
    int tokenCountNeeded = -llama_tokenize(model,
                                    context.c_str(),
                                    static_cast<int>(context.size()),
                                    nullptr, 0,
                                    /*add_special=*/true,
                                    /*parse_special=*/false);
    if (tokenCountNeeded <= 0) {
        return {};
    }

    std::vector<llama_token> tokens(static_cast<size_t>(tokenCountNeeded));
    int nTokens = llama_tokenize(model,
                                 context.c_str(),
                                 static_cast<int>(context.size()),
                                 tokens.data(),
                                 static_cast<int>(tokens.size()),
                                 /*add_special=*/true,
                                 /*parse_special=*/false);
    if (nTokens < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(nTokens));

    // Truncate to fit within the context window (keep the most-recent tokens).
    // Reserve 1 slot for the token we are about to predict.
    const int nCtx = impl_->cfg.nCtx;
    if (nTokens > nCtx - 1) {
        tokens.erase(tokens.begin(),
                     tokens.begin() + (nTokens - (nCtx - 1)));
        nTokens = static_cast<int>(tokens.size());
    }

    // ── Clear KV-cache and run a fresh decode ─────────────────────────────
    llama_kv_cache_clear(ctx);

    llama_batch batch = llama_batch_init(nTokens, /*embd=*/0, /*n_seq_max=*/1);

    for (int i = 0; i < nTokens; ++i) {
        batch.token[i]     = tokens[static_cast<size_t>(i)];
        batch.pos[i]       = static_cast<llama_pos>(i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        // Only request logits for the very last token – we predict what
        // comes AFTER the full context.
        batch.logits[i]    = (i == nTokens - 1) ? 1 : 0;
    }
    batch.n_tokens = nTokens;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return {};
    }

    // ── Retrieve logits for the last token position ───────────────────────
    // llama_get_logits returns a flat array [n_outputs * n_vocab].
    // Since we set logits=1 only for the last token, n_outputs==1 and
    // llama_get_logits points directly to that token's logits.
    float *logits = llama_get_logits(ctx);
    const int nVocab = llama_n_vocab(model);

    // ── Softmax ───────────────────────────────────────────────────────────
    std::vector<float> probs(static_cast<size_t>(nVocab));

    float maxLogit = *std::max_element(logits, logits + nVocab);
    float sumExp   = 0.0f;
    for (int i = 0; i < nVocab; ++i) {
        probs[static_cast<size_t>(i)] = std::exp(logits[i] - maxLogit);
        sumExp += probs[static_cast<size_t>(i)];
    }
    for (int i = 0; i < nVocab; ++i) {
        probs[static_cast<size_t>(i)] /= sumExp;
    }

    // ── Partial sort by descending probability ────────────────────────────
    std::vector<int> indices(static_cast<size_t>(nVocab));
    std::iota(indices.begin(), indices.end(), 0);

    const int k = std::min(topK, nVocab);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&probs](int a, int b) {
            return probs[static_cast<size_t>(a)] >
                   probs[static_cast<size_t>(b)];
        });

    // ── Decode token IDs to text pieces ──────────────────────────────────
    std::vector<TokenCandidate> candidates;
    candidates.reserve(static_cast<size_t>(k));

    for (int i = 0; i < k; ++i) {
        const int tokenId = indices[static_cast<size_t>(i)];

        // llama_token_to_piece: negative return means buffer too small.
        char buf[256] = {};
        int  len = llama_token_to_piece(model, tokenId,
                                        buf, static_cast<int>(sizeof(buf) - 1),
                                        /*lstrip=*/0,
                                        /*special=*/false);
        if (len <= 0) {
            continue;
        }
        buf[len] = '\0';

        // Skip empty, purely-whitespace, or BOS/EOS/control tokens.
        bool hasVisibleChar = false;
        for (int c = 0; c < len; ++c) {
            unsigned char ch = static_cast<unsigned char>(buf[c]);
            if (ch > 0x20) { // anything above ASCII space
                hasVisibleChar = true;
                break;
            }
        }
        if (!hasVisibleChar) {
            continue;
        }

        candidates.push_back({std::string(buf, static_cast<size_t>(len)),
                               probs[static_cast<size_t>(tokenId)]});
    }

    llama_batch_free(batch);

    return candidates;
}
