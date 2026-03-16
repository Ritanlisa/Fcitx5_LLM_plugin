#pragma once

#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// A single next-token prediction from the LLM.
// ---------------------------------------------------------------------------
struct TokenCandidate {
    std::string text;        ///< The decoded text piece for this token.
    float       probability; ///< Softmax probability in [0, 1].
};

// ---------------------------------------------------------------------------
// Thin wrapper around a llama.cpp model + context.
//
// Usage:
//   LLMPredictor::Config cfg;
//   cfg.modelPath   = "/path/to/model.gguf";
//   cfg.nGpuLayers  = -1;   // all layers on GPU
//   LLMPredictor pred(cfg);
//   if (pred.isLoaded()) {
//       auto cands = pred.predict("Hello, my name is", 10);
//       // cands[0] is the most probable next token, cands[1] second, …
//   }
// ---------------------------------------------------------------------------
class LLMPredictor {
public:
    struct Config {
        std::string modelPath;
        int nGpuLayers = -1; ///< -1 = offload all layers to GPU
        int nCtx       = 2048;
        int nThreads   = 4;
        int seed       = -1; ///< -1 = random seed
    };

    explicit LLMPredictor(const Config &config);
    ~LLMPredictor();

    // Non-copyable, movable.
    LLMPredictor(const LLMPredictor &)            = delete;
    LLMPredictor &operator=(const LLMPredictor &) = delete;
    LLMPredictor(LLMPredictor &&)                 = default;
    LLMPredictor &operator=(LLMPredictor &&)      = default;

    /// Returns true if the model was loaded successfully.
    bool isLoaded() const noexcept;

    /// Predict the next topK tokens after \p context, sorted by descending
    /// probability.  Returns an empty vector on error or if not loaded.
    std::vector<TokenCandidate> predict(const std::string &context,
                                        int topK = 10);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
