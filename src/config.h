#pragma once

#include <fcitx-config/configuration.h>
#include <fcitx-config/option.h>
#include <string>

// ---------------------------------------------------------------------------
// Plugin configuration – editable via fcitx5-configtool or by hand at
//   ~/.config/fcitx5/conf/fcitx5-llm-predict.conf
// ---------------------------------------------------------------------------
FCITX_CONFIGURATION(
    LLMPredictConfig,

    // Full path to the BitNet/GGUF model file.
    fcitx::Option<std::string> modelPath{
        this, "ModelPath",
        "Path to the GGUF model file",
        ""};

    // How many transformer layers to offload to the GPU.
    // -1  → offload all layers (recommended when CUDA is available).
    //  0  → CPU-only inference.
    fcitx::Option<int> nGpuLayers{
        this, "NGpuLayers",
        "Number of layers to offload to GPU (-1 = all)",
        -1};

    // Maximum number of tokens kept in the KV-cache context.
    fcitx::Option<int> nCtx{
        this, "NCtx",
        "Context window size (tokens)",
        2048};

    // Number of CPU threads used for inference.
    fcitx::Option<int> nThreads{
        this, "NThreads",
        "CPU threads for inference",
        4};

    // How many top-probability candidates to show in the candidate panel.
    fcitx::Option<int> topK{
        this, "TopK",
        "Number of prediction candidates to display",
        10};
)
