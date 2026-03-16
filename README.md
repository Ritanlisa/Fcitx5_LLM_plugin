# Fcitx5 LLM Predict Plugin

An input-method plugin for [Fcitx5](https://fcitx-im.org) that uses a **local BitNet / GGUF
LLM** (via [llama.cpp](https://github.com/ggerganov/llama.cpp)) to predict the next word or
token in real time.  Predictions are shown in the candidate panel sorted by **descending
probability** and are committed to the application when the user presses the corresponding
digit key.

## Features

| Feature | Detail |
|---|---|
| GGUF model support | Any model that llama.cpp can load (BitNet, Llama, Mistral, …) |
| CUDA acceleration | All transformer layers are offloaded to the GPU by default (`n_gpu_layers=-1`) |
| Surrounding-text context | Reads **all text already present in the input box** up to the cursor as LLM context |
| Ranked candidates | Up to 10 (configurable) next-token predictions, sorted by softmax probability ↓ |
| Direct key pass-through | Typed characters are committed immediately – no preedit buffering |

---

## Requirements

| Dependency | Version |
|---|---|
| Ubuntu | 22.04 or later |
| CMake | ≥ 3.19 |
| GCC / Clang | C++17 support |
| Extra CMake Modules (ECM) | ≥ 1.0 (optional) |
| Fcitx5 development headers | `libfcitx5core-dev` + `libfcitx5config-dev` + `libfcitx5utils-dev` (Ubuntu) |
| llama.cpp | Built **with CUDA support** – see below |
| CUDA Toolkit | ≥ 11.7 (for `GGML_CUDA=ON`) |

---

## Install Build Dependencies

Run the following once to install all required system packages before building:

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install -y \
    cmake \
    g++ \
    libfcitx5core-dev \
    libfcitx5config-dev \
    libfcitx5utils-dev \
    extra-cmake-modules   # optional but recommended
```

**Arch Linux**
```bash
sudo pacman -S --needed cmake gcc fcitx5 extra-cmake-modules
```

**Fedora / RHEL**
```bash
sudo dnf install -y cmake gcc-c++ fcitx5-devel extra-cmake-modules
```

---

## Building llama.cpp with CUDA

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build
```

This installs headers and the shared library under `~/.local`.

---

## Building the Plugin

```bash
git clone https://github.com/Ritanlisa/Fcitx5_LLM_plugin
cd Fcitx5_LLM_plugin

cmake -B build \
      -DCMAKE_PREFIX_PATH="$HOME/.local" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build
```

> **Tip:** If llama.cpp was installed to a non-standard prefix you can also pass
> `-DLLAMA_CPP_DIR=<prefix>` directly.

If you install the plugin under `~/.local`, do not use `sudo cmake --install`.
Using `sudo` there can leave root-owned files under `~/.local/share/fcitx5`, and
subsequent user installs will fail with permission errors.

On some Debian/Ubuntu setups, Fcitx5 will discover addon descriptors from
`~/.local/share/fcitx5/addon` but still fail to locate the addon shared library
from a user-local prefix. If that happens, either install the plugin into the
system Fcitx5 directories instead, or expose your local library directories to
the runtime linker before restarting Fcitx5.

Example cleanup for a previous mistaken `sudo` install into `~/.local`:

```bash
sudo chown -R "$USER":"$USER" ~/.local/share/fcitx5 ~/.local/lib/fcitx5 ~/.local/lib/*/fcitx5
```

---

## Configuration

The plugin reads
`~/.config/fcitx5/conf/fcitx5-llm-predict.conf`.  Create it (if it does not exist) and
set at least `ModelPath`:

```ini
[LLMPredictConfig]
# Full path to your BitNet / any other GGUF model file.
ModelPath=/path/to/your/bitnet-model.gguf

# Number of transformer layers to offload to GPU.
# -1 = all layers on GPU (fastest); 0 = CPU-only.
NGpuLayers=-1

# Transformer context window size (in tokens).
NCtx=2048

# CPU threads used for the (non-GPU) parts of inference.
NThreads=4

# How many top candidates to show in the panel.
TopK=10
```

---

## Enabling the Input Method

1. Restart Fcitx5 after installation:
   ```bash
   fcitx5 -r
   ```
2. Open **Fcitx5 Configuration** → **Input Method** tab.
3. Click **+**, search for **"LLM Predict"**, and add it to your IM list.
4. Switch to it with your configured toggle shortcut (usually `Ctrl+Space`).

---

## How It Works

```
User types a character
        │
        ▼
Character committed to app  ←──────────────────────────┐
        │                                               │
        ▼                                               │
Read surrounding text (everything in the input box      │
up to the cursor) via Fcitx5 SurroundingText API        │
        │                                               │
        ▼                                               │
Tokenise context → llama_decode() (CUDA) → softmax      │
        │                                               │
        ▼                                               │
Top-K next tokens sorted by P(token|context) ↓         │
        │                                               │
        ▼                                               │
Show in candidate panel                                 │
        │                                               │
   User presses 1–9 to select  ─────────────────────────┘
   (selected token is committed)
```

---

## License

GNU General Public License v3 – see [LICENSE](LICENSE).
