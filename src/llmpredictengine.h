#pragma once

#include "config.h"
#include "llmpredictor.h"

#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/instance.h>

#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// The Fcitx5 InputMethodEngine that wraps the LLMPredictor.
//
// Behaviour
// ─────────
// • Printable characters typed by the user are committed immediately to the
//   application (direct / "raw" pass-through).
// • After every commit the engine reads the surrounding text (i.e. all text
//   already present in the input box up to the cursor) and passes it to the
//   LLM as context.
// • The topK most-probable next tokens are shown in the Fcitx5 candidate
//   panel, sorted by descending probability.
// • Pressing a digit key (1–9) selects the corresponding candidate.
// • Backspace, Return, and Escape are forwarded to the application unchanged.
// ---------------------------------------------------------------------------
class LLMPredictEngine : public fcitx::InputMethodEngine {
public:
    explicit LLMPredictEngine(fcitx::Instance *instance);
    ~LLMPredictEngine() override;

    // ── InputMethodEngine interface ──────────────────────────────────────
    void keyEvent(const fcitx::InputMethodEntry &entry,
                  fcitx::KeyEvent &keyEvent) override;

    void activate(const fcitx::InputMethodEntry &entry,
                  fcitx::InputContextEvent &event) override;

    void deactivate(const fcitx::InputMethodEntry &entry,
                    fcitx::InputContextEvent &event) override;

    void reset(const fcitx::InputMethodEntry &entry,
               fcitx::InputContextEvent &event) override;

private:
    /// Rebuild the candidate panel from fresh LLM predictions.
    void updateCandidates(fcitx::InputContext *ic);

    /// Load (or reload) the predictor from the current config.
    void reloadPredictor();

    fcitx::Instance            *instance_;
    LLMPredictConfig            config_;
    std::unique_ptr<LLMPredictor> predictor_;
};

// ---------------------------------------------------------------------------
// Addon factory – instantiated by Fcitx5 when the plugin library is loaded.
// ---------------------------------------------------------------------------
class LLMPredictEngineFactory : public fcitx::AddonFactory {
public:
    fcitx::AddonInstance *create(fcitx::AddonManager *manager) override {
        return new LLMPredictEngine(manager->instance());
    }
};
