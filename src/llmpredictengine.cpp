#include "llmpredictengine.h"

#include <fcitx-config/iniparser.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/standardpath.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/userinterfacemanager.h>

#include <string>

FCITX_DEFINE_LOG_CATEGORY(llmPredict, "llm_predict")
#define FCITX_LLM_DEBUG() FCITX_LOGC(llmPredict, Debug)
#define FCITX_LLM_INFO()  FCITX_LOGC(llmPredict, Info)
#define FCITX_LLM_WARN()  FCITX_LOGC(llmPredict, Warn)

// ---------------------------------------------------------------------------
// CandidateWord: represents one LLM prediction in the candidate panel.
// Selecting it commits the text piece to the active application.
// ---------------------------------------------------------------------------
class LLMCandidateWord : public fcitx::CandidateWord {
public:
    explicit LLMCandidateWord(std::string text)
        : fcitx::CandidateWord(fcitx::Text(text))
        , text_(std::move(text))
    {}

    void select(fcitx::InputContext *ic) const override {
        ic->commitString(text_);
    }

private:
    std::string text_;
};

// ---------------------------------------------------------------------------
// LLMPredictEngine
// ---------------------------------------------------------------------------
LLMPredictEngine::LLMPredictEngine(fcitx::Instance *instance)
    : instance_(instance)
{
    // Load configuration from
    //   ~/.config/fcitx5/conf/fcitx5-llm-predict.conf  (user, PkgConfig)
    // or the system-wide equivalent if the user file is absent.
    fcitx::readAsIni(config_,
                     fcitx::StandardPath::Type::PkgConfig,
                     "conf/fcitx5-llm-predict.conf");

    reloadPredictor();
}

LLMPredictEngine::~LLMPredictEngine() = default;

// ---------------------------------------------------------------------------

void LLMPredictEngine::reloadPredictor() {
    const std::string &modelPath = *config_.modelPath;

    if (modelPath.empty()) {
        FCITX_LLM_WARN() << "LLM model path is not configured. "
                            "Set ModelPath in fcitx5-llm-predict.conf.";
        predictor_.reset();
        return;
    }

    FCITX_LLM_INFO() << "Loading LLM model: " << modelPath
                     << "  gpu_layers=" << *config_.nGpuLayers
                     << "  n_ctx="      << *config_.nCtx
                     << "  threads="    << *config_.nThreads;

    LLMPredictor::Config cfg;
    cfg.modelPath  = modelPath;
    cfg.nGpuLayers = *config_.nGpuLayers;
    cfg.nCtx       = *config_.nCtx;
    cfg.nThreads   = *config_.nThreads;

    predictor_ = std::make_unique<LLMPredictor>(cfg);

    if (predictor_->isLoaded()) {
        FCITX_LLM_INFO() << "LLM model loaded successfully.";
    } else {
        FCITX_LLM_WARN() << "Failed to load LLM model from: " << modelPath;
        predictor_.reset();
    }
}

// ---------------------------------------------------------------------------

void LLMPredictEngine::activate(const fcitx::InputMethodEntry &,
                                fcitx::InputContextEvent &event)
{
    auto *ic = event.inputContext();
    // Request surrounding text support from the client application so that
    // we can read everything already typed in the input box.
    ic->setCapabilityFlags(
        ic->capabilityFlags() | fcitx::CapabilityFlag::SurroundingText);
}

void LLMPredictEngine::deactivate(const fcitx::InputMethodEntry &entry,
                                   fcitx::InputContextEvent &event)
{
    reset(entry, event);
}

void LLMPredictEngine::reset(const fcitx::InputMethodEntry &,
                              fcitx::InputContextEvent &event)
{
    auto *ic = event.inputContext();
    ic->inputPanel().reset();
    ic->updatePreedit();
    ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

// ---------------------------------------------------------------------------

void LLMPredictEngine::keyEvent(const fcitx::InputMethodEntry &,
                                 fcitx::KeyEvent &keyEvent)
{
    // Ignore key-release events; we only act on key-press.
    if (keyEvent.isRelease()) {
        return;
    }

    auto *ic  = keyEvent.inputContext();
    auto  key = keyEvent.key();

    // ── Digit keys 1–9 select the corresponding candidate ─────────────────
    if (!key.hasModifier()) {
        if (key.check(FcitxKey_1) || key.check(FcitxKey_2) ||
            key.check(FcitxKey_3) || key.check(FcitxKey_4) ||
            key.check(FcitxKey_5) || key.check(FcitxKey_6) ||
            key.check(FcitxKey_7) || key.check(FcitxKey_8) ||
            key.check(FcitxKey_9))
        {
            auto *panel = &ic->inputPanel();
            auto clist = panel->candidateList();
            if (clist) {
                int idx = static_cast<int>(key.sym() - FcitxKey_1);
                if (idx < clist->size()) {
                    clist->candidate(idx).select(ic);
                    // Clear candidates after selection.
                    ic->inputPanel().reset();
                    ic->updateUserInterface(
                        fcitx::UserInterfaceComponent::InputPanel);
                    keyEvent.filterAndAccept();
                    return;
                }
            }
        }
    }

    // ── Printable ASCII characters are committed directly ─────────────────
    // This keeps the plugin transparent: the user types normally and the
    // surrounding text accumulates, which we use as LLM context.
    if (key.isSimple() && !key.hasModifier()) {
        const uint32_t sym = key.sym();
        if (sym >= 0x20 && sym < 0x7f) {
            char ch[2] = {static_cast<char>(sym), '\0'};
            ic->commitString(ch);
            keyEvent.filterAndAccept();
            updateCandidates(ic);
            return;
        }
    }

    // All other keys (arrows, Backspace, Return, …) pass through unmodified.
    // After navigation or deletion, refresh predictions.
    if (key.check(FcitxKey_BackSpace) ||
        key.check(FcitxKey_Delete)    ||
        key.check(FcitxKey_Return)    ||
        key.check(FcitxKey_KP_Enter)  ||
        key.check(FcitxKey_space))
    {
        // Let the key reach the application, then update predictions.
        // We do not filterAndAccept() so Fcitx5 forwards the key normally.
        updateCandidates(ic);
    }
}

// ---------------------------------------------------------------------------

void LLMPredictEngine::updateCandidates(fcitx::InputContext *ic) {
    if (!predictor_) {
        return;
    }

    // ── Build the context string from surrounding text ─────────────────────
    std::string context;

    const auto &surrounding = ic->surroundingText();
    if (surrounding.isValid()) {
        const std::string &full   = surrounding.text();
        const uint32_t     cursor = surrounding.cursor();

        // cursor == full.size() means the cursor is at the very end of the
        // text; substr(0, full.size()) gives us the entire string, which is
        // correct.  cursor > full.size() is invalid; fall back to full text.
        if (cursor <= full.size()) {
            context = full.substr(0, cursor);
        } else {
            context = full;
        }
    }

    if (context.empty()) {
        // Nothing to predict from – clear the candidate panel.
        ic->inputPanel().reset();
        ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        return;
    }

    // How many trailing bytes to show in the debug log.
    static constexpr size_t kDebugPreviewLen = 40;

    FCITX_LLM_DEBUG() << "Predicting next token for context ("
                      << context.size() << " bytes): \""
                      << context.substr(context.size() > kDebugPreviewLen
                                            ? context.size() - kDebugPreviewLen
                                            : 0)
                      << "\"";

    // ── Run LLM inference ─────────────────────────────────────────────────
    auto candidates = predictor_->predict(context, *config_.topK);

    if (candidates.empty()) {
        ic->inputPanel().reset();
        ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        return;
    }

    // ── Populate the Fcitx5 candidate panel ───────────────────────────────
    auto clist = std::make_unique<fcitx::CommonCandidateList>();
    clist->setPageSize(*config_.topK);

    for (const auto &cand : candidates) {
        clist->append<LLMCandidateWord>(cand.text);
    }

    // Bind digit keys 1–9 as selection shortcuts.
    fcitx::KeyList selKeys;
    for (auto k : {FcitxKey_1, FcitxKey_2, FcitxKey_3, FcitxKey_4,
                   FcitxKey_5, FcitxKey_6, FcitxKey_7, FcitxKey_8,
                   FcitxKey_9})
    {
        selKeys.emplace_back(k);
    }
    clist->setSelectionKey(selKeys);

    ic->inputPanel().setCandidateList(std::move(clist));
    ic->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

// ---------------------------------------------------------------------------
// Fcitx5 plugin entry point.
// ---------------------------------------------------------------------------
FCITX_ADDON_FACTORY(LLMPredictEngineFactory)
