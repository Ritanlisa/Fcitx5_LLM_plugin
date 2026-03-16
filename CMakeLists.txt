cmake_minimum_required(VERSION 3.19)
project(fcitx5-llm-predict VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Standard CMake modules ─────────────────────────────────────────────────────
include(GNUInstallDirs)

# Debian/Ubuntu Fcitx5 looks for addon libraries under the multiarch libdir.
# For a local prefix like ~/.local, GNUInstallDirs defaults to plain "lib",
# which makes the addon descriptor visible but leaves the shared library in a
# directory that Fcitx5 does not scan.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND EXISTS "/etc/debian_version"
   AND CMAKE_LIBRARY_ARCHITECTURE
   AND CMAKE_INSTALL_LIBDIR STREQUAL "lib")
    set(CMAKE_INSTALL_LIBDIR "lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "Object code libraries" FORCE)
endif()

# ── ECM (optional) ────────────────────────────────────────────────────────────
# Extra CMake Modules provide convenience helpers (ECMSetupVersion,
# ECMUninstallTarget, …).  They are not required for a successful build; the
# plugin compiles and installs correctly without them.
find_package(ECM 1.0.0 QUIET NO_MODULE)
if(ECM_FOUND)
    list(APPEND CMAKE_MODULE_PATH ${ECM_MODULE_PATH})
    include(ECMSetupVersion)
    include(ECMUninstallTarget)
endif()

# ── Fcitx5 ─────────────────────────────────────────────────────────────────────
# Fcitx5Utils provides install-dir variables (FCITX_INSTALL_ADDONDIR, etc.)
find_package(Fcitx5Core 5.0 REQUIRED)
find_package(Fcitx5Utils REQUIRED)

# ── llama.cpp ──────────────────────────────────────────────────────────────────
# Try cmake package first (installed via `cmake --install`).
find_package(llama QUIET)

if(NOT llama_FOUND)
    # Fall back to pkg-config (some distro packages expose this).
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LLAMA QUIET llama)
    endif()
endif()

if(NOT llama_FOUND AND NOT LLAMA_FOUND)
    # Last resort: user-supplied install prefix via -DLLAMA_CPP_DIR=...
    set(LLAMA_CPP_DIR "" CACHE PATH
        "Root directory of a llama.cpp installation (contains include/ and lib/)")
    if(LLAMA_CPP_DIR STREQUAL "")
        message(FATAL_ERROR
            "llama.cpp not found.\n"
            "Please build & install llama.cpp with CUDA support and either:\n"
            "  • set CMAKE_PREFIX_PATH to the install prefix, or\n"
            "  • pass -DLLAMA_CPP_DIR=<install-prefix> to cmake.\n"
            "Example:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON \\\n"
            "        -DCMAKE_INSTALL_PREFIX=~/.local\n"
            "  cmake --build llama.cpp/build -j$(nproc)\n"
            "  cmake --install llama.cpp/build\n"
            "  cmake -B build -S . -DCMAKE_PREFIX_PATH=~/.local")
    endif()
    find_library(LLAMA_LIBRARY NAMES llama
        PATHS "${LLAMA_CPP_DIR}/lib" "${LLAMA_CPP_DIR}/lib64" REQUIRED)
    set(LLAMA_INCLUDE_DIRS "${LLAMA_CPP_DIR}/include")
    message(STATUS "Found llama.cpp (manual): lib=${LLAMA_LIBRARY} inc=${LLAMA_INCLUDE_DIRS}")
endif()

# ── Plugin shared library ──────────────────────────────────────────────────────
add_library(fcitx5-llm-predict SHARED
    src/llmpredictor.cpp
    src/llmpredictengine.cpp
)

# Ensure runtime dependencies can be resolved when Fcitx5 is launched without
# inheriting user shell environment variables.
set_target_properties(fcitx5-llm-predict PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Keep the default "lib" prefix so the installed file is libfcitx5-llm-predict.so,
# consistent with all other Fcitx5 addons shipped by the distro.

target_include_directories(fcitx5-llm-predict PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(fcitx5-llm-predict PRIVATE
    Fcitx5::Core
    Fcitx5::Config
)

# Link llama.cpp via whichever method succeeded.
if(TARGET llama)
    target_link_libraries(fcitx5-llm-predict PRIVATE llama)
elseif(LLAMA_FOUND)
    target_include_directories(fcitx5-llm-predict PRIVATE ${LLAMA_INCLUDE_DIRS})
    target_link_libraries(fcitx5-llm-predict PRIVATE ${LLAMA_LIBRARIES})
else()
    target_include_directories(fcitx5-llm-predict PRIVATE ${LLAMA_INCLUDE_DIRS})
    target_link_libraries(fcitx5-llm-predict PRIVATE ${LLAMA_LIBRARY})
endif()

# When a custom llama runtime prefix is provided, embed its lib dirs into
# RUNPATH so /usr/lib/.../fcitx5/libfcitx5-llm-predict.so can find libllama.so.
if(LLAMA_CPP_DIR)
    set(_llama_rpath_candidates
        "${LLAMA_CPP_DIR}/lib"
        "${LLAMA_CPP_DIR}/lib64"
    )
    set(_llama_runtime_rpaths "")
    foreach(_dir IN LISTS _llama_rpath_candidates)
        if(EXISTS "${_dir}")
            list(APPEND _llama_runtime_rpaths "${_dir}")
        endif()
    endforeach()
    if(_llama_runtime_rpaths)
        set_target_properties(fcitx5-llm-predict PROPERTIES
            BUILD_RPATH "${_llama_runtime_rpaths}"
            INSTALL_RPATH "${_llama_runtime_rpaths}"
        )
    endif()
endif()

# ── Installation ───────────────────────────────────────────────────────────────
# Plugin .so → ${FCITX_INSTALL_ADDONDIR}  (e.g. /usr/lib/<arch>/fcitx5/)
install(TARGETS fcitx5-llm-predict
    LIBRARY DESTINATION "${FCITX_INSTALL_ADDONDIR}"
)

# Addon descriptor → ${FCITX_INSTALL_PKGDATADIR}/addon/
# Installed as "fcitx5-llm-predict.conf"; Fcitx5 derives addon ID from stem.
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/fcitx5-llm-predict-addon.conf"
    DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon"
    RENAME "fcitx5-llm-predict.conf"
)

# Input method descriptor → ${FCITX_INSTALL_PKGDATADIR}/inputmethod/
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/llm-predict.conf"
    DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod"
)
