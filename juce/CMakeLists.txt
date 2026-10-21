cmake_minimum_required(VERSION 3.22)
project(AbletonVocoderJuce VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(JUCE_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
set(JUCE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(JUCE_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/JUCE")
if(EXISTS "${JUCE_LOCAL}/CMakeLists.txt")
    add_subdirectory("${JUCE_LOCAL}" "${CMAKE_BINARY_DIR}/JUCE" EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.4
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(JUCE)
endif()

juce_add_plugin(AbletonVocoder
    COMPANY_NAME "AbletonVocoder"
    PLUGIN_MANUFACTURER_CODE AbVo
    PLUGIN_CODE Avc2
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Ableton Vocoder"
    VERSION 2.0.0
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    VST3_CATEGORIES Fx
    AU_MAIN_TYPE kAudioUnitType_Effect
    NEEDS_WEB_BROWSER FALSE
    NEEDS_CURL FALSE
)

target_sources(AbletonVocoder PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/VocoderEngine.cpp
)

target_compile_definitions(AbletonVocoder PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(AbletonVocoder PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

juce_add_console_app(vocoder_juce_bench PRODUCT_NAME "vocoder_juce_bench")
target_sources(vocoder_juce_bench PRIVATE
    Source/BenchmarkMain.cpp
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/VocoderEngine.cpp
)
target_compile_definitions(vocoder_juce_bench PRIVATE
    VOC_ENABLE_TEST_HOOKS=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_APPLICATION_NAME_STRING="vocoder_juce_bench"
    JUCE_APPLICATION_VERSION_STRING="2.0.0"
)
target_link_libraries(vocoder_juce_bench PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_audio_processors
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

function(vocoder_release_flags target)
    target_compile_options(${target} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:AppleClang,Clang,GNU>>:-O3>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:AppleClang,Clang,GNU>>:-fno-math-errno>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:AppleClang,Clang,GNU>>:-ffp-contract=fast>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:AppleClang,Clang,GNU>>:-fno-trapping-math>
    )
    if(APPLE)
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-Xarch_arm64>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-mcpu=apple-m1>
        )
        target_link_libraries(${target} PRIVATE "-framework Accelerate")
    endif()
endfunction()

vocoder_release_flags(AbletonVocoder)
vocoder_release_flags(vocoder_juce_bench)
