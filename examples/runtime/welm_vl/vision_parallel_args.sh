#!/usr/bin/env bash
# Sourced by both launchers; forward the exact same DP switch to preflight.
vision_dp_args=()
for argument in "$@"; do
    if [[ ${argument} == --mm-enable-dp-encoder ]]; then
        vision_dp_args=(--mm-enable-dp-encoder)
    fi
done
for flag in SGLANG_VLM_VISION_USE_ULYSSES SGLANG_VLM_VISION_USE_DP; do
    value=${!flag:-0}
    case "${value,,}" in
        0|false) value=0 ;;
        1|true) value=1 ;;
        *) echo "${flag} must be 0, 1, false or true." >&2; exit 2 ;;
    esac
    if [[ ${flag} == SGLANG_VLM_VISION_USE_ULYSSES ]]; then
        export SGLANG_VLM_VISION_USE_ULYSSES=${value}
    elif [[ ${value} == 1 ]]; then
        echo 'SGLANG_VLM_VISION_USE_DP is removed; unset it and use --mm-enable-dp-encoder.' >&2
        if (( ${#vision_dp_args[@]} == 0 )); then exit 2; fi
    fi
done
if [[ ${SGLANG_VLM_VISION_USE_ULYSSES} == 1 ]] && (( ${#vision_dp_args[@]} )); then
    echo 'Vision Ulysses and image DP are mutually exclusive.' >&2
    exit 2
fi
