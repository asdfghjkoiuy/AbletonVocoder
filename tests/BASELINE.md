# v1 baseline

`AbletonVocoder_v1.jsfx` through the offline suite. Numbers are the run stored in [baseline_v1.json](baseline_v1.json).

- command: `tools/.venv/bin/python tools/suite.py --v1`
- ysfx: JoepVanlier/ysfx `5c3452fee62583aa3d1b7e877d0c758c4024af89`
- sample rate 48000, block 512
- wall time about 30 s
- result: PASS 3, FAIL 10, N/A 1 (V0 is one of the three passes)

V0 checks the meters on known signals and on `tests/fixtures/passthrough_carrier.jsfx`. It is not a plugin score. The passthrough ripple is about 0 dB, and YIN on the synthetic voice itself is 4.9 cents median error.

## Tests

| test | result | headline |
| --- | --- | --- |
| V0 self-check | PASS | passthrough 1/3-oct ripple 3.3e-8 dB; YIN voice 4.9 cents |
| T1 flatness | FAIL | 1/3-oct ripple 19.5 dB, 1/12-oct 24.9 dB; retro tilt +1.93 dB/oct |
| T2 envelope | PASS | distortion -59.5 dBc (fast envelope -41.3 dBc, info) |
| T3 enhance | FAIL | slope off -2.11, on -0.04 dB/oct; improvement 2.08 < 3 |
| T4 resynthesis | FAIL | output 21.6 dB quieter; spectrum std 9.6 dB |
| T5 band level | FAIL | 8 bands -41.8 dBFS, 32 bands -70.4 dBFS; spread 28.6 dB |
| T6 gate | PASS | pauses -96.6 dBFS; speech within 0.54 dB of gate-off |
| T7 unvoiced | FAIL | fricatives +72 dB, but voiced frames also +56 dB; Sens 0 residual +36 dB |
| T8 pitch | FAIL | median error 2087 cents; 100% of voiced frames are octave errors |
| T9 channels | FAIL | Mono outputs match; L/R is not a dual vocoder (both about -70 dBFS) |
| T10 robustness | FAIL | silence peak 2.0e-7; automation peak +22.8 dBFS; click ratio 4790 |
| T11 CPU | FAIL | 3.68x realtime at 32 bands (threshold 10x) |
| T12 formant | FAIL | +7 st measures as 0.00 st (correlation 1.0, outputs identical) |
| T13 noise carrier | N/A | v1 has no Noise Rate / Noise Crackle |

## All metrics

| test | metric | value | threshold | result |
| --- | --- | --- | --- | --- |
| V0 | self_pure_sine_dbc | -147.7148 | <= -45 | PASS |
| V0 | self_harmonic_dbc | -20.0000 | -25 .. -15 | PASS |
| V0 | self_slope_error_db_per_oct | 1.776e-15 | abs <= 0.05 | PASS |
| V0 | self_flat_ripple_db | 0.0000 | <= 0.01 | PASS |
| V0 | self_tilt_error_db_per_oct | -1.908e-04 | abs <= 0.15 | PASS |
| V0 | self_shift_error_st | 0.1221 | abs <= 0.2 | PASS |
| V0 | self_audio_shift_error_st | -0.0179 | abs <= 1 | PASS |
| V0 | self_audio_shift_corr | 0.9522 | info | INFO |
| V0 | self_yin_sine_cents | 0.0166 | <= 2 | PASS |
| V0 | self_yin_voiced_frames | 318 | info | INFO |
| V0 | self_yin_voice_median_cents | 4.9052 | <= 20 | PASS |
| V0 | self_yin_voice_octave_pct | 0.3145 | <= 1 | PASS |
| V0 | self_passthrough_ripple_1_3_db | 3.325e-08 | <= 0.2 | PASS |
| V0 | self_passthrough_ripple_1_12_db | 6.094e-08 | <= 0.5 | PASS |
| V0 | self_passthrough_ratio_median_db | -3.913e-10 | abs <= 0.2 | PASS |
| V0 | self_zero_peak | 0 | <= 0 | PASS |
| T1 | precise_output_rms_dbfs | -55.0199 | >= -60 | PASS |
| T1 | ripple_1_3_oct_db | 19.4936 | <= 2 | FAIL |
| T1 | precise_ratio_median_db | -48.1444 | info | INFO |
| T1 | ripple_1_12_oct_db | 24.9345 | <= 6 | FAIL |
| T1 | retro_tilt_db_per_oct | 1.9318 | info | INFO |
| T2 | distortion_dbc | -59.4781 | <= -40 | PASS |
| T2 | distortion_fast_env_dbc | -41.3266 | info | INFO |
| T3 | slope_off_db_per_oct | -2.1125 | info | INFO |
| T3 | slope_on_db_per_oct | -0.0359 | abs <= 1 | PASS |
| T3 | slope_improvement_db_per_oct | 2.0766 | >= 3 | FAIL |
| T4 | rms_delta_db | -21.6465 | abs <= 1.5 | FAIL |
| T4 | spectrum_diff_std_db | 9.6340 | <= 2.5 | FAIL |
| T5 | rms_dbfs_8bands | -41.7550 | info | INFO |
| T5 | rms_dbfs_12bands | -51.6985 | info | INFO |
| T5 | rms_dbfs_16bands | -56.0602 | info | INFO |
| T5 | rms_dbfs_20bands | -63.7579 | info | INFO |
| T5 | rms_dbfs_24bands | -65.3183 | info | INFO |
| T5 | rms_dbfs_28bands | -68.4468 | info | INFO |
| T5 | rms_dbfs_32bands | -70.3582 | info | INFO |
| T5 | rms_spread_db | 28.6032 | <= 2 | FAIL |
| T6 | pause_rms_dbfs | -96.5677 | <= -80 | PASS |
| T6 | speech_vs_gate_off_db | -0.5380 | abs <= 3 | PASS |
| T6 | speech_gate_on_dbfs | -64.3892 | info | INFO |
| T6 | speech_gate_off_dbfs | -63.8512 | info | INFO |
| T7 | fricative_frames | 52 | info | INFO |
| T7 | voiced_frames | 130 | info | INFO |
| T7 | fricative_hf_improvement_db | 71.9836 | >= 10 | PASS |
| T7 | voiced_hf_change_db | 56.2682 | abs <= 3 | FAIL |
| T7 | contrast_delta_db | 15.7154 | info | INFO |
| T7 | sens0_residual_db | 36.0410 | <= -60 | FAIL |
| T7 | sens100_voiced_hf_increase_db | 80.0896 | >= 6 | PASS |
| T8 | voiced_frames | 318 | info | INFO |
| T8 | median_abs_cents | 2086.5927 | <= 25 | FAIL |
| T8 | octave_error_pct | 100.0000 | <= 2 | FAIL |
| T8 | gap_rms_dbfs | -103.5058 | info | INFO |
| T8 | gap_hold_median_abs_cents | 1107.8209 | info | INFO |
| T8 | speech_jump_gt_600_pct | 8.2397 | info | INFO |
| T9 | lr_right_rms_dbfs | -69.8165 | <= -80 | FAIL |
| T9 | lr_left_rms_dbfs | -69.7000 | >= -40 | FAIL |
| T9 | stereo_l_rms_dbfs | -71.3702 | >= -40 | FAIL |
| T9 | stereo_r_rms_dbfs | -71.3742 | >= -40 | FAIL |
| T9 | stereo_lr_correlation | 0.0153 | <= 0.5 | PASS |
| T9 | stereo_lr_correlation_signed | 0.0153 | info | INFO |
| T9 | mono_max_abs_diff | 0 | <= 1e-06 | PASS |
| T10 | silence_peak_44100 | 1.975e-07 | info | INFO |
| T10 | square_peak_dbfs_44100 | -41.6907 | info | INFO |
| T10 | silence_peak_48000 | 1.674e-07 | info | INFO |
| T10 | square_peak_dbfs_48000 | -40.9658 | info | INFO |
| T10 | silence_peak_96000 | 1.582e-07 | info | INFO |
| T10 | square_peak_dbfs_96000 | -43.0570 | info | INFO |
| T10 | silence_peak_abs | 1.975e-07 | <= 1e-09 | FAIL |
| T10 | silence_nan_inf | 0 | == 0 | PASS |
| T10 | square_peak_dbfs | -40.9658 | <= 12 | PASS |
| T10 | square_nan_inf | 0 | == 0 | PASS |
| T10 | auto_peak_dbfs | 22.7550 | <= 12 | FAIL |
| T10 | click_ref | 0.0032 | info | INFO |
| T10 | click_auto | 15.4599 | info | INFO |
| T10 | click_ratio | 4789.7703 | <= 4 | FAIL |
| T11 | realtime_factor | 3.6769 | >= 10 | FAIL |
| T11 | wall_s | 1.0879 | info | INFO |
| T11 | bands | 32 | info | INFO |
| T12 | formant_shift_st | 0.0000 | 5.5 .. 8.5 | FAIL |
| T12 | formant_corr | 1.0000 | info | INFO |
| T13 | feature | n/a | n/a | NA |

## How to read the failures

These are v1 behaviors, measured with the v2 pass lines. The plugin file was not changed.

- T1. Precise mode is not flat. Output sits about 48 dB under the carrier. Retro tilt is positive, as expected.
- T3. Enhance injects noise. It moves the slope from -2.11 to about 0 dB/oct, short of the +3 dB/oct correction the test asks for.
- T4. Carrier=Modulator does not match the input level or the long-term spectrum.
- T5. Level falls steadily as the band count rises. v1 only has 8..32 bands.
- T7. Unvoiced at 0 dB adds high-frequency energy in fricatives and in voiced frames. Sens 0 does not match Unvoiced-off. Sens 100 does add noise to voiced frames.
- T8. On the synthetic voice the output spectrum during the glide sits near 600 Hz, while the ground truth moves 100 to 300 Hz. Median absolute error is 2087 cents. In the silent gaps the output is -104 dBFS, so the oscillator is gated off; the reported gap cents are from a few frames just above the -80 dBFS estimator gate, not a held tone. Speech pitch-jump rate is informational.
- T9. Mono really is identical. L/R mode does not silence the right channel. Stereo correlation is low because the carrier noise is decorrelated, but both sides are about -71 dBFS, under the -40 dBFS activity line.
- T10. No NaN or Inf. Digital silence still leaves a peak of 2.0e-7. Random automation of every slider every 2048 samples reaches +22.8 dBFS and a second-difference 4790 times the constant-slider render. T11 is wall-clock, so a re-run can move the factor by a few percent; 3.7x is far from 10x.
- T12. In Carrier=Modulator, v1 forces the formant shift to 0, so +7 st and 0 st render the same audio.

## Meter choices

- Fractional-octave levels are the mean power spectral density inside the band, in dB. White noise then has slope 0 dB/oct. Integrated band power would add about +3 dB/oct and could not meet the Enhance limit of 1 dB/oct.
- T1 ripple is max minus min of that ratio from 1.25 times Low to 0.8 times High, after 1/3-oct or 1/12-oct smoothing. Output RMS must also be at least -60 dBFS, so a silent render cannot look flat.
- T4 spectrum standard deviation uses 1/3-oct densities from 100 Hz to 8 kHz, and only bins within 40 dB of the input peak. Speech has no energy out to 18 kHz.
- T7 Sens 0 residual is `20*log10(rms(sens0 - off) / rms(off))`.
- T8 counts a voiced frame with no estimate as an octave error. The estimator gate is -80 dBFS.
- T9 passes the correlation check on the absolute correlation.
- T10's click reference is the same 20 s noise with sliders held at the test defaults. Automation starts at sample 2048.
- T12 cross-correlates the first difference of the log-frequency envelope from 400 Hz to 6 kHz (step 0.25 st). The difference stops a sloping spectrum inside a fixed window from pulling the lag toward zero. The self-check recovers a known +7 st shift to within 0.02 st.
