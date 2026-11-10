# Experiment notes

## Data pipeline

1. Downloaded the official DIV2K HR/LR ×4 train and validation archives and Flickr2K archive using `scripts/data/download_training_data.py`. The downloader uses two concurrent HTTP range workers, retries, validates the server's `Content-Range`, and records SHA-256. The observed hashes are in [`dataset_sha256.txt`](../results/dataset_sha256.txt).
2. Extracted only the images needed for ×4 bicubic super-resolution. Checked the PNG headers, image count, and exact ×4 dimensions for 800 DIV2K training pairs, 2,650 Flickr2K pairs, and 100 DIV2K validation pairs.
3. Used the upstream SAFMN crop worker to create aligned 480×480 HR and 120×120 LR crops with strides 240/60. The recorded run produced 139,995 pairs. A BasicSR metadata index was generated after confirming HR/LR filenames matched, avoiding slow folder-pair enumeration at startup.

The source images and crop files remain on the cloud data disk and are not part of this repository. The Set5 files used for external testing are from a third-party mirror.

## Training sequence

The cloud run used a project-local `.venv`, one RTX 4090, and the SAFMN repository commit in the main README. The network settings, batch size 64, optimizer, L1/FFT losses, and 500k cosine schedule follow the pinned SAFMN configuration. The data paths and BasicSR `meta_info_file` index were adapted to the cloud dataset layout.

| Stage | Purpose | Result |
| --- | --- | --- |
| 30-step smoke run | Verify data, optimizer, validation, and save path | Checkpoint/state saved; not a quality result |
| 100-step data-loader checks | Compare full-image decoding and prepared crop loading | 1.021 versus 0.460 seconds/step in these short runs |
| 5k training stage | First interpretable checkpoint | DIV2K val100 RGB PSNR 28.1530 dB (EMA) |
| 5k without FFT loss | Single loss ablation against the 5k stage | DIV2K val100 RGB PSNR 27.7030 dB (EMA) |
| Resume from `5000.state` | Continue the same experiment | 30k EMA PSNR 28.7915 dB; stopped after checkpoint and validation |

The [no-FFT configuration](../configs/safmn/train/SAFMN_train_DF2K_x4_sub_nofft_5000.yml) differs from the first 5k configuration only in the experiment name and disabling `fft_opt`; the random seed, data, batch size, network, optimizer, and 500k-period learning-rate schedule match. Both runs stopped at 5k and used EMA validation on DIV2K val100 with RGB PSNR and a four-pixel border crop. The [no-FFT log](../results/logs/nofft5000.stdout.log) reports 27.7030 dB at iteration 5000, **0.4500 dB below** the 28.1530 dB FFT-loss run. Its process exited normally after training and final validation. The `net_g_5000.pth` checkpoint loaded with both `params` and `params_ema`; `5000.state` loaded with iteration 5000. This is one short run per setting, not evidence for a full 500k-step conclusion or a statistically reliable effect.

| No-FFT 5k cloud file (kept outside this repository) | SHA-256 |
| --- | --- |
| `net_g_5000.pth` | `3afcd792ce1dbe315237bf7cdbc478c04cb5a9f96b753874253e47dfa7f7bcb1` |
| `5000.state` | `c0c36c735f5512c9dadfaf56f36b7ed03486c290c3c78bd479a0df911186ea7b` |

The resume log is [`full500k_resume.stdout.log`](../results/logs/full500k_resume.stdout.log). The 30k checkpoint held both `params` and `params_ema`. Its optimizer state reported iteration 30000. The files were readable in PyTorch, and local/cloud SHA-256 values matched:

| File | SHA-256 |
| --- | --- |
| `net_g_30000.pth` | `c8f372864a1479fb555d1f2a4793b31ce5d1e6a55741ca2393b37606dc0370f3` |
| `30000.state` | `d35fabbd2c8108ce62b3c0c68a4835294e1027a0121cff469c0c050f5014ebbe` |

The experiment-specific [`stop_safmn_at_50000.py`](../scripts/training/stop_safmn_at_50000.py) observed each saved checkpoint and validation score. From 20k onward it would stop after a complete checkpoint if the EMA PSNR was within 0.20 dB of the pretrained baseline and its 10k gain was below 0.05 dB; 50k was the upper limit. It sent SIGTERM at 30k after PSNR 28.7915 dB was logged. Remaining orphan data-loader workers were stopped separately.

## Reading the metrics correctly

SAFMN's training validation calls `SRModel.test()`, which selects `net_g_ema` when it exists. A direct test config loads `params` unless configured otherwise. Thus the training EMA score 28.7915 dB and the direct 30k ordinary-weight score 28.6901 dB answer different questions. For a matching weight-key comparison with the author's pretrained `params`, use 28.6901 versus 28.9615 dB on the same 100 DIV2K images, RGB PSNR, border crop 4.

The 30k Set5 result for ordinary weights was 31.7874 dB PSNR / 0.8895 SSIM on the Y channel. The author's pretrained SAFMN weights scored 32.1836 / 0.8948 with the same five-image mirror and protocol. This small dataset does not establish general performance. CAMixerSR's pretrained Set5 test is a separate model check; its subsequent from-scratch short training is documented below.

An independent retest of the 30k EMA weights used [`SAFMN_DIV2K_val100_30000_ema.yml`](../configs/safmn/test/SAFMN_DIV2K_val100_30000_ema.yml). It differs from the ordinary-weight test configuration only in the experiment name and `path.param_key_g: params_ema`. BasicSR's direct test defaults to `params` when this key is absent. On the same 100 DIV2K validation images and RGB PSNR protocol, the direct EMA test returned **28.7915 dB**, matching the training-loop EMA validation to four decimal places. The [direct test log](../results/logs/SAFMN_DIV2K_val100_30000_ema.stdout.log) records the image count, loaded parameter key, and score. Adjust the cloud paths before using this configuration on another machine.

The first direct invocation found the installed BasicSR package and failed before network creation because that registry lacked SAFMN. The successful command set `PYTHONPATH=/root/fsas/sr-repro/SAFMN` so Python imported the BasicSR code bundled with SAFMN. This import path matters when the environment also has BasicSR installed separately.

Stopping at 30k was a time-budget decision informed by the validation trend. The original scheduler still had a 500k period and the full author training target was not reached.

## CAMixerSR from-scratch 5k training — 2026-10-22

The CAMixerSR run completed 5,000 optimizer updates on one RTX 5090, using the existing 139,995 paired DF2K crops, random HR/LR patches of 256×256 / 64×64, batch size 16, and seed 10. The project-local environment used Python 3.10.12, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, and BasicSR 1.4.2. Full environment versions are saved in [`camixersr_5090_environment.txt`](../results/camixersr_5090_environment.txt).

The [active configuration](../configs/camixersr/train/CAMixerSR_train_DF2K_x4_cloud_5000.yml) uses the author's `CAModel` training class, L1 plus the routing constraint, AdamW (learning rate 0.0005, weight decay 0.0001, betas 0.9/0.99), and EMA decay 0.999. The original long-schedule milestones remain beyond this short run, so the learning rate did not decay. The original 1,000-step plan was superseded by this 5,000-step configuration before formal training began. Project registration and the missing network ratio attribute were fixed with the saved [training entry patch](../patches/CAMixerSR_training_entry.patch); the setup and compatibility changes are described in the [training plan](CAMixerSR-short-training-plan.md).

Every 1,000 updates, the run saved the network and optimizer/scheduler state and evaluated the EMA network on all 100 DIV2K validation pairs. The protocol is RGB PSNR with a four-pixel border crop.

| Optimizer updates | Scheduled EMA validation PSNR (dB) |
| --- | ---: |
| 1,000 | 17.3931 |
| 2,000 | 22.1289 |
| 3,000 | 25.4935 |
| 4,000 | 26.8954 |
| 5,000 | **27.5543** |

The training script calls validation again after its training loop and latest-model save. That additional evaluation logged **27.5540 dB**, a difference of 0.0003 dB. Both values are retained separately; the scheduled 5,000-update score is the primary result. The extra validation did not perform another optimizer update. The small difference has not been isolated to a specific cause. See the [complete training log](../results/logs/camixersr5000.stdout.log) and [machine-readable result](../results/camixersr5000.result.json).

The log reports a training-loop duration of 35 minutes 46 seconds; the additional final validation finished at 20:28:39 Asia/Shanghai. No traceback or non-finite logged loss was found. At 20:29–20:30, the original training process (PID 276) no longer existed and `nvidia-smi` reported no GPU compute processes. The recurring monitor was paused after verification.

The [checkpoint verification record](../results/camixersr5000.checkpoint_verification.json) confirms that `net_g_5000.pth` and `5000.state` loaded successfully with the training environment. The state reports iteration 5000 and contains one optimizer and one scheduler. The checkpoint contains both `params` and `params_ema`, each with 718 tensors and 765,322 elements; all values are finite. All five scheduled model/state pairs were present. The binary files remain on the cloud shared disk and are excluded from this repository.

| Cloud file | SHA-256 |
| --- | --- |
| `net_g_5000.pth` | `3f44eed645ae11c14c4af0b12091a359031d141cef74a129e1d761b3fe4d6f63` |
| `5000.state` | `41489c0e90b4ef9b3eb6de12a38e674fbae7a82728a74c93b175bdc079a4d893` |
| Active training YAML | `32ada9a33c571a0df521c5ca66e4f50eb94d03a019fefe1088ece60a70ba36d4` |

This is one 5,000-step short training run, not the author's full training or a fair network ranking against SAFMN. The batch size, optimizer, losses, learning-rate schedule, and hardware differ from the earlier SAFMN experiments. The result verifies CAMixerSR's data, training, checkpoint, and EMA validation path through BasicSR; it does not establish a controlled architecture comparison. Ending the process releases the GPU workload, but the cloud instance must be stopped separately in the provider's console to stop instance billing.
