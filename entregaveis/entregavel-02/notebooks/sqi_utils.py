# ============================================================
# SQI Utils — Métricas de Qualidade de ECG
# ============================================================

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats


# ============================================================
# 1. HOS SQI
# ============================================================

def compute_hos_sqi(sig):
    sSQI = stats.skew(sig, axis=0)
    kSQI = stats.kurtosis(sig, axis=0, fisher=False)

    hosSQI = np.minimum((np.abs(sSQI) * kSQI) / 5, 10)

    return sSQI, kSQI, hosSQI


# ============================================================
# 2. Frequência (pSQI e basSQI)
# ============================================================

def compute_freq_sqi(sig, fs):
    psqi_list = []
    bassqi_list = []

    for i in range(sig.shape[1]):
        freqs, psd = signal.welch(sig[:, i], fs=fs, nperseg=256)

        mask_qrs = (freqs >= 5) & (freqs <= 15)
        mask_5_40 = (freqs >= 5) & (freqs <= 40)
        mask_above_1 = (freqs >= 1)
        mask_0_40 = (freqs <= 40)

        energy_qrs = np.sum(psd[mask_qrs])
        energy_5_40 = np.sum(psd[mask_5_40])
        energy_above_1 = np.sum(psd[mask_above_1])
        energy_0_40 = np.sum(psd[mask_0_40])

        psqi = energy_qrs / energy_5_40 if energy_5_40 > 0 else 0
        bassqi = energy_above_1 / energy_0_40 if energy_0_40 > 0 else 0

        psqi_list.append(psqi)
        bassqi_list.append(bassqi)

    return np.array(psqi_list), np.array(bassqi_list)


# ============================================================
# 3. Flatline SQI
# ============================================================

def compute_flatline_sqi(sig, fs, threshold=1e-3, min_duration=1.5):
    flat_ratios = []
    flatline_flags = []

    min_samples = int(min_duration * fs)

    for i in range(sig.shape[1]):
        lead = sig[:, i]

        diff = np.abs(np.diff(lead))
        flat_mask = diff < threshold

        flat_ratio = np.sum(flat_mask) / len(flat_mask)

        max_len = 0
        current_len = 0

        for val in flat_mask:
            if val:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0

        has_flatline = max_len >= min_samples

        flat_ratios.append(flat_ratio)
        flatline_flags.append(int(has_flatline))

    return np.array(flat_ratios), np.array(flatline_flags)


# ============================================================
# 4. qSQI
# ============================================================

def compute_qsqi(sig, fs):
    qsqi_list = []

    for i in range(sig.shape[1]):
        lead = sig[:, i]

        peaks, _ = signal.find_peaks(lead, distance=fs * 0.4)

        if len(peaks) < 3:
            qsqi_list.append(0)
            continue

        rr_intervals = np.diff(peaks) / fs

        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)

        qsqi = 1 - (rr_std / rr_mean) if rr_mean > 0 else 0
        qsqi = max(0, qsqi)

        qsqi_list.append(qsqi)

    return np.array(qsqi_list)


# ============================================================
# 5. Correlação inter-derivações
# ============================================================

def compute_rinter(sig):
    n_leads = sig.shape[1]
    correlations = []

    for i in range(n_leads):
        for j in range(i + 1, n_leads):

            std_i = np.std(sig[:, i])
            std_j = np.std(sig[:, j])

            if std_i < 1e-6 or std_j < 1e-6:
                continue

            corr = np.corrcoef(sig[:, i], sig[:, j])[0, 1]

            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0


# ============================================================
# 6. SNR
# ============================================================

def compute_snr(sig, fs):
    nyq = 0.5 * fs
    b, a = signal.butter(4, [0.5/nyq, 40/nyq], btype='bandpass')

    s_clean = signal.filtfilt(b, a, sig, axis=0)
    noise = sig - s_clean

    rms_clean = np.sqrt(np.mean(s_clean**2, axis=0))
    rms_noise = np.sqrt(np.mean(noise**2, axis=0))

    eps = 1e-10
    rms_clean = np.maximum(rms_clean, eps)
    rms_noise = np.maximum(rms_noise, eps)

    snr_db = 10 * np.log10((rms_clean**2) / (rms_noise**2))
    snr_db = np.clip(snr_db, -20, 60)

    return np.median(snr_db)


# ============================================================
# 7. Entropia espectral
# ============================================================

def compute_spectral_entropy(sig, fs):
    entropies = []

    for i in range(sig.shape[1]):
        lead = sig[:, i]

        freqs, psd = signal.welch(lead, fs=fs, nperseg=256)

        psd_sum = np.sum(psd)
        if psd_sum == 0:
            continue

        psd_norm = psd / psd_sum

        H = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        H_norm = H / np.log2(len(psd))

        entropies.append(H_norm)

    return np.mean(entropies) if entropies else 1.0


# ============================================================
# 8. Função principal
# ============================================================

def compute_all_sqi(sig, fs):
    sSQI, kSQI, hosSQI = compute_hos_sqi(sig)
    pSQI, basSQI = compute_freq_sqi(sig, fs)
    flat_ratio, flat_flag = compute_flatline_sqi(sig, fs)
    qsqi = compute_qsqi(sig, fs)
    rinter = compute_rinter(sig)
    snr = compute_snr(sig, fs)
    entropy = compute_spectral_entropy(sig, fs)

    return {
        'hosSQI_mean': np.mean(hosSQI),
        'hosSQI_median': np.median(hosSQI),

        'pSQI_mean': np.mean(pSQI),
        'basSQI_mean': np.mean(basSQI),

        'snr_db': snr,
        'spectral_entropy': entropy,

        'flat_ratio_max': np.max(flat_ratio),
        'has_flatline': int(np.any(flat_flag)),

        'qSQI_mean': np.mean(qsqi),
        'rinter': rinter
    }

    # ============================================================
# 9. Thresholds de Classificação
# ============================================================

DEFAULT_THRESHOLDS = {
    'threshold_flat_ratio_max': 0.25,

    'threshold_snr_db_U': 0,
    'threshold_snr_db_A': 10,

    'threshold_qSQI_mean_A': 0.9,
    'threshold_qSQI_mean_U': 0.6,

    'threshold_basSQI_A': 0.95,
    'threshold_basSQI_U': 0.905,

    'threshold_spectral_entropy_A': 0.8,
    'threshold_spectral_entropy_U': 0.95,

    'threshold_hosSQI_mean_A': 0.8,
    'threshold_hosSQI_mean_U': 0.5,

    'threshold_rinter_A': 0.6,
    'threshold_rinter_U': 0.3,

    'threshold_pSQI_mean_A_sup': 0.8,
    'threshold_pSQI_mean_A_inf': 0.5,
    'threshold_pSQI_mean_U': 0.4,
}


# ============================================================
# 10. Classificação de Qualidade (SQI)
# ============================================================

def classify_sqi(df, thresholds=None, return_debug=False):
    """
    Classifica sinais em:
    G (Excelente), A (Aceitável), P (Processável), U (Inaceitável)

    df deve conter:
    hosSQI_mean, pSQI_mean, basSQI_mean, snr_db,
    spectral_entropy, flat_ratio_max, has_flatline,
    qSQI_mean, rinter
    """

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    th = thresholds
    n = len(df)

    result = np.full(n, 'A', dtype=object)

    # -------------------------
    # 1. HARD GATE (U direto)
    # -------------------------
    mask_U = (
        (df['flat_ratio_max'] > th['threshold_flat_ratio_max']) |
        (df['snr_db'] < th['threshold_snr_db_U']) |
        (df['has_flatline']) |
        (df['qSQI_mean'] < th['threshold_qSQI_mean_U'])
    )

    result[mask_U] = 'U'

    # -------------------------
    # 2. Problemas corrigíveis
    # -------------------------
    corrigivel = np.zeros(n)

    corrigivel += (df['basSQI_mean'] < th['threshold_basSQI_U']).values
    corrigivel += (df['snr_db'] < th['threshold_snr_db_A']).values
    corrigivel += (df['spectral_entropy'] > th['threshold_spectral_entropy_A']).values
    corrigivel += (df['hosSQI_mean'] < th['threshold_hosSQI_mean_U']).values

    # -------------------------
    # 3. Votação
    # -------------------------
    score_G = np.zeros(n)
    score_A_fail = np.zeros(n)

    # hosSQI
    score_G += (df['hosSQI_mean'] > th['threshold_hosSQI_mean_A']).values
    score_A_fail += (df['hosSQI_mean'] < th['threshold_hosSQI_mean_U']).values

    # qSQI
    score_G += (df['qSQI_mean'] > th['threshold_qSQI_mean_A']).values
    score_A_fail += (df['qSQI_mean'] < th['threshold_qSQI_mean_U']).values

    # rinter
    score_G += (df['rinter'] > th['threshold_rinter_A']).values
    score_A_fail += (df['rinter'] < th['threshold_rinter_U']).values

    # pSQI
    score_G += (
        (df['pSQI_mean'] >= th['threshold_pSQI_mean_A_inf']) &
        (df['pSQI_mean'] <= th['threshold_pSQI_mean_A_sup'])
    ).values
    score_A_fail += (df['pSQI_mean'] < th['threshold_pSQI_mean_U']).values

    # basSQI
    score_G += (df['basSQI_mean'] > th['threshold_basSQI_A']).values
    score_A_fail += (df['basSQI_mean'] < th['threshold_basSQI_U']).values

    # spectral entropy
    score_G += (df['spectral_entropy'] <= th['threshold_spectral_entropy_A']).values
    score_A_fail += (df['spectral_entropy'] > th['threshold_spectral_entropy_U']).values

    # -------------------------
    # 4. Decisão final
    # -------------------------

    mask_P = (corrigivel >= 2) & (~mask_U)

    mask_G = (
        (score_G >= 5) &
        (corrigivel <= 1) &
        (~mask_U)
    )

    result[mask_G] = 'G'

    mask_U2 = (
        (score_A_fail >= 3) |
        ((score_A_fail >= 2) & (corrigivel <= 1))
    ) & (~mask_U) & (~mask_P)

    result[mask_U2] = 'U'

    result[mask_P] = 'P'

    if return_debug:
        return pd.DataFrame({
            'quality_class': result,
            'corrigivel': corrigivel,
            'score_G': score_G,
            'score_A_fail': score_A_fail
        }, index=df.index)

    return pd.Series(result, index=df.index)