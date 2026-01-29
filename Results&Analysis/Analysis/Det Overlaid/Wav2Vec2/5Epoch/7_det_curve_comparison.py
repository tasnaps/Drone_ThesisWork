import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
from scipy.stats import norm


def _parse_epoch(filename: str):
    """Extract epoch number from filename; returns int or None."""
    m = re.search(r"(\d+)\s*Epoch", filename, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_augmentation_label(filename: str) -> str:
    """Extract a readable augmentation label from wav2vec2-style filenames.

    Expected examples:
      - 1EpochAllAugmentsCalibrationDataset_detailed_files.csv
      - 1EpochBandPassCalibrationDataset_detailed_files.csv
      - 1EpochNoAugmentCalibrationDataset_detailed_files.csv

    Falls back to a cleaned filename if pattern doesn't match.
    """
    base = re.sub(r"\.csv$", "", filename, flags=re.IGNORECASE)

    # Try to capture AUGMENT between 'Epoch' and 'CalibrationDataset'
    m = re.search(r"Epoch(?P<aug>.+?)CalibrationDataset", base, flags=re.IGNORECASE)
    if m:
        aug = m.group('aug')
        aug = re.sub(r"[_\-]+", " ", aug)
        aug = re.sub(r"(?<!^)([A-Z])", r" \1", aug).strip()  # split CamelCase
        return aug

    # Generic fallback: remove common suffix and prettify
    base = re.sub(r"_detailed_files.*$", "", base, flags=re.IGNORECASE)
    base = base.replace("_", " ")
    return base


def plot_combined_det_curves():
    # Get all CSV files in the current directory
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("Ei CSV-tiedostoja nykyisessä kansiossa.")
        return

    # Try to detect single-epoch comparison; we focus on augmentation labels.
    file_infos = []
    for f in csv_files:
        epoch = _parse_epoch(f)
        aug = _parse_augmentation_label(f)
        file_infos.append((f, epoch, aug))

    # Stable order: augmentation name, then epoch (if present)
    file_infos.sort(key=lambda t: (t[2].lower(), t[1] if t[1] is not None else 10**9, t[0].lower()))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#ffffff')

    # Probit scale plot bounds (these match the ticks below)
    p_min = 0.001
    p_max = 0.80
    x_min = norm.ppf(p_min)
    x_max = norm.ppf(p_max)

    # Distinct colors per file/augmentation
    base_cmap = plt.get_cmap('tab10' if len(file_infos) <= 10 else 'tab20')

    # Keep linestyles subtle
    default_linestyles = ['-', '--', ':', '-.']

    any_oob = False

    for idx, (csv_file, epoch, aug) in enumerate(file_infos):
        print(f"Käsitellään: {csv_file}")

        df = pd.read_csv(csv_file)

        # Check for required columns
        if 'true_label' in df.columns and 'drone_probability' in df.columns:
            y_true = df['true_label'].values
            y_scores = df['drone_probability'].values
        else:
            print(f"  VIRHE: Pakollisia sarakkeita ei löytynyt tiedostosta {csv_file}")
            continue

        # Compute DET curve
        fpr, fnr, _ = det_curve(y_true, y_scores)

        # Convert to probit coords
        x = norm.ppf(fpr)
        y = norm.ppf(fnr)

        # Detect out-of-bounds before plotting (line may get clipped by axis limits)
        oob = (
            (x < x_min).any() or (x > x_max).any() or
            (y < x_min).any() or (y > x_max).any()
        )
        any_oob = any_oob or oob

        # Label: augmentation (and epoch if not constant)
        label = aug
        if epoch is not None:
            label = f"{label} ({epoch} ep)"

        color = base_cmap(idx % base_cmap.N)
        linestyle = default_linestyles[idx % len(default_linestyles)]

        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.6,
            alpha=0.95,
        )

    # Set up probit scale axes
    ticks = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.80]
    tick_labels = ['0,1%', '1,0%', '5,0%', '10,0%', '20,0%', '30,0%', '40,0%', '80,0%']
    tick_locations = norm.ppf(ticks)

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)

    ax.set_xlabel('Väärä positiivinen osuus (FPR) – probit-asteikko', fontsize=12)
    ax.set_ylabel('Väärä negatiivinen osuus (FNR) – probit-asteikko', fontsize=12)
    ax.set_title('DET-käyrät (wav2vec2, 5 epookki) – augmentaatiovertailu', fontsize=14)

    legend_title = 'Augmentaatio'

    ax.legend(loc='upper left', fontsize=10, title=legend_title)
    ax.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout()
    plt.savefig('combined_det_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Kuva tallennettu nimellä 'combined_det_curves.png'")


if __name__ == "__main__":
    plot_combined_det_curves()