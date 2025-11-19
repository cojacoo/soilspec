"""
Visualization tools for spectral data with band annotations.

Provides functions to plot spectra with chemical band assignments overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, List, Tuple
import pandas as pd

from .band_parser import SpectralBandDatabase


class SpectralPlotter:
    """
    Create annotated spectral plots using band assignments.

    Example:
        >>> from soilspec.knowledge import SpectralPlotter
        >>> plotter = SpectralPlotter()
        >>> fig = plotter.plot_spectrum_with_bands(
        ...     wavenumbers, spectrum, title="Sample soil spectrum"
        ... )
    """

    def __init__(self, spectral_bands_csv: Optional[str] = None):
        """
        Initialize plotter with spectral band database.

        Args:
            spectral_bands_csv: Path to spectral_bands.csv
        """
        self.band_db = SpectralBandDatabase(spectral_bands_csv)

        # Color scheme for different types
        self.type_colors = {
            'org': '#90EE90',      # Light green
            'min': '#FFB6C1',      # Light pink
            'water': '#87CEEB',    # Sky blue
            'rg': '#FFD700'        # Gold
        }

    def plot_spectrum_with_bands(
        self,
        wavenumbers: np.ndarray,
        spectrum: np.ndarray,
        title: str = "Soil MIR Spectrum",
        highlight_regions: Optional[List[str]] = None,
        show_labels: bool = True,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Plot spectrum with chemical band regions highlighted.

        Args:
            wavenumbers: Array of wavenumber values (cm⁻¹)
            spectrum: Array of absorbance/reflectance values
            title: Plot title
            highlight_regions: List of region types to highlight ('org', 'min', 'water')
                              If None, shows key regions only
            show_labels: Whether to show band labels
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure object

        Example:
            >>> plotter = SpectralPlotter()
            >>> fig = plotter.plot_spectrum_with_bands(
            ...     wavenumbers, spectrum,
            ...     highlight_regions=['org', 'min']
            ... )
            >>> plt.show()
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

        # Main spectrum plot
        ax1.plot(wavenumbers, spectrum, 'k-', linewidth=1.5, label='Spectrum')
        ax1.set_ylabel('Absorbance', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Highlight regions
        if highlight_regions is None:
            # Show key regions by default
            regions = self.band_db.get_key_regions()
            for region_name, region_info in regions.items():
                wn_min, wn_max = region_info['range']
                region_type = region_info.get('type', 'org')
                color = self.type_colors.get(region_type, '#CCCCCC')

                ax1.axvspan(
                    wn_min, wn_max,
                    alpha=0.2,
                    color=color,
                    label=region_name if show_labels else None
                )
        else:
            # Highlight specified types
            for band_type in highlight_regions:
                bands = self.band_db.get_bands(type=band_type)
                color = self.type_colors.get(band_type, '#CCCCCC')

                for _, band in bands.iterrows():
                    ax1.axvspan(
                        band['band_start'],
                        band['band_end'],
                        alpha=0.15,
                        color=color
                    )

        if show_labels:
            ax1.legend(loc='upper right', fontsize=9, ncol=2)

        # Band type distribution plot
        self._plot_band_distribution(ax2, wavenumbers)

        ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        ax2.set_ylabel('Band Type', fontsize=10)

        # Share x-axis
        ax1.set_xlim(wavenumbers.min(), wavenumbers.max())
        ax2.set_xlim(wavenumbers.min(), wavenumbers.max())

        # Invert x-axis (standard for IR spectra)
        ax1.invert_xaxis()
        ax2.invert_xaxis()

        plt.tight_layout()
        return fig

    def _plot_band_distribution(self, ax, wavenumbers):
        """
        Plot band type distribution along wavenumber axis.

        Shows which regions are organic, mineral, water, etc.
        """
        # Create color bars for each type
        types = ['org', 'min', 'water']
        y_positions = {'org': 2, 'min': 1, 'water': 0}

        for band_type in types:
            bands = self.band_db.get_bands(type=band_type)
            color = self.type_colors[band_type]

            for _, band in bands.iterrows():
                rect = Rectangle(
                    (band['band_start'], y_positions[band_type] - 0.4),
                    band['band_end'] - band['band_start'],
                    0.8,
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.7
                )
                ax.add_patch(rect)

        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Water', 'Mineral', 'Organic'])
        ax.grid(True, alpha=0.3, axis='x')

    def plot_multiple_spectra(
        self,
        wavenumbers: np.ndarray,
        spectra: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Soil Spectra Comparison",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot multiple spectra with band regions.

        Args:
            wavenumbers: Array of wavenumber values
            spectra: 2D array (n_spectra, n_wavelengths)
            labels: List of labels for each spectrum
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure

        Example:
            >>> plotter = SpectralPlotter()
            >>> fig = plotter.plot_multiple_spectra(
            ...     wavenumbers, spectra,
            ...     labels=['Sample 1', 'Sample 2', 'Sample 3']
            ... )
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot spectra
        n_spectra = spectra.shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_spectra))

        for i in range(n_spectra):
            label = labels[i] if labels else f'Spectrum {i+1}'
            ax.plot(wavenumbers, spectra[i, :], color=colors[i],
                   linewidth=1.5, label=label, alpha=0.8)

        # Add key region shading
        regions = self.band_db.get_key_regions()
        for region_name, region_info in list(regions.items())[:5]:  # Top 5 regions
            wn_min, wn_max = region_info['range']
            ax.axvspan(wn_min, wn_max, alpha=0.1, color='gray')

        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Absorbance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        plt.tight_layout()
        return fig

    def annotate_peaks(
        self,
        wavenumbers: np.ndarray,
        spectrum: np.ndarray,
        peak_wavenumbers: List[float],
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot spectrum with specific peaks annotated with chemical information.

        Args:
            wavenumbers: Array of wavenumber values
            spectrum: Spectrum array
            peak_wavenumbers: List of wavenumbers to annotate
            figsize: Figure size

        Returns:
            Matplotlib figure

        Example:
            >>> plotter = SpectralPlotter()
            >>> # Annotate key peaks
            >>> fig = plotter.annotate_peaks(
            ...     wavenumbers, spectrum,
            ...     peak_wavenumbers=[2920, 1630, 1030, 3620]
            ... )
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot spectrum
        ax.plot(wavenumbers, spectrum, 'k-', linewidth=1.5)

        # Annotate each peak
        for peak_wn in peak_wavenumbers:
            # Get chemical information
            info = self.band_db.get_region_type(peak_wn)

            if info:
                # Find closest wavenumber in spectrum
                idx = np.argmin(np.abs(wavenumbers - peak_wn))
                peak_value = spectrum[idx]

                # Create annotation text
                main_info = info[0]  # First match (highest priority)
                annotation = f"{peak_wn:.0f} cm⁻¹\n{main_info['information']}"

                # Annotate
                ax.annotate(
                    annotation,
                    xy=(peak_wn, peak_value),
                    xytext=(0, 20),
                    textcoords='offset points',
                    fontsize=9,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Absorbance', fontsize=12)
        ax.set_title('Annotated Spectrum', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        plt.tight_layout()
        return fig

    def plot_band_importance(
        self,
        importance_scores: np.ndarray,
        wavenumbers: np.ndarray,
        title: str = "Spectral Feature Importance",
        top_n: int = 10,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Plot feature importance with chemical band interpretation.

        Useful for visualizing which wavelengths contribute to model predictions.

        Args:
            importance_scores: Array of importance values (same length as wavenumbers)
            wavenumbers: Array of wavenumber values
            title: Plot title
            top_n: Number of top regions to annotate
            figsize: Figure size

        Returns:
            Matplotlib figure

        Example:
            >>> # From SHAP or permutation importance
            >>> importance = shap_values.abs().mean(0)
            >>> plotter = SpectralPlotter()
            >>> fig = plotter.plot_band_importance(importance, wavenumbers)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

        # Importance plot
        ax1.plot(wavenumbers, importance_scores, 'b-', linewidth=1.5)
        ax1.fill_between(wavenumbers, importance_scores, alpha=0.3)
        ax1.set_ylabel('Importance Score', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Find and annotate top peaks
        top_indices = np.argsort(importance_scores)[-top_n:]

        for idx in top_indices:
            wn = wavenumbers[idx]
            score = importance_scores[idx]

            # Get chemical info
            info = self.band_db.get_region_type(wn)
            if info:
                label = info[0]['information']
            else:
                label = f"{wn:.0f} cm⁻¹"

            ax1.annotate(
                label,
                xy=(wn, score),
                xytext=(0, 10),
                textcoords='offset points',
                fontsize=8,
                ha='center',
                rotation=45,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.6),
                arrowprops=dict(arrowstyle='->', lw=0.5)
            )

        # Band distribution
        self._plot_band_distribution(ax2, wavenumbers)
        ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        ax2.set_ylabel('Band Type', fontsize=10)

        # Share x-axis
        ax1.set_xlim(wavenumbers.min(), wavenumbers.max())
        ax2.set_xlim(wavenumbers.min(), wavenumbers.max())
        ax1.invert_xaxis()
        ax2.invert_xaxis()

        plt.tight_layout()
        return fig


def plot_band_summary(spectral_bands_csv: Optional[str] = None) -> plt.Figure:
    """
    Create summary visualization of spectral band database.

    Shows distribution of bands by type and wavenumber range.

    Args:
        spectral_bands_csv: Path to spectral_bands.csv

    Returns:
        Matplotlib figure

    Example:
        >>> from soilspec.knowledge import plot_band_summary
        >>> fig = plot_band_summary()
        >>> plt.show()
    """
    band_db = SpectralBandDatabase(spectral_bands_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Band count by type
    type_counts = band_db.bands['type'].value_counts()
    axes[0, 0].bar(type_counts.index, type_counts.values, color='steelblue')
    axes[0, 0].set_xlabel('Band Type', fontsize=11)
    axes[0, 0].set_ylabel('Number of Bands', fontsize=11)
    axes[0, 0].set_title('Band Count by Type', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Wavenumber distribution
    all_wn = []
    all_types = []
    for _, row in band_db.bands.iterrows():
        all_wn.append(row['band_start'])
        all_types.append(row['type'])

    type_colors = {'org': 'green', 'min': 'red', 'water': 'blue', 'rg': 'orange'}
    for band_type in band_db.bands['type'].unique():
        mask = np.array(all_types) == band_type
        wn_subset = np.array(all_wn)[mask]
        axes[0, 1].hist(
            wn_subset,
            bins=30,
            alpha=0.6,
            label=band_type,
            color=type_colors.get(band_type, 'gray')
        )

    axes[0, 1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    axes[0, 1].set_ylabel('Number of Bands', fontsize=11)
    axes[0, 1].set_title('Wavenumber Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Top references
    refs = band_db.get_references().head(10)
    axes[1, 0].barh(range(len(refs)), refs['num_bands'], color='coral')
    axes[1, 0].set_yticks(range(len(refs)))
    axes[1, 0].set_yticklabels(refs['reference'], fontsize=9)
    axes[1, 0].set_xlabel('Number of Bands', fontsize=11)
    axes[1, 0].set_title('Top 10 References', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Summary statistics
    summary = band_db.summarize()
    summary_text = (
        f"Total Bands: {summary['total_bands']}\n\n"
        f"By Type:\n"
    )
    for band_type, count in sorted(summary['by_type'].items()):
        summary_text += f"  {band_type}: {count}\n"

    summary_text += f"\nWavenumber Range:\n"
    summary_text += f"  {summary['wavenumber_range'][0]:.0f} - "
    summary_text += f"{summary['wavenumber_range'][1]:.0f} cm⁻¹\n\n"
    summary_text += f"Unique References: {summary['unique_references']}"

    axes[1, 1].text(
        0.1, 0.5, summary_text,
        fontsize=11,
        verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Database Summary', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig
