"""
Chemical consistency constraints based on soil science domain knowledge.

These constraints encode empirical relationships between soil properties,
not physical laws or differential equations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ChemicalConstraints:
    """
    Encode known relationships between soil properties.

    These are empirical rules from soil science literature, not physics equations.
    Used for regularization in multi-task learning and validation of predictions.

    Example:
        >>> from soilspec_pinn.knowledge import ChemicalConstraints
        >>> constraints = ChemicalConstraints()
        >>> expected_cec = constraints.cec_constraint(soc=2.5, clay=25)
        >>> validation = constraints.validate_prediction({'SOC': 2.5, 'clay': 25})
    """

    @staticmethod
    def cec_constraint(soc: float, clay: float, silt: Optional[float] = None) -> float:
        """
        Estimate CEC (cation exchange capacity) from SOC and texture.

        Empirical relationship: CEC depends on organic matter and clay content.
        Typical coefficients (vary by soil type):
        - Clay: 0.3-0.7 cmol/kg per % clay
        - SOC: 1.5-3.0 cmol/kg per % SOC
        - Silt: 0.1-0.3 cmol/kg per % silt (minor contribution)

        Args:
            soc: Soil organic carbon (%)
            clay: Clay content (%)
            silt: Silt content (%) - optional

        Returns:
            Expected CEC in cmol/kg

        Reference:
            Obalum et al. (2012). Soil organic matter as sole indicator of soil
            degradation. Environmental Monitoring and Assessment.

        Example:
            >>> constraints = ChemicalConstraints()
            >>> cec = constraints.cec_constraint(soc=2.5, clay=25)
            >>> print(f"Expected CEC: {cec:.1f} cmol/kg")
        """
        # Conservative mid-range coefficients
        cec = 0.5 * clay + 2.0 * soc

        if silt is not None:
            cec += 0.2 * silt

        return cec

    @staticmethod
    def texture_constraint(clay: float, silt: float, sand: float) -> float:
        """
        Check if texture fractions sum to 100%.

        Args:
            clay: Clay content (%)
            silt: Silt content (%)
            sand: Sand content (%)

        Returns:
            Residual (should be near 0)

        Example:
            >>> constraints = ChemicalConstraints()
            >>> residual = constraints.texture_constraint(25, 40, 35)
            >>> print(f"Texture sum: {100 + residual:.1f}%")
        """
        return (clay + silt + sand) - 100.0

    @staticmethod
    def ph_buffering_capacity(soc: float, clay: float) -> str:
        """
        Estimate pH buffering capacity based on SOC and clay.

        Higher SOC and clay content → stronger pH buffering.

        Args:
            soc: Soil organic carbon (%)
            clay: Clay content (%)

        Returns:
            Buffering capacity category ('low', 'medium', 'high')

        Reference:
            Bolan et al. (2003). Soil acidification and liming interactions
            with nutrient and heavy metal transformation and bioavailability.

        Example:
            >>> constraints = ChemicalConstraints()
            >>> buffering = constraints.ph_buffering_capacity(soc=3.5, clay=30)
            >>> print(f"pH buffering: {buffering}")
        """
        # Simple heuristic
        buffering_score = 0.3 * clay + 0.7 * soc

        if buffering_score > 15:
            return 'high'
        elif buffering_score > 8:
            return 'medium'
        else:
            return 'low'

    @staticmethod
    def soc_clay_correlation(soc: float, clay: float) -> bool:
        """
        Check if SOC and clay relationship is reasonable.

        Generally, SOC increases with clay content due to:
        - Physical protection of OM in aggregates
        - Clay mineral surface interactions
        - Reduced decomposition rates

        Typical SOC:clay ratios range from 0.03 to 0.15.

        Args:
            soc: Soil organic carbon (%)
            clay: Clay content (%)

        Returns:
            True if relationship is reasonable, False if suspicious

        Reference:
            Baldock & Skjemstad (2000). Role of the soil matrix and minerals
            in protecting natural organic materials against biodegradation.

        Example:
            >>> constraints = ChemicalConstraints()
            >>> reasonable = constraints.soc_clay_correlation(soc=0.5, clay=50)
            >>> if not reasonable:
            >>>     print("Warning: SOC unusually low for clay content")
        """
        if clay < 5:  # Sandy soil - SOC can vary widely
            return True

        ratio = soc / clay if clay > 0 else 0

        # Typical range: 0.03 to 0.15
        # Allow wider tolerance for edge cases
        return 0.01 <= ratio <= 0.25

    @staticmethod
    def nitrogen_carbon_ratio(total_n: float, soc: float) -> Tuple[float, str]:
        """
        Calculate C:N ratio and check if it's reasonable.

        Typical soil C:N ratios:
        - Agricultural soils: 8-15
        - Forest soils: 15-25
        - Peatlands: 20-40

        Args:
            total_n: Total nitrogen (%)
            soc: Soil organic carbon (%)

        Returns:
            Tuple of (C:N ratio, interpretation)

        Reference:
            Stevenson & Cole (1999). Cycles of Soil: Carbon, Nitrogen,
            Phosphorus, Sulfur, Micronutrients.

        Example:
            >>> constraints = ChemicalConstraints()
            >>> cn_ratio, interpretation = constraints.nitrogen_carbon_ratio(
            ...     total_n=0.2, soc=2.5
            ... )
            >>> print(f"C:N = {cn_ratio:.1f} ({interpretation})")
        """
        if total_n <= 0:
            return np.nan, "Invalid: N must be positive"

        # Convert %C to C:N ratio (assuming organic C)
        cn_ratio = soc / total_n

        if cn_ratio < 8:
            interpretation = "Very low (unusual, check measurements)"
        elif 8 <= cn_ratio < 15:
            interpretation = "Agricultural/grassland soil"
        elif 15 <= cn_ratio < 25:
            interpretation = "Forest/woodland soil"
        elif 25 <= cn_ratio < 40:
            interpretation = "Peatland/highly organic soil"
        else:
            interpretation = "Very high (unusual, check measurements)"

        return cn_ratio, interpretation

    def validate_prediction(
        self,
        predictions: Dict[str, float],
        tolerance: Dict[str, float] = None
    ) -> Dict[str, any]:
        """
        Validate predictions against chemical constraints.

        Args:
            predictions: Dictionary of predicted values
                         {'SOC': 2.5, 'clay': 25, 'CEC': 15, ...}
            tolerance: Dictionary of tolerance values for each constraint
                      {'cec': 5.0, 'texture': 2.0, ...}

        Returns:
            Dictionary with validation results:
                {'valid': True/False, 'warnings': [...], 'errors': [...]}

        Example:
            >>> constraints = ChemicalConstraints()
            >>> preds = {'SOC': 2.5, 'clay': 25, 'sand': 40, 'silt': 33, 'CEC': 15}
            >>> result = constraints.validate_prediction(preds)
            >>> if not result['valid']:
            >>>     for warning in result['warnings']:
            >>>         print(f"Warning: {warning}")
        """
        if tolerance is None:
            tolerance = {
                'cec': 5.0,        # CEC tolerance in cmol/kg
                'texture': 2.0,    # Texture sum tolerance in %
            }

        warnings = []
        errors = []

        # Check CEC consistency
        if all(k in predictions for k in ['SOC', 'clay', 'CEC']):
            silt = predictions.get('silt', None)
            expected_cec = self.cec_constraint(
                soc=predictions['SOC'],
                clay=predictions['clay'],
                silt=silt
            )
            cec_diff = abs(predictions['CEC'] - expected_cec)

            if cec_diff > tolerance['cec']:
                warnings.append(
                    f"CEC ({predictions['CEC']:.1f} cmol/kg) differs from expected "
                    f"value ({expected_cec:.1f} cmol/kg) based on SOC and clay. "
                    f"Difference: {cec_diff:.1f} cmol/kg"
                )

        # Check texture sum
        if all(k in predictions for k in ['clay', 'silt', 'sand']):
            residual = self.texture_constraint(
                clay=predictions['clay'],
                silt=predictions['silt'],
                sand=predictions['sand']
            )

            if abs(residual) > tolerance['texture']:
                errors.append(
                    f"Texture fractions sum to {100 + residual:.1f}%, not 100%. "
                    f"Clay + silt + sand should equal 100%."
                )

        # Check SOC-clay relationship
        if all(k in predictions for k in ['SOC', 'clay']):
            if not self.soc_clay_correlation(
                soc=predictions['SOC'],
                clay=predictions['clay']
            ):
                warnings.append(
                    f"SOC ({predictions['SOC']:.2f}%) appears unusual for clay "
                    f"content ({predictions['clay']:.1f}%). "
                    f"SOC:clay ratio = {predictions['SOC']/predictions['clay']:.3f}"
                )

        # Check C:N ratio if both available
        if all(k in predictions for k in ['SOC', 'total_N']):
            cn_ratio, interpretation = self.nitrogen_carbon_ratio(
                total_n=predictions['total_N'],
                soc=predictions['SOC']
            )

            if "unusual" in interpretation.lower():
                warnings.append(
                    f"C:N ratio ({cn_ratio:.1f}) is {interpretation}"
                )

        # Check physical bounds
        for prop in ['SOC', 'clay', 'silt', 'sand']:
            if prop in predictions:
                value = predictions[prop]
                if value < 0:
                    errors.append(f"{prop} cannot be negative ({value:.2f})")
                if value > 100 and prop != 'CEC':  # CEC can exceed 100
                    errors.append(f"{prop} cannot exceed 100% ({value:.2f})")

        if 'pH' in predictions:
            ph = predictions['pH']
            if ph < 3 or ph > 11:
                warnings.append(
                    f"pH ({ph:.1f}) is outside typical soil range (3-11)"
                )

        return {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'num_warnings': len(warnings),
            'num_errors': len(errors)
        }

    def suggest_constraints_for_training(
        self,
        property_name: str
    ) -> Dict[str, any]:
        """
        Suggest constraint parameters for training models.

        Args:
            property_name: Name of property to predict
                          ('SOC', 'clay', 'CEC', etc.)

        Returns:
            Dictionary with suggested constraints and weights

        Example:
            >>> constraints = ChemicalConstraints()
            >>> soc_constraints = constraints.suggest_constraints_for_training('SOC')
            >>> print(soc_constraints['related_properties'])
            ['clay', 'total_N', 'CEC']
        """
        constraints_map = {
            'SOC': {
                'related_properties': ['clay', 'total_N', 'CEC'],
                'typical_range': (0.1, 15.0),
                'units': '%',
                'constraint_weight': 0.1,
                'description': 'SOC typically 0.1-15%, correlates with clay and N'
            },
            'clay': {
                'related_properties': ['silt', 'sand', 'CEC', 'SOC'],
                'typical_range': (0, 100),
                'units': '%',
                'constraint_weight': 0.2,  # Higher weight for texture sum
                'description': 'Clay 0-100%, must satisfy texture sum = 100%'
            },
            'CEC': {
                'related_properties': ['SOC', 'clay', 'silt'],
                'typical_range': (1, 60),
                'units': 'cmol/kg',
                'constraint_weight': 0.1,
                'description': 'CEC depends on SOC and clay content'
            },
            'pH': {
                'related_properties': ['SOC', 'clay'],
                'typical_range': (3.5, 9.5),
                'units': '',
                'constraint_weight': 0.05,
                'description': 'Soil pH buffered by SOC and clay'
            },
        }

        return constraints_map.get(property_name, {
            'related_properties': [],
            'typical_range': None,
            'units': '',
            'constraint_weight': 0.0,
            'description': 'No specific constraints defined'
        })
