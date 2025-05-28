# src/agriculsight/tools/soil_analysis.py

from typing import Optional, Dict, Any, List, AsyncIterator

from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field
import numpy as np

from aiq.builder.function_info import FunctionInfo
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function


class SoilAnalysisToolConfig(FunctionBaseConfig, name="soil_analysis_tool"):
    """
    Configuration for the soil analysis tool.
    """
    api_key: Optional[str] = Field(None, description="API key for soil data service")
    data_dir: Optional[str] = Field(None, description="Directory with local soil data")


class SoilAnalysisArgs(BaseModel):
    """Schema for soil analysis arguments."""
    location: Optional[str] = Field(None, description="Location or field identifier to analyze")
    analysis_type: str = Field(
        "moisture",
        description="Type of analysis: 'moisture', 'nutrients', 'ph', 'comprehensive'"
    )
    date_range: Optional[str] = Field(None, description="Date range for historical analysis")


@register_function(config_type=SoilAnalysisToolConfig)
async def soil_analysis_tool(config: SoilAnalysisToolConfig, builder: Builder):
    """Register the soil analysis tool."""

    async def _analyze_soil(
            location: str = Field(..., description="Location or field identifier to analyze"),
            analysis_type: str = Field(
                "moisture",
                description="Type of analysis: 'moisture', 'nutrients', 'ph', 'comprehensive'"
            ),
            date_range: Optional[str] = Field(None, description="Date range for historical analysis"),
    ) -> Dict[str, Any]:
        """
        Analyze soil conditions for agricultural decision-making.

        Args:
            location: Field location or identifier
            analysis_type: Type of soil analysis to perform
            date_range: Optional date range for historical analysis

        Returns:
            Dictionary with soil analysis results
        """
        # Validate analysis_type
        valid_types = ["moisture", "nutrients", "ph", "comprehensive"]
        if not isinstance(analysis_type, str) or analysis_type not in valid_types:
            return {
                "success": False,
                "error": "Invalid analysis type",
                "message": f"Analysis type must be one of {valid_types}"
            }

        # For demo purposes, return mock data
        return generate_mock_soil_data(location, analysis_type, date_range)

    yield FunctionInfo.from_fn(
        _analyze_soil,
        description="Analyze soil conditions for agricultural decision-making",
        input_schema=SoilAnalysisArgs
    )


def generate_mock_soil_data(location: str, analysis_type: str, date_range: Optional[str] = None) -> Dict[str, Any]:
    """Generate mock soil data for demonstration purposes"""

    if analysis_type == "moisture":
        # Generate mock moisture data with some variability
        base_moisture = 0.25  # 25% moisture content
        moisture_std = 0.05  # Standard deviation for variation

        # Generate mock grid data (5x5 grid)
        grid_size = 5
        moisture_grid = np.random.normal(base_moisture, moisture_std, size=(grid_size, grid_size))
        moisture_grid = np.clip(moisture_grid, 0.05, 0.4)  # Clip to realistic range

        # Format grid data for JSON
        grid_data = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid_data.append({
                    "position": {"row": i, "col": j},
                    "value": round(float(moisture_grid[i, j]), 3),
                    "status": get_moisture_status(moisture_grid[i, j])
                })

        avg_moisture = float(np.mean(moisture_grid))
        return {
            "success": True,
            "analysis_type": "moisture",
            "location": location,
            "date_range": date_range or "current",
            "average_moisture": round(avg_moisture, 3),
            "overall_status": get_moisture_status(avg_moisture),
            "grid_data": grid_data,
            "recommendation": generate_moisture_recommendation(avg_moisture)
        }

    elif analysis_type == "nutrients":
        return {
            "success": True,
            "analysis_type": "nutrients",
            "location": location,
            "date_range": date_range or "current",
            "nutrient_levels": {
                "nitrogen": {
                    "value": 24.5,  # ppm
                    "status": "low"
                },
                "phosphorus": {
                    "value": 35.2,  # ppm
                    "status": "adequate"
                },
                "potassium": {
                    "value": 180.3,  # ppm
                    "status": "high"
                },
                "calcium": {
                    "value": 1520.0,  # ppm
                    "status": "adequate"
                },
                "magnesium": {
                    "value": 230.5,  # ppm
                    "status": "adequate"
                },
                "sulfur": {
                    "value": 15.2,  # ppm
                    "status": "low"
                }
            },
            "organic_matter": {
                "value": 3.2,  # percentage
                "status": "adequate"
            },
            "recommendation": "Consider applying nitrogen and sulfur fertilizers. The soil shows adequate levels of phosphorus, potassium, calcium, and magnesium. No additional application needed for these nutrients. Organic matter content is satisfactory."
        }

    elif analysis_type == "ph":
        ph_value = 6.3
        return {
            "success": True,
            "analysis_type": "ph",
            "location": location,
            "date_range": date_range or "current",
            "ph": {
                "value": ph_value,
                "status": get_ph_status(ph_value)
            },
            "buffer_ph": 6.7,
            "recommendation": generate_ph_recommendation(ph_value)
        }

    elif analysis_type == "comprehensive":
        return {
            "success": True,
            "analysis_type": "comprehensive",
            "location": location,
            "date_range": date_range or "current",
            "soil_type": "Clay Loam",
            "soil_health_index": 75,  # 0-100 scale
            "moisture": {
                "value": 0.28,  # 28% moisture content
                "status": "optimal"
            },
            "nutrient_levels": {
                "nitrogen": {
                    "value": 24.5,  # ppm
                    "status": "low"
                },
                "phosphorus": {
                    "value": 35.2,  # ppm
                    "status": "adequate"
                },
                "potassium": {
                    "value": 180.3,  # ppm
                    "status": "high"
                }
            },
            "ph": {
                "value": 6.3,
                "status": "slightly acidic"
            },
            "organic_matter": {
                "value": 3.2,  # percentage
                "status": "adequate"
            },
            "cation_exchange_capacity": 15.3,  # meq/100g
            "base_saturation": 68,  # percentage
            "compaction": {
                "value": "moderate",
                "depth": "15-30 cm"
            },
            "recommendations": [
                "Apply nitrogen fertilizer at a rate of 30-40 lbs/acre to address deficiency.",
                "No lime application needed as pH is within acceptable range for most crops.",
                "Consider deep tillage in areas with moderate compaction.",
                "Moisture levels are optimal - no irrigation needed at this time."
            ]
        }
    else:
        return {
            "success": False,
            "error": "Invalid analysis type",
            "message": f"Analysis type '{analysis_type}' is not supported"
        }


def get_moisture_status(moisture_level: float) -> str:
    """Determine moisture status category based on value"""
    if moisture_level < 0.1:
        return "critically low"
    elif moisture_level < 0.2:
        return "low"
    elif moisture_level < 0.3:
        return "optimal"
    elif moisture_level < 0.35:
        return "high"
    else:
        return "excessive"


def get_ph_status(ph_level: float) -> str:
    """Determine pH status category based on value"""
    if ph_level < 5.5:
        return "strongly acidic"
    elif ph_level < 6.0:
        return "moderately acidic"
    elif ph_level < 6.5:
        return "slightly acidic"
    elif ph_level < 7.5:
        return "neutral"
    elif ph_level < 8.0:
        return "slightly alkaline"
    elif ph_level < 8.5:
        return "moderately alkaline"
    else:
        return "strongly alkaline"


def generate_moisture_recommendation(moisture_level: float) -> str:
    """Generate recommendation based on moisture level"""
    if moisture_level < 0.1:
        return "Critical irrigation needed immediately. Soil moisture is severely depleted."
    elif moisture_level < 0.2:
        return "Irrigation recommended in the next 24-48 hours. Soil moisture is below optimal levels."
    elif moisture_level < 0.3:
        return "Soil moisture is at optimal levels. No irrigation needed at this time."
    elif moisture_level < 0.35:
        return "Soil moisture is high. Delay irrigation until moisture levels decrease."
    else:
        return "Excessive soil moisture detected. Monitor for potential drainage issues or root diseases."


def generate_ph_recommendation(ph_level: float) -> str:
    """Generate recommendation based on pH level"""
    if ph_level < 5.5:
        return "Soil is strongly acidic. Apply limestone to raise pH to appropriate levels for most crops (6.0-7.0)."
    elif ph_level < 6.0:
        return "Soil is moderately acidic. Consider applying limestone if growing crops that prefer neutral pH."
    elif ph_level < 6.5:
        return "Soil is slightly acidic, which is suitable for most crops. No amendment needed unless growing specific plants requiring higher pH."
    elif ph_level < 7.5:
        return "Soil pH is neutral, optimal for most crops. No amendment needed."
    elif ph_level < 8.0:
        return "Soil is slightly alkaline. For acid-loving crops, consider amendments like elemental sulfur or acidic organic matter."
    else:
        return "Soil is moderately to strongly alkaline. Apply acidifying amendments like elemental sulfur if growing crops that prefer lower pH."