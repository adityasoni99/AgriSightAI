import logging
import os
import re
from typing import Optional, AsyncIterator

import PIL
from PIL import Image
from agno.memory import AgentMemory
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.component_ref import LLMRef
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AgriculsightFunctionConfig(FunctionBaseConfig, name="agriculsight"):
    """
    AgriSight multi-agent system for precision agriculture.
    """
    # Add your custom configuration parameters here
    llm_name: LLMRef
    image_processing_tool: Optional[str] = Field(None, description="Name of the image processing tool to use")
    soil_analysis_tool: Optional[str] = Field(None, description="Name of the soil analysis tool to use")
    weather_api_tool: Optional[str] = Field(None, description="Name of the weather API tool to use")
    api_key: Optional[str] = Field(None, description="NVIDIA API Key for NIM services")


@register_function(config_type=AgriculsightFunctionConfig, framework_wrappers=[LLMFrameworkEnum.AGNO])
async def agriculsight_function(
        config: AgriculsightFunctionConfig, builder: Builder
) -> AsyncIterator[FunctionInfo]:
    from textwrap import dedent

    from agno.agent import Agent

    # Get API key from config or environment
    if not config.api_key:
        config.api_key = os.getenv("NVIDIA_API_KEY")

    if not config.api_key:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    # Get LLM from builder
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AGNO)

    # Initialize tools
    tools = []

    # Add image processing tool if specified
    if config.image_processing_tool:
        image_tool = builder.get_tool(fn_name=config.image_processing_tool, wrapper_type=LLMFrameworkEnum.AGNO)
        tools.append(image_tool)

    # Add soil analysis tool if specified
    if config.soil_analysis_tool:
        soil_tool = builder.get_tool(fn_name=config.soil_analysis_tool, wrapper_type=LLMFrameworkEnum.AGNO)
        tools.append(soil_tool)

    # Add weather API tool if specified
    if config.weather_api_tool:
        weather_tool = builder.get_tool(fn_name=config.weather_api_tool, wrapper_type=LLMFrameworkEnum.AGNO)
        tools.append(weather_tool)

    # Define agent memories
    shared_memory = AgentMemory()

    # Create the vision agent for image analysis
    vision_agent = Agent(
        name="VisionAgent",
        role="Analyzes crop imagery to detect health issues, diseases, and growth patterns",
        model=llm,
        memory=shared_memory,
        description=dedent("""\
                   You are an expert agricultural vision analyst. Given crop imagery, you can identify crop health 
                   issues, detect diseases, assess growth stages, and provide detailed analysis of visual patterns
                   that indicate problems or opportunities in agricultural fields.

                   When analyzing images, you should extract any image paths from the user's query and pass them 
                   to the image_processing_tool. If no image path is found, you can still use the tool without 
                   providing an image_url parameter, and it will return analysis based on available data
                   """),
        instructions=[
            "Analyze crop imagery to identify signs of nutrient deficiencies, water stress, or disease",
            "Assess crop growth stages based on visual characteristics",
            "Identify areas of a field that may require special attention",
            "Provide confidence scores for detected issues",
            "Suggest potential causes for observed problems",
            "When using image_processing_tool, extract image paths from user queries when available",
            "If an image path is mentioned, pass it as the image_url parameter to image_processing_tool to perform most relevant task as per user's query out 'health_assessment', 'disease_detection', 'weed_identification'"
            "**The _process_image function of image_processing_tool requires a specific format: _process_image(image_path='image path value', task='health_assessment')**"
            "Use must this format precisely."
        ],
        tools=[t for t in tools if 'image' in t.name.lower()],
        add_datetime_to_instructions=True,
    )

    # Create the Environmental Agent for sensor data analysis
    env_agent = Agent(
        name="EnvironmentalAgent",
        role="Processes sensor data to understand soil conditions and environmental factors",
        model=llm,
        memory=shared_memory,
        description=dedent("""\
                You are an expert in agricultural environmental analysis. Given sensor data about soil conditions,
                weather patterns, and other environmental factors, you can identify issues, trends, and opportunities
                for optimizing crop growth and resource usage.
                """),
        instructions=[
            "Analyze soil moisture, temperature, and pH data to identify optimal conditions",
            "Correlate weather patterns with crop performance",
            "Identify areas with suboptimal environmental conditions",
            "Suggest adjustments to irrigation or fertilization based on environmental data",
            "Predict potential environmental risks based on current conditions and forecasts",
            "Use soil_processing_tool to perform most relevant analysis as per user's query out of 'moisture', 'ph', 'nutrients', 'comprehensive'",
            "**The _analyze_soil function of soil_analysis_tool requires a specific format: _analyze_soil(location='location of field', analysis_type='moisture')**"
            "Use must this format precisely."
        ],
        tools=[t for t in tools if 'soil' in t.name.lower() or 'weather' in t.name.lower()],
        add_datetime_to_instructions=True,
    )

    # Create the Research Agent for knowledge retrieval
    research_agent = Agent(
        name="ResearchAgent",
        role="Retrieves relevant agricultural knowledge and best practices",
        model=llm,
        description=dedent("""\
               You are an agricultural research specialist with extensive knowledge of crop science, pest management,
               soil health, and sustainable farming practices. You can provide evidence-based recommendations and
               best practices for various agricultural scenarios.
               """),
        instructions=[
            "Provide research-backed information on crop management",
            "Suggest best practices for addressing identified issues",
            "Explain the science behind agricultural phenomena",
            "Offer context-specific recommendations based on crop type, region, and growth stage",
            "Cite relevant research or established agricultural knowledge when possible",
        ],
        add_datetime_to_instructions=True,
    )

    # Create the Resource Agent for optimization
    resource_agent = Agent(
        name="ResourceAgent",
        role="Optimizes resource allocation for irrigation, fertilization, and other interventions",
        model=llm,
        description=dedent("""\
            You are an agricultural resource optimization specialist. Given information about field conditions,
            crop needs, and available resources, you can create efficient plans for irrigation, fertilization,
            and other interventions that maximize yield while minimizing resource usage.
            """),
        instructions=[
            "Create optimized irrigation schedules based on crop needs and soil conditions",
            "Develop targeted fertilization plans that minimize waste",
            "Prioritize interventions based on potential impact and resource constraints",
            "Calculate resource savings compared to standard practices",
            "Balance short-term needs with long-term sustainability",
        ],
        add_datetime_to_instructions=True,
    )

    # Create the strategy agent that coordinates
    strategy_agent = Agent(
        name="StrategyAgent",
        role="Synthesize insights from various data sources to provide actionable recommendations.",
        model=llm,
        memory=shared_memory,
        description=dedent("""\
                You are an agricultural strategy expert. Your role is to create a plan and strategy
                to answer a user question. You can synthesize insights from
                vision analysis and environmental data to provide actionable recommendations.
                Consider both immediate interventions and long-term strategies for sustainable farming.
                Prioritize recommendations based on urgency and potential impact.
                """),
        instructions=[
            "Provide:",
            "1. Overall crop health assessment (score from 0-1)",
            "2. Prioritized list of issues detected",
            "3. Soil condition summary",
            "4. Integrated recommendations with priority levels (high, medium, low)",
        ],
        tools=tools,
        add_datetime_to_instructions=True,
    )

    async def _arun(inputs: str) -> str:
        """
        State your concerns. we will make the most appropriate strategy
        Args:
            inputs : user query
        """
        try:
            # Extract image path if present in the input
            image_path = None
            path_match = re.search(r"['\"]([^'\"]+\.(jpg|jpeg|png|gif|bmp))['\"]", inputs)
            if path_match:
                potential_path = path_match.group(1)
                logger.info(f"Extracted potential image path from input: {potential_path}")
                image_path = potential_path

            # Create a team of agents
            team = [vision_agent, env_agent, strategy_agent]

            # Process the query - first parse the request
            parsed_query = await strategy_agent.arun(
                f"Parse the following agricultural query to understand what the farmer needs: {inputs}\n\n"
                "Extract key information such as:\n"
                "1. What crop(s) are mentioned?\n"
                "2. What specific problems or concerns are mentioned?\n"
                "3. What type of guidance is being requested?\n"
                "4. Are there any time constraints or urgency mentioned?\n\n"
                "Format as a concise summary.", stream=False)

            vision_input = f"""
            Based on this request: '{parsed_query}', what visual analysis would you perform? 
            If there are images available, analyze them for crop health issues.
            Original query: {inputs}

            Identify any visible signs of:
            1. Nutrient deficiencies
            2. Water stress
            3. Disease or pest damage
            4. Growth abnormalities

            Provide confidence scores for each detected issue and suggest potential causes.
            """

            # If we extracted an image path, add it explicitly to the vision input
            if image_path:
                vision_input += f"\n\nIMPORTANT: Use the image_processing_tool with this IMAGE PATH: {image_path}"

            # Have vision agent analyze any imagery mentioned
            vision_result = await vision_agent.arun(vision_input, stream=False)
            vision_content = vision_result.content if hasattr(vision_result, 'content') else str(vision_result)

            env_input = f"""
            Based on this request: '{parsed_query}' and the output of the vision analysis agent: '{vision_content}',
            what environmental factors should be analyzed?
            Consider soil moisture, nutrient levels, weather conditions, and other relevant environmental data.
            Identify any environmental issues, assess soil health, and suggest optimal conditions for the given crop(s).
            Original query: {inputs}
            """

            # Have environmental agent analyze conditions
            env_task = env_agent.arun(env_input, stream=False)

            research_input = f"""
            Based on this request: '{parsed_query}', what research  would you perform?
            Provide relevant agricultural knowledge for the given crop(s).

            Include information on:
            1. Optimal growing conditions for this stage
            2. Common issues at this growth stage
            3. Best practices for the given management
            4. Research-backed recommendations for maximizing yield

            Focus on practical, actionable information that would be useful for a farmer.
            Original query: {inputs}
            """

            # Run research to get relevant knowledge
            research_task = research_agent.arun(research_input)

            # Wait for all tasks to complete
            import asyncio
            env_result, research_result = await asyncio.gather(
                env_task, research_task
            )

            # Extract content from RunResponse objects if needed
            env_content = env_result.content if hasattr(env_result, 'content') else str(env_result)
            research_content = research_result.content if hasattr(research_result, 'content') else str(research_result)

            # Combine results for integrated analysis
            integration_input = f"""
            Based on this request: '{parsed_query}', synthesize a detailed and integrated analysis using the outputs
            from these agents:

            VISION ANALYSIS:
            {vision_content}

            ENVIRONMENTAL ANALYSIS:
            {env_content}

            RESEARCH KNOWLEDGE:
            {research_content}

            Your response must include:

            1. Overall Crop Health Assessment
                - Provide a numerical health score between 0 (poor) and 1 (excellent).
                - Briefly explain how this score was derived based on the data.

            2. Prioritized List of Detected Issues
                - Identify and rank issues (e.g., disease signs, pest presence, nutrient deficiencies).
                - Include severity, location (if available), and likely cause.

            3. Soil Condition Summary
                - Describe key soil parameters (e.g., moisture, pH, nutrient levels).
                - Highlight any abnormalities or areas of concern.

            4. Integrated Recommendations
                - Deliver actionable recommendations categorized by priority: High, Medium, or Low.
                - Ensure suggestions are practical, specific, and based on a synthesis of all inputs.

            Format:
            Organize the response as a clearly structured report suitable for presentation to a farmer.
            Use headers, bullet points, and concise language to improve readability and usability in the field.

            Original query: {inputs}
            """

            # Have strategy agent synthesize insights and provide recommendations
            strategy_result = await strategy_agent.arun(
                integration_input, stream=False
            )

            # Extract content from RunResponse
            strategy_content = (strategy_result.content
                                if hasattr(strategy_result, 'content') else str(strategy_result))

            logger.info("response from agricultural_sight: \n %s", strategy_content)

            return strategy_content

        except Exception as e:
            logger.error(f"Error in agriculsight function: {str(e)}")
            return f"Sorry, I encountered an error while generating your agricultural plan: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun, description="Generate agricultural insights and recommendations based on multi-agent analysis")
