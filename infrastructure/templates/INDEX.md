"""
MFE Toolbox Index Template

This file serves as a template for generating the standardized index 
for the MFE Toolbox documentation. It provides alphabetical listings 
of functions, classes, types, and other key elements with cross-references.

The MFE (MATLAB Financial Econometrics) Toolbox has been reimplemented 
in Python 3.12, incorporating modern programming constructs such as 
async/await patterns and strict type hints.

The index template can be processed programmatically to create 
the final index document.
"""

import yaml
from typing import Dict, List, Optional, Union, Any
import os
import re
from pathlib import Path

# Import glossary formatting from GLOSSARY.md
try:
    from GLOSSARY import glossary_format
except ImportError:
    # Default glossary format if import fails
    glossary_format = {
        "statistical_term": "{term}: {definition}",
        "technical_term": "{term}: {definition}",
        "acronym": "{acronym}: {expansion}"
    }

# Global dictionaries
index_sections = {
    "functions": {},   # Dictionary mapping function names to metadata
    "classes": {},     # Dictionary mapping class names to metadata
    "types": {},       # Dictionary mapping type names to metadata
    "modules": {},     # Dictionary mapping module names to metadata
    "constants": {}    # Dictionary mapping constant names to metadata
}

index_formats = {
    "function": "**{name}**({params}) -> {return_type}: {short_description} [{module}]",
    "class": "**{name}**: {short_description} [{module}]",
    "type": "**{name}**: {description} [{module}]",
    "module": "**{name}**: {description}",
    "constant": "**{name}**: {type} - {description} [{module}]"
}

section_formats = {
    "functions": "## Functions\n\nAlphabetical list of functions in the MFE Toolbox:\n\n{content}\n",
    "classes": "## Classes\n\nAlphabetical list of classes in the MFE Toolbox:\n\n{content}\n",
    "types": "## Types\n\nAlphabetical list of custom types in the MFE Toolbox:\n\n{content}\n",
    "modules": "## Modules\n\nAlphabetical list of modules in the MFE Toolbox:\n\n{content}\n",
    "constants": "## Constants\n\nAlphabetical list of constants in the MFE Toolbox:\n\n{content}\n"
}

section_links = {
    "core": "#core-statistical-modules",
    "models": "#time-series--volatility-modules",
    "utils": "#utility-modules",
    "ui": "#user-interface-modules",
    "functions": "#functions",
    "classes": "#classes",
    "types": "#types"
}

def load_index_metadata(yaml_path: str) -> Dict[str, Any]:
    """
    Loads and validates index metadata from YAML format.
    
    Parameters:
        yaml_path: Path to the YAML file containing index metadata
        
    Returns:
        Parsed index metadata dictionary
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            
        # Validate required sections
        required_sections = ["functions", "classes", "types"]
        for section in required_sections:
            if section not in data:
                data[section] = {}
                
        return data
    except Exception as e:
        print(f"Error loading index metadata: {str(e)}")
        return {"functions": {}, "classes": {}, "types": {}}

def format_index_entry(entry_info: Dict[str, Any]) -> str:
    """
    Formats a single index entry according to its type.
    
    Parameters:
        entry_info: Dictionary containing entry details
        
    Returns:
        Formatted index entry string
    """
    entry_type = entry_info.get("type", "function")
    
    if entry_type not in index_formats:
        return f"**{entry_info.get('name', 'Unknown')}**: Missing format for type '{entry_type}'"
    
    # Apply appropriate template
    try:
        formatted_entry = index_formats[entry_type].format(**entry_info)
        return formatted_entry
    except KeyError as e:
        return f"**{entry_info.get('name', 'Unknown')}**: Missing required field {str(e)}"

def generate_section_links() -> Dict[str, str]:
    """
    Generates cross-reference links for documentation sections.
    
    Returns:
        Dictionary of section links
    """
    links = {}
    
    # Core documentation sections
    links.update({
        "getting_started": "#getting-started",
        "installation": "#installation",
        "overview": "#overview",
        "examples": "#examples"
    })
    
    # API reference sections
    links.update({
        "core_bootstrap": "#bootstrap-module",
        "core_distributions": "#distributions-module",
        "core_tests": "#statistical-tests",
        "models_timeseries": "#time-series-models",
        "models_univariate": "#univariate-volatility-models",
        "models_multivariate": "#multivariate-volatility-models",
        "models_realized": "#high-frequency-module"
    })
    
    # User guide sections
    links.update({
        "guide_bootstrap": "#bootstrap-guide",
        "guide_garch": "#garch-modeling-guide",
        "guide_ui": "#user-interface-guide"
    })
    
    # Add standard section links
    links.update(section_links)
    
    return links

def generate_index_document(index_data: Dict[str, Any]) -> str:
    """
    Generates the complete index document.
    
    Parameters:
        index_data: Dictionary containing all index information
        
    Returns:
        Complete index document in markdown
    """
    document_parts = []
    
    # Add header
    document_parts.append("# MFE Toolbox Index\n")
    document_parts.append("## Overview\n")
    document_parts.append(
        "This document provides a comprehensive index of all components in the "
        "MFE (MATLAB Financial Econometrics) Toolbox Python implementation. "
        "The toolbox has been reimplemented in Python 3.12, incorporating modern "
        "programming constructs such as async/await patterns and strict type hints.\n"
    )
    
    # Add quick navigation
    document_parts.append("## Quick Navigation\n")
    document_parts.append("- [Functions](#functions)")
    document_parts.append("- [Classes](#classes)")
    document_parts.append("- [Types](#types)")
    document_parts.append("- [Modules](#modules)")
    document_parts.append("- [Constants](#constants)\n")
    
    # Process each section
    for section_name, section_format in section_formats.items():
        if section_name in index_data and index_data[section_name]:
            # Sort entries alphabetically
            sorted_entries = sorted(index_data[section_name].items(), key=lambda x: x[0].lower())
            
            # Format each entry
            formatted_entries = []
            for name, info in sorted_entries:
                info["name"] = name
                formatted_entries.append(f"- {format_index_entry(info)}")
            
            # Add to document
            section_content = "\n".join(formatted_entries)
            document_parts.append(section_format.format(content=section_content))
    
    # Add cross-references
    links = generate_section_links()
    document_parts.append("## Cross References\n")
    document_parts.append("For detailed information about specific components, refer to the following sections:\n")
    
    for category, sections in {
        "Core Modules": ["core_bootstrap", "core_distributions", "core_tests"],
        "Model Modules": ["models_timeseries", "models_univariate", "models_multivariate", "models_realized"],
        "User Interface": ["guide_ui"],
        "Guides": ["guide_bootstrap", "guide_garch"]
    }.items():
        document_parts.append(f"\n### {category}")
        for section in sections:
            if section in links:
                section_name = section.replace("_", " ").replace("guide ", "").title()
                document_parts.append(f"- [{section_name}]({links[section]})")
    
    # Return complete document
    return "\n".join(document_parts)

# Example usage
"""
# Load the index metadata
index_data = load_index_metadata('docs/index_metadata.yaml')

# Generate the index document
index_document = generate_index_document(index_data)

# Write to file
with open('docs/INDEX.md', 'w', encoding='utf-8') as f:
    f.write(index_document)
"""

# Index format templates for export
index_format = {
    "function": index_formats["function"],
    "class": index_formats["class"],
    "type": index_formats["type"],
    "module": index_formats["module"],
    "constant": index_formats["constant"]
}

# Section format templates for export
functions_section = section_formats["functions"]
classes_section = section_formats["classes"]
types_section = section_formats["types"]

# This file is intended to be used as a template for generating the MFE Toolbox index.
# To generate the index, you would typically:
# 1. Collect metadata about functions, classes, types, etc.
# 2. Format this metadata using the provided functions
# 3. Generate the final index document