"""
Template for generating the Code of Conduct documentation that establishes
community guidelines and expected behavior for the MFE Toolbox project.

This module provides a template class and generation function to create
a standardized Code of Conduct for the project, based on the Contributor Covenant.
"""

import logging
from typing import Dict, Optional, Any
import markdown  # version: 3.5.1

# Global constants
PROJECT_NAME = "MFE Toolbox"
TEMPLATE_VERSION = "1.0"

class CodeOfConductTemplate:
    """
    Template class for generating project Code of Conduct documentation.
    
    This class provides functionality to render a Code of Conduct document
    with project-specific information inserted into the standard template.
    """
    
    def __init__(self, template_path: str):
        """
        Initialize template with path and load structure.
        
        Parameters
        ----------
        template_path : str
            Path to the template file
        """
        self.content = ""
        self.sections = {
            "header": "# Code of Conduct\n\n",
            "intro": "## Our Pledge\n\nIn the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to make participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.\n\n",
            "standards": "## Our Standards\n\nExamples of behavior that contributes to creating a positive environment include:\n\n* Using welcoming and inclusive language\n* Being respectful of differing viewpoints and experiences\n* Gracefully accepting constructive criticism\n* Focusing on what is best for the community\n* Showing empathy towards other community members\n\nExamples of unacceptable behavior include:\n\n* The use of sexualized language or imagery and unwelcome sexual attention or advances\n* Trolling, insulting/derogatory comments, and personal or political attacks\n* Public or private harassment\n* Publishing others' private information, such as a physical or electronic address, without explicit permission\n* Other conduct which could reasonably be considered inappropriate in a professional setting\n\n",
            "responsibilities": "## Our Responsibilities\n\nProject maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.\n\nProject maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.\n\n",
            "scope": "## Scope\n\nThis Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project email address, posting via an official social media account, or acting as an appointed representative at an online or offline event. Representation of a project may be further defined and clarified by project maintainers.\n\n",
            "enforcement": "## Enforcement\n\nInstances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at [EMAIL]. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident. Further details of specific enforcement policies may be posted separately.\n\nProject maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.\n\n",
            "attribution": "## Attribution\n\nThis Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org), version 1.4, available at [https://www.contributor-covenant.org/version/1/4/code-of-conduct.html](https://www.contributor-covenant.org/version/1/4/code-of-conduct.html)\n\n"
        }
    
    def render_template(self, project_info: Dict[str, Any]) -> str:
        """
        Renders the Code of Conduct documentation with provided info.
        
        Parameters
        ----------
        project_info : dict
            Dictionary containing project details to insert into the template
        
        Returns
        -------
        str
            Rendered Code of Conduct documentation
        """
        try:
            # Create a copy of sections to avoid modifying the originals
            rendered_sections = self.sections.copy()
            
            # Replace placeholders in the header section
            rendered_sections["header"] = f"# {project_info.get('name', 'Project')} Code of Conduct\n\n"
            
            # Replace the contact email in the enforcement section
            contact_email = project_info.get('contact_email', '[EMAIL]')
            rendered_sections["enforcement"] = rendered_sections["enforcement"].replace('[EMAIL]', contact_email)
            
            # Add version information
            footer = f"\n\n_This Code of Conduct was generated from template version {TEMPLATE_VERSION} for {project_info.get('name', 'Project')}._\n"
            
            # Combine all sections
            content = ""
            for section in rendered_sections.values():
                content += section
            
            # Add footer
            content += footer
            
            # Set the content property
            self.content = content
            
            return content
            
        except Exception as e:
            logging.error(f"Error rendering Code of Conduct template: {str(e)}")
            return "Error rendering Code of Conduct template."


def generate_code_of_conduct(output_path: str, project_info: Dict[str, Any]) -> str:
    """
    Generates the Code of Conduct content from the template with project-specific information.
    
    Parameters
    ----------
    output_path : str
        Path where the Code of Conduct file should be written
    project_info : dict
        Dictionary containing project details
    
    Returns
    -------
    str
        Generated CODE_OF_CONDUCT.md content
    """
    try:
        # Create template instance
        template = CodeOfConductTemplate(output_path)
        
        # Render the template with project info
        content = template.render_template(project_info)
        
        # Write content to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logging.info(f"Code of Conduct generated successfully at {output_path}")
        return content
        
    except Exception as e:
        logging.error(f"Failed to generate Code of Conduct: {str(e)}")
        raise