"""
Help dialog for displaying context-sensitive help information in the MFE Toolbox GUI.

This dialog provides information and documentation for various UI components and model features,
supporting rich text content with hyperlinks to related topics.
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser, QPushButton  # PyQt6 version 6.6.1
from PyQt6.QtGui import QIcon, QDesktopServices  # PyQt6 version 6.6.1
from PyQt6.QtCore import Qt, QUrl  # PyQt6 version 6.6.1

# Internal imports
from web.utils.qt_helpers import create_widget
from web.styles.theme import initialize_theme

# Path to help icon
HELP_ICON_PATH = "src/web/assets/icons/help.png"

class HelpDialog(QDialog):
    """
    Modal dialog providing context-sensitive help information.
    
    This dialog displays help content for a specific topic, supporting rich text formatting 
    and hyperlinks to related topics or external documentation.
    """
    
    def __init__(self, parent=None, help_topic="general"):
        """
        Initializes the Help dialog with specified topic.
        
        Args:
            parent: Parent widget
            help_topic: The help topic to display
        """
        super().__init__(parent)
        
        # Set dialog properties
        self.setWindowTitle("Help")
        self.setMinimumSize(500, 400)
        self.setModal(True)
        
        # Store current help topic
        self._current_topic = help_topic
        
        # Create UI components
        self._create_ui()
        
        # Load help content
        self._load_help_content()
        
        # Apply theme
        initialize_theme()
    
    def _create_ui(self):
        """
        Creates and configures dialog UI components.
        """
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create title with help icon
        title_layout = QHBoxLayout()
        
        # Create icon label
        icon_label = create_widget("QLabel", {
            "objectName": "helpIconLabel"
        })
        
        try:
            help_icon = QIcon(HELP_ICON_PATH)
            if not help_icon.isNull():
                icon_label.setPixmap(help_icon.pixmap(24, 24))
        except Exception as e:
            print(f"Could not load help icon: {str(e)}")
        
        # Create title label
        self._title_label = create_widget("QLabel", {
            "text": "Help Topic: " + self._current_topic.title(),
            "objectName": "helpTitle"
        })
        
        # Add icon and title to layout
        title_layout.addWidget(icon_label)
        title_layout.addWidget(self._title_label, 1)  # 1 for stretch factor
        title_layout.addStretch()
        
        main_layout.addLayout(title_layout)
        
        # Create text browser for content
        self._content_browser = create_widget("QTextBrowser", {
            "openExternalLinks": False,
            "objectName": "helpContent"
        })
        self._content_browser.anchorClicked.connect(self._on_link_clicked)
        main_layout.addWidget(self._content_browser)
        
        # Create close button
        self._close_button = create_widget("QPushButton", {
            "text": "Close",
            "objectName": "helpCloseButton"
        })
        self._close_button.clicked.connect(self.accept)
        main_layout.addWidget(self._close_button, alignment=Qt.AlignmentFlag.AlignRight)
    
    def _load_help_content(self):
        """
        Loads and displays help content for current topic.
        """
        try:
            # Get help content based on current topic
            content = self._get_help_text_for_topic(self._current_topic)
            
            # Set content in browser with HTML styling
            styled_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 10px; }}
                    h1 {{ color: #337ab7; font-size: 18px; }}
                    h2 {{ color: #5bc0de; font-size: 16px; }}
                    p {{ font-size: 14px; }}
                    a {{ color: #337ab7; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    .note {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #5bc0de; }}
                </style>
            </head>
            <body>
                {content}
            </body>
            </html>
            """
            
            self._content_browser.setHtml(styled_content)
        except Exception as e:
            error_message = f"""
            <html>
            <body>
                <h1>Error Loading Help Content</h1>
                <p>Could not load help content for topic: {self._current_topic}</p>
                <p>Error: {str(e)}</p>
                <p><a href="topic:general">Return to general help</a></p>
            </body>
            </html>
            """
            self._content_browser.setHtml(error_message)
    
    def _get_help_text_for_topic(self, topic):
        """
        Returns the help text for a specific topic.
        
        Args:
            topic: The help topic
            
        Returns:
            str: HTML-formatted help content
        """
        # Dictionary of help content by topic
        help_topics = {
            "general": """
                <h1>MFE Toolbox Help</h1>
                <p>Welcome to the MFE Toolbox, a comprehensive suite of Python modules designed for modeling 
                financial time series and conducting advanced econometric analyses.</p>
                
                <h2>Getting Started</h2>
                <p>To begin using the toolbox, select a model type from the main interface and configure
                its parameters. For more detailed information, see the following topics:</p>
                <ul>
                    <li><a href="topic:arma_models">ARMA/ARMAX Models</a></li>
                    <li><a href="topic:garch_models">GARCH Models</a></li>
                    <li><a href="topic:realized_volatility">Realized Volatility</a></li>
                </ul>
                
                <div class="note">
                <p><strong>Note:</strong> For additional documentation, please visit the 
                <a href="https://external.example.com/docs">online documentation</a>.</p>
                </div>
            """,
            
            "arma_models": """
                <h1>ARMA/ARMAX Models</h1>
                <p>The ARMA (AutoRegressive Moving Average) model combines autoregressive (AR) 
                and moving average (MA) components to model time series data.</p>
                
                <h2>Parameters</h2>
                <ul>
                    <li><strong>AR Order (p):</strong> Number of autoregressive terms</li>
                    <li><strong>MA Order (q):</strong> Number of moving average terms</li>
                    <li><strong>Include Constant:</strong> Whether to include a constant term</li>
                    <li><strong>Exogenous Variables:</strong> Optional external regressors</li>
                </ul>
                
                <h2>Related Topics</h2>
                <ul>
                    <li><a href="topic:parameter_estimation">Parameter Estimation</a></li>
                    <li><a href="topic:forecasting">Forecasting</a></li>
                    <li><a href="topic:general">Back to General Help</a></li>
                </ul>
            """,
            
            "garch_models": """
                <h1>GARCH Models</h1>
                <p>GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models are used 
                to analyze volatility clustering in financial time series.</p>
                
                <h2>Model Variants</h2>
                <ul>
                    <li><strong>GARCH:</strong> Standard GARCH model</li>
                    <li><strong>EGARCH:</strong> Exponential GARCH for asymmetric effects</li>
                    <li><strong>GJR-GARCH:</strong> Glosten-Jagannathan-Runkle GARCH</li>
                    <li><strong>APARCH:</strong> Asymmetric Power ARCH</li>
                </ul>
                
                <h2>Related Topics</h2>
                <ul>
                    <li><a href="topic:volatility_forecasting">Volatility Forecasting</a></li>
                    <li><a href="topic:general">Back to General Help</a></li>
                </ul>
            """,
            
            "realized_volatility": """
                <h1>Realized Volatility</h1>
                <p>Realized volatility measures use high-frequency data to estimate the volatility 
                of an asset over a specific time period.</p>
                
                <h2>Available Estimators</h2>
                <ul>
                    <li><strong>Realized Variance:</strong> Sum of squared returns</li>
                    <li><strong>Realized Kernel:</strong> Noise-robust estimator</li>
                    <li><strong>Bipower Variation:</strong> Robust to jumps</li>
                </ul>
                
                <h2>Related Topics</h2>
                <ul>
                    <li><a href="topic:sampling_schemes">Sampling Schemes</a></li>
                    <li><a href="topic:general">Back to General Help</a></li>
                </ul>
            """,
            
            "parameter_estimation": """
                <h1>Parameter Estimation</h1>
                <p>The MFE Toolbox uses various optimization methods to estimate model parameters:</p>
                
                <h2>Estimation Methods</h2>
                <ul>
                    <li><strong>Maximum Likelihood:</strong> Optimizes the likelihood function</li>
                    <li><strong>Quasi-Maximum Likelihood:</strong> Robust to distributional assumptions</li>
                    <li><strong>Generalized Method of Moments:</strong> Uses moment conditions</li>
                </ul>
                
                <h2>Related Topics</h2>
                <ul>
                    <li><a href="topic:arma_models">Back to ARMA Models</a></li>
                    <li><a href="topic:general">Back to General Help</a></li>
                </ul>
            """,
            
            "forecasting": """
                <h1>Forecasting</h1>
                <p>Time series forecasting predicts future values based on previously observed values.</p>
                
                <h2>Forecasting Options</h2>
                <ul>
                    <li><strong>Point Forecasts:</strong> Single value predictions</li>
                    <li><strong>Interval Forecasts:</strong> Confidence intervals</li>
                    <li><strong>Density Forecasts:</strong> Full predictive distribution</li>
                </ul>
                
                <h2>Related Topics</h2>
                <ul>
                    <li><a href="topic:arma_models">Back to ARMA Models</a></li>
                    <li><a href="topic:general">Back to General Help</a></li>
                </ul>
            """,
            
            "volatility_forecasting": """
                <h1>Volatility Forecasting</h1>
                <p>Volatility forecasting predicts future volatility based on GARCH-type models.</p>
                
                <h2>Forecast Horizons</h2>
                <ul>
                    <li><strong>One-step-ahead:</strong> Next period forecast</li>
                    <li><strong>Multi-step:</strong> Multiple periods ahead</li>
                    <li><strong>Term Structure:</strong> Entire forecast path</li>
                </ul>
                
                <h2>Related Topics</h2>
                <ul>
                    <li><a href="topic:garch_models">Back to GARCH Models</a></li>
                    <li><a href="topic:general">Back to General Help</a></li>
                </ul>
            """,
            
            "sampling_schemes": """
                <h1>Sampling Schemes</h1>
                <p>Different sampling schemes for high-frequency data analysis:</p>
                
                <h2>Available Schemes</h2>
                <ul>
                    <li><strong>Calendar Time:</strong> Fixed time intervals</li>
                    <li><strong>Tick Time:</strong> Every transaction</li>
                    <li><strong>Business Time:</strong> Volume-based sampling</li>
                </ul>
                
                <h2>Related Topics</h2>
                <ul>
                    <li><a href="topic:realized_volatility">Back to Realized Volatility</a></li>
                    <li><a href="topic:general">Back to General Help</a></li>
                </ul>
            """
        }
        
        # Return the help content for the specified topic or general help if not found
        return help_topics.get(topic, help_topics["general"])
    
    def _on_link_clicked(self, url):
        """
        Handles clicks on hyperlinks in help content.
        
        Args:
            url: The clicked URL
        """
        try:
            # Get the URL as a string
            url_str = url.toString()
            
            # Check if it's an internal topic link
            if url_str.startswith("topic:"):
                # Extract topic name
                topic = url_str.split(":", 1)[1]
                
                # Update current topic
                self._current_topic = topic
                
                # Update title
                self._title_label.setText("Help Topic: " + self._current_topic.title())
                
                # Load new content
                self._load_help_content()
            else:
                # For external links, use the system's default browser
                QDesktopServices.openUrl(url)
        except Exception as e:
            # Display error in help content
            error_message = f"""
            <html>
            <body>
                <h1>Error Navigating to Topic</h1>
                <p>Could not navigate to the requested link: {url.toString()}</p>
                <p>Error: {str(e)}</p>
            </body>
            </html>
            """
            self._content_browser.setHtml(error_message)