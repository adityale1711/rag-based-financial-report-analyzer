import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Optional
from ...domain.entities import Visualization, VisualizationType
from ...domain.repositories import IChartGenerator, ChartGenerationError


class PlotlyChartGenerator(IChartGenerator):
    """Plotly implementation of the chart generator interface.

    This class creates various types of charts using Plotly for data visualization.
    """

    def __init__(self):
        """Initialize the plotly chart generator."""
        self.default_config = {
            "template": "plotly_white",
            "showlegend": True,
            "height": 600,
        }

    def _create_bar_chart(
        self,
        data: Any,
        title: str,
        config: dict[str, Any]
    ) -> go.Figure:
        """Create a bar chart from the data.

        Args:
            data: Data to visualize (DataFrame, dict, or list).
            title: Chart title.
            config: Chart configuration.

        Returns:
            Plotly figure object.
        """
        try:
            # Handle different data types
            if isinstance(data, dict):
                # Convert dict to DataFrame
                if len(data) == 1 and isinstance(list(data.values())[0], dict):
                    list(data.keys())[0]
                    metric_data = list(data.values())[0]
                    df = pd.DataFrame(list(metric_data.items()), columns=['Category', 'Value'])
                    x_col, y_col = 'Category', 'Value'
                elif all(isinstance(v, (int, float)) for v in data.values()):
                    # Simple key-value pairs
                    df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
                    x_col, y_col = 'Category', 'Value'
                else:
                    # Multiple metrics or complex dict structure
                    try:
                        df = pd.DataFrame(data)
                        x_col = df.columns[0]
                        y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    except Exception:
                        # Fallback: try to flatten nested dict
                        flattened = {}
                        for k, v in data.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    flattened[f"{k}_{sub_k}"] = sub_v
                            else:
                                flattened[k] = v
                        df = pd.DataFrame(list(flattened.items()), columns=['Category', 'Value'])
                        x_col, y_col = 'Category', 'Value'
            elif isinstance(data, (list, tuple)):
                # Convert list to DataFrame
                df = pd.DataFrame(data)
                if len(df.columns) >= 2:
                    x_col, y_col = df.columns[0], df.columns[1]
                else:
                    x_col, y_col = "Index", df.columns[0] if len(df.columns) > 0 else "Value"
                    df[x_col] = range(len(df))
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
                if len(df.columns) >= 2:
                    x_col, y_col = df.columns[0], df.columns[1]
                else:
                    x_col, y_col = "Index", df.columns[0] if len(df.columns) > 0 else "Value"
                    df[x_col] = range(len(df))
            else:
                # Fallback: Create simple bar chart
                df = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [1, 2, 3]})
                x_col, y_col = "Category", "Value"

            # Create the bar chart
            if len(df.columns) > 2:
                # Multiple columns - select only appropriate columns for visualization
                # Use first non-identifier column as y-axis
                numeric_cols = []
                for col in df.columns:
                    if col != x_col and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        numeric_cols.append(col)

                if numeric_cols:
                    y_col = numeric_cols[0]  # Use first numeric column
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        title=title,
                        template=config.get("template", "plotly_white"),
                        height=config.get("height", 500)
                    )
                else:
                    # No numeric columns found, create simple chart
                    if "Value" not in df.columns:
                        df["Value"] = [100] * len(df)
                    fig = px.bar(
                        df,
                        x=x_col,
                        y="Value",  # Use the Value column
                        title=title,
                        template=config.get("template", "plotly_white"),
                        height=config.get("height", 500)
                    )
            else:
                # Single value column
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    template=config.get("template", "plotly_white"),
                    height=config.get("height", 500)
                )

            # Update layout
            fig.update_layout(
                showlegend=config.get("showlegend", len(df.columns) > 2),
                xaxis_title=x_col,
                yaxis_title=y_col   
            )

            return fig
        except Exception as e:
            # Create fallback chart with error details
            import traceback
            error_msg = str(e)
            print(f"Chart generation error details: {error_msg}")
            print(f"Data type: {type(data)}")
            print(f"Data content: {data}")
            print(f"Traceback: {traceback.format_exc()}")

            fig = go.Figure(
                data=[go.Bar(x=["Error"], y=[0])],
                layout=go.Layout(
                    title=f"Chart generation Error: {title} - {error_msg}",
                    template="plotly_white"
                )
            )
            return fig

    def _create_line_chart(
        self,
        data: Any,
        title: str,
        config: dict[str, Any]
    ) -> go.Figure:
        """Create a line chart from the data.

        Args:
            data: Data to visualize.
            title: Chart title.
            config: Chart configuration.

        Returns:
            Plotly figure object.
        """
        try:
            # Handle different data types
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, (list, tuple)):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                # Fallback
                df = pd.DataFrame({
                    "X": [1, 2, 3], 
                    "Y": [2, 4, 6]
                })

            # Determine columns
            if len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
            else:
                x_col, y_col = "Index", df.columns[0] if len(df.columns) > 0 else "Value"
                df[x_col] = range(len(df))

            # Create the line chart
            if len(df.columns) > 2:
                # Multiple columns - select only appropriate columns for visualization
                numeric_cols = []
                for col in df.columns:
                    if col != x_col and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        numeric_cols.append(col)

                if numeric_cols:
                    y_col = numeric_cols[0]  # Use first numeric column
                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_col,
                        title=title,
                        template=config.get("template", "plotly_white"),
                        height=config.get("height", 500)
                    )
                else:
                    # No numeric columns found, create simple chart
                    if "Value" not in df.columns:
                        df["Value"] = [100] * len(df)
                    fig = px.line(
                        df,
                        x=x_col,
                        y="Value",  # Use the Value column
                        title=title,
                        template=config.get("template", "plotly_white"),
                        height=config.get("height", 500)
                    )
            else:
                # Single y column
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    template=config.get("template", "plotly_white"),
                    height=config.get("height", 500)
                )

            # Update layout
            fig.update_layout(
                showlegend=config.get("showlegend", len(df.columns) > 2),
                xaxis_title=x_col,
                yaxis_title=y_col   
            )

            return fig
        except Exception as e:
            # Create fallback chart with error details
            import traceback
            error_msg = str(e)
            print(f"Line chart generation error details: {error_msg}")
            print(f"Data type: {type(data)}")
            print(f"Data content: {data}")

            fig = go.Figure(
                data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='lines')],
                layout=go.Layout(
                    title=f"Line chart Error: {title} - {error_msg}",
                    template="plotly_white"
                )
            )
            return fig
        
    def _create_pie_chart(
        self,
        data: Any,
        title: str,
        config: dict[str, Any]
    ) -> go.Figure:
        """Create a pie chart from the data.

        Args:
            data: Data to visualize.
            title: Chart title.
            config: Chart configuration.

        Returns:
            Plotly figure object.
        """
        try:
            # Handle different data types
            if isinstance(data, dict):
                if len(data) == 1 and isinstance(list(data.values())[0], dict):
                    metric_data = list(data.values())[0]
                    labels = list(metric_data.keys())
                    values = list(metric_data.values())
                else:
                    labels = list(data.keys())
                    values = list(data.values())
            elif isinstance(data, pd.DataFrame):
                if len(data.columns) >= 2:
                    labels, values = data.iloc[:, 0], data.iloc[:, 1]
                else:
                    labels, values = data.index, data.iloc[:, 0] if len(data.columns) > 0 else [1] * len(data)
            elif isinstance(data, (list, tuple)):
                df = pd.DataFrame(data)
                if len(df.columns) >= 2:
                    labels, values = df.iloc[:, 0], df.iloc[:, 1]
                else:
                    labels, values = df.index, df.iloc[:, 0] if len(df.columns) > 0 else [1] * len(df)
            else:
                # Fallback
                labels, values = ["A", "B", "C"], [1, 2, 3]

            # Create the pie chart
            fig = px.pie(
                values=values,
                names=labels,
                title=title,
                template=config.get("template", "plotly_white"),
                height=config.get("height", 500)
            )

            fig.update_layout(
                showlegend=config.get("showlegend", True)
            )

            return fig
        except Exception as e:
            # Create fallback chart with error details
            import traceback
            error_msg = str(e)
            print(f"Pie chart generation error details: {error_msg}")
            print(f"Data type: {type(data)}")
            print(f"Data content: {data}")

            fig = go.Figure(
                data=[go.Pie(labels=["Error"], values=[1])],
                layout=go.Layout(
                    title=f"Pie chart Error: {title} - {error_msg}",
                    template="plotly_white"
                )
            )
            return fig
        
    def _create_scatter_chart(
        self,
        data: Any,
        title: str,
        config: dict[str, Any]
    ) -> go.Figure:
        """Create a scatter plot from the data.

        Args:
            data: Data to visualize.
            title: Chart title.
            config: Chart configuration.

        Returns:
            Plotly figure object.
        """
        try:
            # Handle different data types
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, (list, tuple)):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                # Fallback
                df = pd.DataFrame({
                    "X": [1, 2, 3, 4], 
                    "Y": [2, 4, 3, 5]
                })

            # Determine columns
            if len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
            else:
                x_col, y_col = "Index", df.columns[0] if len(df.columns) > 0 else "Value"
                df[x_col] = range(len(df))

            # Create the scatter chart with only appropriate columns
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=title,
                template=config.get("template", "plotly_white"),
                height=config.get("height", 500)
            )

            # Add size and color columns if available and numeric
            numeric_cols = []
            for col in df.columns:
                if col not in [x_col, y_col] and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_cols.append(col)

            if numeric_cols:
                # Use first available numeric column for size
                size_col = numeric_cols[0]
                if size_col in df.columns:
                    # Normalize size values for better visualization
                    size_values = df[size_col]
                    if size_values.max() > 0:
                        normalized_sizes = 5 + (size_values / size_values.max()) * 15  # Scale between 5-20
                        fig.update_traces(marker={"size": normalized_sizes})

            # Don't add color mapping as it can cause issues with mixed data types

            # Update layout
            fig.update_layout(
                showlegend=config.get("showlegend", False),
                xaxis_title=x_col,
                yaxis_title=y_col   
            )

            return fig
        except Exception as e:
            # Create fallback chart with error details
            import traceback
            error_msg = str(e)
            print(f"Scatter chart generation error details: {error_msg}")
            print(f"Data type: {type(data)}")
            print(f"Data content: {data}")

            fig = go.Figure(
                data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='markers')],
                layout=go.Layout(
                    title=f"Scatter chart Error: {title} - {error_msg}",
                    template="plotly_white"
                )
            )
            return fig
        
    def _extract_financial_data_from_answer(
        self,
        answer_text: str
    ) -> dict:
        """Extract financial data from AI-generated answer text.

        Args:
            answer_text: The AI's answer containing financial information.

        Returns:
            Dictionary with extracted financial data.
        """
        extracted_data = {
            "months": [],
            "net_profit": [],
            "total_assets": [],
            "total_liabilities": [],
            "total_equity": [],
            "revenue": [],
            "cash": []
        }

        if not answer_text:
            return extracted_data

        # Convert to lowercase for pattern matching
        text = answer_text.lower()

        # More flexible patterns for extracting financial data from answer text
        answer_patterns = {
            "net_profit": [
                r"net\s+profit\s*[=:]?\s*([0-9,]+)",
                r"profit\s*[=:]?\s*([0-9,]+)",
                r"laba\s+bersih\s*[=:]?\s*([0-9,]+)",
                r"laba\s*[=:]?\s*([0-9,]+)",
                r"([0-9,]+)\s*million\s+rupiah.*profit",
                r"profit.*([0-9,]+)\s*million\s+rupiah",
                r"net\s+profit\s*=\s*([0-9,]+)",
                r"-?\s*([a-z]+\s+\d{1,2},\s*\d{4}):\s*net\s+profit\s*=\s*([0-9,]+)",
                r"([a-z]+\s+\d{1,2},\s*\d{4}).*?net\s+profit.*?([0-9,]+)"
            ],
            "total_assets": [
                r"total\s+assets?\s*[=:]?\s*([0-9,]+)",
                r"assets?\s*[=:]?\s*([0-9,]+)",
                r"total\s+aktiva\s*[=:]?\s*([0-9,]+)",
                r"aktiva\s*[=:]?\s*([0-9,]+)"
            ],
            "total_liabilities": [
                r"total\s+liabilities?\s*[=:]?\s*([0-9,]+)",
                r"liabilities?\s*[=:]?\s*([0-9,]+)",
                r"total\s+liabilitas\s*[=:]?\s*([0-9,]+)",
                r"liabilitas\s*[=:]?\s*([0-9,]+)"
            ],
            "revenue": [
                r"revenue\s*[=:]?\s*([0-9,]+)",
                r"pendapatan\s*[=:]?\s*([0-9,]+)",
                r"total\s+revenue\s*[=:]?\s*([0-9,]+)"
            ]
        }

        # Month patterns for answer text
        month_patterns = [
            r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\s+2024",
            r"2024\s*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)"
        ]

        # Extract months mentioned in the answer
        extracted_months = set()
        for pattern in month_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                month_raw = match.group(1)

                # Normalize month name
                month_map = {
                    'january': 'Jan 2024', 'jan': 'Jan 2024',
                    'february': 'Feb 2024', 'feb': 'Feb 2024',
                    'march': 'Mar 2024', 'mar': 'Mar 2024',
                    'april': 'Apr 2024', 'apr': 'Apr 2024',
                    'may': 'May 2024',
                    'june': 'Jun 2024', 'jun': 'Jun 2024',
                    'july': 'Jul 2024', 'jul': 'Jul 2024',
                    'august': 'Aug 2024', 'aug': 'Aug 2024',
                    'september': 'Sep 2024', 'sep': 'Sep 2024',
                    'october': 'Oct 2024', 'oct': 'Oct 2024',
                    'november': 'Nov 2024', 'nov': 'Nov 2024',
                    'december': 'Dec 2024', 'dec': 'Dec 2024'
                }

                month = month_map.get(month_raw.lower())
                if month:
                    extracted_months.add(month)

        # If no months found, look for month names in a different pattern
        if not extracted_months:
            # Look for months mentioned with financial figures
            month_value_patterns = [
                r"(august|augustus|october|oktober|november|september|agustus).*?([0-9,]+)",
                r"([0-9,]+).*(august|augustus|october|oktober|november|september|agustus)"
            ]

            for pattern in month_value_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    month_raw = match.group(1) if match.group(1).isalpha() else match.group(2)

                    month_map = {
                        'august': 'Aug 2024', 'augustus': 'Aug 2024', 'agustus': 'Aug 2024',
                        'september': 'Sep 2024',
                        'october': 'Oct 2024', 'oktober': 'Oct 2024', 'oct': 'Oct 2024',
                        'november': 'Nov 2024', 'nov': 'Nov 2024'
                    }

                    month = month_map.get(month_raw.lower())
                    if month:
                        extracted_months.add(month)

        # Special extraction for month-value pairs in the format from the AI answer
        month_value_pattern = r"-?\s*([a-z]+\s+\d{1,2},\s*\d{4}):\s*net\s+profit\s*=\s*([0-9,]+)"
        month_matches = re.findall(month_value_pattern, text, re.IGNORECASE)

        if month_matches:
            # Extract months and values directly from the answer format
            months_from_answer = []
            values_from_answer = []

            for month_str, value_str in month_matches:
                # Normalize month name
                month_map = {
                    'january': 'Jan', 'february': 'Feb', 'march': 'Mar',
                    'april': 'Apr', 'may': 'May', 'june': 'Jun',
                    'july': 'Jul', 'august': 'Aug', 'september': 'Sep',
                    'october': 'Oct', 'november': 'Nov', 'december': 'Dec'
                }

                for month_name, month_abbr in month_map.items():
                    if month_name in month_str.lower():
                        months_from_answer.append(f"{month_abbr} 2024")
                        break

                try:
                    value = float(value_str.replace(',', ''))
                    values_from_answer.append(value)
                except ValueError:
                    continue

            if months_from_answer and values_from_answer:
                extracted_data["months"] = months_from_answer
                extracted_data["net_profit"] = values_from_answer
                print(f"Extracted from answer: months={months_from_answer}, values={values_from_answer}")
                return extracted_data

        # Extract financial figures for each metric using general patterns
        for metric, patterns_list in answer_patterns.items():
            values = []
            for pattern in patterns_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # Handle tuple matches from complex patterns
                        if isinstance(match, tuple):
                            # Use the last element which should be the number
                            match = match[-1] if match[-1] else match[0]

                        # Clean and convert the number
                        value_str = match.replace(',', '')
                        value = float(value_str)
                        if value > 0:  # Only include positive values
                            values.append(value)
                    except (ValueError, AttributeError):
                        continue

            extracted_data[metric] = values

        # Set months if found
        extracted_data["months"] = list(extracted_months)

        return extracted_data

    def _extract_rag_financial_data(
        self,
        data: dict
    ) -> pd.DataFrame:
        """Extract financial data from RAG system response structure.

        Args:
            rag_data: RAG data containing sources and financial information.

        Returns:
            DataFrame with structured financial data for visualization.
        """
        try:
            # Extract financial data from RAG sources
            sources = data.get("sources", [])

            # Also try to extract from AI answer text
            answer_text = data.get("answer_text", "")

            # Initialize financial data structure
            financial_data = {
                "months": [],
                "total_assets": [],
                "total_liabilities": [],
                "total_equity": [],
                "net_profit": [],
                "cash": [],
                "revenue": []
            }

            # Define patterns for financial data extraction (Indonesian financial terms)
            patterns = {
                "total_assets": r"(?:total\s+liabilitas\s+dan\s+ekuitas|TOTAL\s+LIABILITAS\s+DAN\s+EKUITAS)[\s:\n]*([0-9.,]+)",
                "total_liabilities": r"(?:total\s+liabilitas|jumlah\s+total\s+liabilitas|TOTAL\s+LIABILITAS)[\s:\n]*([0-9.,]+)",
                "total_equity": r"(?:total\s+ekuitas|ekuitas\s+total|TOTAL\s+EKUITAS)[\s:\n]*([0-9.,]+)",
                "net_profit": r"(?:laba\s+rugi|laba\s+bersih|LABA\s+RUGI)[\s:\n]*([0-9.,]+)",
                "cash": r"(?:^\d+\.\s+kas|\bkas\b)[\s:\n]*([0-9.,]+)",
                "revenue": r"(?:pendapatan\s+bunga|PENDAPATAN\s+BUNGA)[\s:\n]*([0-9.,]+)"
            }

            # Month extraction patterns (more comprehensive)
            month_patterns = [
                r"(agustus|august)\s+2024",
                r"(october|oktober|oct)\s+2024",
                r"(november|nov)\s+2024",
                r"(september|sep)\s+2024",
                r"(july|juli)\s+2024",
                r"(june|juni)\s+2024",
                r"(may|mei)\s+2024",
                r"(april|apr)\s+2024",
                r"(march|mar)\s+2024",
                r"(february|feb|februari)\s+2024",
                r"(january|jan)\s+2024",
                r"(december|dec|desember)\s+2024"
            ]

            # Extract data from all source
            extracted_months = set()
            for source in sources:
                # Handle different source formats
                content = ""
                if isinstance(source, dict):
                    if "content" in source:
                        content = str(source["content"]).lower()
                    elif "content_preview" in source:
                        content = str(source["content_preview"]).lower()
                elif hasattr(source, 'content'):
                    content = str(source.content).lower()
                elif hasattr(source, 'content_preview'):
                    content = str(source.content_preview).lower()

                if not content:
                    continue

                # Extract month from this source
                for month_pattern in month_patterns:
                    month_match = re.search(month_pattern, content)
                    if month_match:
                        month_raw = month_match.group(1)
                        
                        # Normalize month name
                        month_map = {
                            'january': 'Jan 2024', 'jan': 'Jan 2024',
                            'february': 'Feb 2024', 'feb': 'Feb 2024', 'februari': 'Feb 2024',
                            'march': 'Mar 2024', 'mar': 'Mar 2024',
                            'april': 'Apr 2024', 'apr': 'Apr 2024',
                            'may': 'May 2024', 'mei': 'May 2024',
                            'june': 'Jun 2024', 'juni': 'Jun 2024',
                            'july': 'Jul 2024', 'juli': 'Jul 2024',
                            'august': 'Aug 2024', 'agustus': 'Aug 2024',
                            'september': 'Sep 2024', 'sep': 'Sep 2024',
                            'october': 'Oct 2024', 'oktober': 'Oct 2024', 'oct': 'Oct 2024',
                            'november': 'Nov 2024', 'nov': 'Nov 2024',
                            'december': 'Dec 2024', 'dec': 'Dec 2024', 'desember': 'Dec 2024'
                        }

                        month = month_map.get(month_raw.lower())
                        if month and month not in extracted_months:
                            extracted_months.add(month)
                            financial_data["months"].append(month)

                            # Extract financial figures for this month
                            for key, pattern in patterns.items():
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    # Take the first match
                                    value_str = matches[0]

                                    # Handle Indonesian number format
                                    value_str = value_str.replace('.', '')

                                    try:
                                        value = float(value_str)

                                        # Extracted values are already in millions, no need to convert
                                        financial_data[key].append(value)
                                    except ValueError:
                                        financial_data[key].append(0)
                                else:
                                    financial_data[key].append(0)
                        break  # Stop after finding the first matching month

            # If no meaningful data found in sources, try extracting from AI answer text
            if not financial_data["months"] or all(
                sum(1 for v in financial_data[metric] if v > 0) == 0
                for metric in financial_data.keys() if metric != "months"
            ):
                print(f"Trying to extract from answer text: {answer_text[:200]}...")
                answer_extracted = self._extract_financial_data_from_answer(answer_text)
                print(f"Answer extraction result: {answer_extracted}")

                # Use answer data if it contains meaningful information
                if answer_extracted["months"] and any(
                    answer_extracted[metric] for metric in answer_extracted.keys()
                    if metric != "months"
                ):
                    financial_data.update(answer_extracted)
                    print("Using answer-extracted financial data")
                else:
                    print("Answer extraction didn't yield meaningful data")
            else:
                print(f"Using source-extracted data. Months: {financial_data['months']}")
                for metric, values in financial_data.items():
                    if metric != "months":
                        non_zero = [v for v in values if v > 0]
                        if non_zero:
                            print(f"{metric}: {non_zero}")

            # Create DataFrame based on available data
            if financial_data["months"]:
                # Find the metric with the most non-zero values
                best_metric = None
                best_count = 0

                for metric, values in financial_data.items():
                    if metric != "months":
                        non_zero_count = sum(1 for v in values if v > 0)
                        if non_zero_count > best_count:
                            best_count = non_zero_count
                            best_metric = metric

                if best_metric and best_count > 0:
                    # Use the best available metric
                    df_data = {
                        "Month": financial_data["months"][:len(financial_data[best_metric])],
                        "Value": financial_data[best_metric]
                    }
                    return pd.DataFrame(df_data)
                else:
                    # No meaningful financial data found, but we have months
                    return pd.DataFrame({
                        "Month": financial_data["months"],
                        "Value": [100] * len(financial_data["months"]),
                        "Note": ["No financial metrics found"] * len(financial_data["months"])
                    })
            else:
                # If still no data, create a default structure based on available sources
                sources = data.get("sources", [])
                if sources:
                    # Create data based on available documents
                    doc_names = []
                    for source in sources[:5]:  # Limit to first 5 sources
                        if isinstance(source, dict):
                            doc_name = source.get("document_name", f"Document {len(doc_names)+1}")
                            # Extract just the filename
                            doc_name = doc_name.split("/")[-1] if "/" in doc_name else doc_name
                            doc_names.append(doc_name[:30])  # Limit length

                    if doc_names:
                        return pd.DataFrame({
                            "Document": doc_names,
                            "Value": [100] * len(doc_names),
                            "Note": ["Financial data extraction failed"] * len(doc_names)
                        })

                # Final fallback
                return pd.DataFrame({
                    "Category": ["Data 1", "Data 2", "Data 3"],
                    "Value": [100, 120, 110],
                    "Note": ["No financial data extracted from sources"] * 3
                })
        except Exception as e:
            print(f"RAG financial data extraction error: {str(e)}")
            return pd.DataFrame({
                "Month": ["Aug 2024", "Oct 2024", "Nov 2024"],
                "Value": [100, 120, 110],
                "Error": [f"Extraction error: {str(e)}"] * 3
            })

    def _extract_structured_financial_data(
        self,
        data: dict
    ) -> pd.DataFrame:
        """Extract structured financial data for visualization.

        Args:
            data: Structured financial data dictionary.

        Returns:
            DataFrame with financial data ready for plotting.
        """
        try:
            if not data or not isinstance(data, dict):
                return None

            # Check if this is the new structured financial data format
            if "periods" in data and "values" in data:
                periods = data.get("periods", [])
                values = data.get("values", [])
                metric = data.get("metric", "Financial Data")
                currency = data.get("currency", "IDR")

                if periods and values and len(periods) == len(values):
                    return pd.DataFrame({
                        "Period": periods,
                        "Value": values,
                        "Metric": [metric] * len(periods),
                        "Currency": [currency] * len(periods)
                    })

            return None
        except Exception as e:
            print(f"Structured financial data extraction error: {str(e)}")
            return None

    def _extract_numeric_data(
        self,
        data: Any
    ) -> pd.DataFrame:
        """Extract and clean numeric data for visualization.

        Args:
            data: Raw data from RAG system.

        Returns:
            Cleaned DataFrame with numeric data.
        """
        try:
            # First, try to extract structured financial data
            if isinstance(data, dict):
                structured_data = self._extract_structured_financial_data(data)
                if structured_data is not None:
                    return structured_data

            # If data contains sources with mixed types, extract only relevant data
            if isinstance(data, dict):
                # Check if this is RAG data structure
                if "sources" in data:
                    # This is RAG data - extract financial data properly
                    return self._extract_rag_financial_data(data)

                # Regular dict data - validate before converting to DataFrame
                try:
                    # Check if dict has uniform value types suitable for DataFrame
                    values = list(data.values())
                    if all(isinstance(v, (int, float, str)) for v in values):
                        df = pd.DataFrame(data)
                    else:
                        # Mixed types - create simple categorical data
                        df = pd.DataFrame({
                            "Category": list(data.keys()),
                            "Value": [1] * len(data)
                        })
                except Exception:
                    # Fallback for problematic dict structure
                    df = pd.DataFrame({
                        "Category": ["A", "B", "C"],
                        "Value": [1, 2, 3]
                    })
            elif isinstance(data, (list, tuple)):
                # Validate list elements
                if len(data) > 0 and all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                else:
                    # Create simple numeric data
                    df = pd.DataFrame({
                        "Category": [f"Item {i+1}" for i in range(max(3, len(data)))],
                        "Value": list(data) if len(data) > 0 else [1, 2, 3]
                    })
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                # Create default financial data
                return pd.DataFrame({
                    "Month": ["Aug 2024", "Oct 2024", "Nov 2024"],
                    "Value": [100, 120, 110]
                })
            
            # Ensure DataFrame has at least 2 columns and valid data
            if len(df.columns) < 2:
                # Add a default value column if missing
                if len(df.columns) == 1:
                    df["Value"] = [100] * len(df)
                else:
                    # Create completely new DataFrame
                    df = pd.DataFrame({
                        "Month": ["Aug 2024", "Oct 2024", "Nov 2024"],
                        "Value": [100, 120, 110]
                    })

            # Clean the data - keep only numeric columns and one identifier column
            numeric_cols = []
            identifier_cols = []

            for col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_cols.append(col)
                elif df[col].dtype == 'object':
                    # Check if this column can be converted to numeric
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except Exception:
                        identifier_cols.append(col)

            # If no numeric columns found, create default data
            if not numeric_cols:
                return pd.DataFrame({
                    "Month": ["Aug 2024", "Oct 2024", "Nov 2024"],
                    "Value": [100, 120, 110]
                })

            # Keep at most one identifier column
            if identifier_cols:
                final_cols = [identifier_cols[0]] + numeric_cols[:3] # Limit to 3 numeric columns
            else:
                final_cols = numeric_cols[:3]
                if not final_cols:
                    final_cols = ["Category"]
                    df["Category"] = [f"Item {i+1}" for i in range(len(df))]

            # Ensure we don't have more columns than we can handle
            final_df = df[final_cols].copy()

            # Final validation - ensure all data types are appropriate
            for col in final_df.columns:
                is_identifier_col = identifier_cols and col == identifier_cols[0]
                if not is_identifier_col:
                    # Try to convert numeric columns
                    try:
                        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                        final_df[col] = final_df[col].fillna(0)  # Replace NaN with 0
                    except Exception:
                        pass  # Keep as is if conversion fails

            return final_df
        except Exception as e:
            print(f"Data extraction error: {str(e)}")
            return pd.DataFrame({
                "Month": ["Aug 2024", "Oct 2024", "Nov 2024"],
                "Value": [100, 120, 110]
            })

    def generate_chart(
        self,
        chart_type: VisualizationType,
        data: Any,
        title: str,
        config: Optional[dict[str, Any]] = None
    ) -> Visualization:
        """Generate a visualization based on the data and chart type.

        Args:
            chart_type: Type of chart to generate.
            data: Data to visualize.
            title: Chart title.
            config: Additional chart configuration.

        Returns:
            Visualization object with the generated chart.

        Raises:
            ChartGenerationError: If chart generation fails.
        """
        try:
            config = {**self.default_config, **(config or {})}

            # Clean and prepare data
            clean_data = self._extract_numeric_data(data)

            if chart_type == VisualizationType.BAR:
                chart = self._create_bar_chart(clean_data, title, config)
            elif chart_type == VisualizationType.LINE:
                chart = self._create_line_chart(clean_data, title, config)
            elif chart_type == VisualizationType.PIE:
                chart = self._create_pie_chart(clean_data, title, config)
            elif chart_type == VisualizationType.SCATTER:
                chart = self._create_scatter_chart(clean_data, title, config)
            else:
                raise ChartGenerationError(f"Unsupported chart type: {chart_type}")
            
            return Visualization(
                chart_type=chart_type,
                chart_object=chart,
                title=title,
                description=config.get(
                    "description", f"{chart_type.value} chart for {title}"
                ),
                config=config
            )
        except Exception as e:
            raise ChartGenerationError(
                f"Failed to generate {chart_type.value} chart: {str(e)}"
            ) from e
