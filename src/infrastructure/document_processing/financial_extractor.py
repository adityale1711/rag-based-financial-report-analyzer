import re
from datetime import datetime
from typing import List, Optional
from ...domain.entities import FinancialDataPoint, StructuredFinancialData


class FinancialDataExtractor:
    """Utility class for extracting financial data from text content."""

    # Financial data extraction patterns (Indonesian and English terms)
    FINANCIAL_PATTERNS = {
        "total_assets": [
            # Handle table format: "TOTAL ASET 1 .400.604.114" or "TOTAL ASET 400.604.114"
            r"(?:total\s+assets|aset\s+total|total\s+aset)\s+\d+\s*\.?\s*([0-9.,]+)"
        ],
        "total_liabilities": [
            # Handle table format: "TOTAL LIABILITAS 1 .159.841.409" or "TOTAL LIABILITAS 159.841.409"
            r"(?:total\s+liabilities|kewajiban\s+total|total\s+kewajiban)\s+\d+\s*\.?\s*([0-9.,]+)"
        ],
        "total_equity": [
            # Handle table format: "TOTAL EKUITAS 2 40.762.705" or "TOTAL EKUITAS 40.762.705"
            r"(?:total\s+equity|ekuitas\s+total|total\s+ekuitas)\s+\d+\s*\.?\s*([0-9.,]+)"
        ],
        "net_profit": [
            r"(?:net\s+profit|laba\s+bersih|profit\s+bersih)\s+\d+\s*\.?\s*([0-9.,]+)"
        ],
        "cash": [
            # Handle numbered line items: "1. Kas 15.897.902" or "Kas 15.897.902"
            r"(?:\d+\.\s*)?(?:cash|kas)\s+([0-9.,]+)"
        ],
        "revenue": [
            r"(?:revenue|pendapatan|total\s+revenue|total\s+pendapatan)\s+\d+\s*\.?\s*([0-9.,]+)"
        ]
    }

    # Month/period extraction patterns
    PERIOD_PATTERNS = [
        r"(january|januari|february|februari|march|maret|april|april|may|mei|june|juni|july|juli|august|agustus|september|september|october|oktober|november|november|december|desember)\s+(\d{4})",
        r"(q[1-4]\s+\d{4})",  # Q1 2024, Q2 2024, etc.
        r"(\d{4}/\d{2})",     # 2024/08
        r"(\d{1,2}/\d{1,2}/\d{4})",  # 8/31/2024
    ]

    @classmethod
    def extract_period_from_text(cls, text: str) -> Optional[str]:
        """Extract time period from text.

        Args:
            text: Text content to search for period information.

        Returns:
            Normalized period string or None if not found.
        """
        text_lower = text.lower()

        for pattern in cls.PERIOD_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                if "q" in match.group(1).lower():
                    return match.group(1).upper()
                elif "/" in match.group(1):
                    if len(match.group(1)) == 7:  # 2024/08
                        year, month = match.group(1).split('/')
                        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        if month.isdigit() and 1 <= int(month) <= 12:
                            return f"{month_names[int(month)]} {year}"
                    elif len(match.group(1)) >= 8:  # 8/31/2024
                        parts = match.group(1).split('/')
                        if len(parts) == 3 and parts[2].isdigit():
                            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            month = int(parts[0])
                            if 1 <= month <= 12:
                                return f"{month_names[month]} {parts[2]}"
                else:
                    # Month name pattern
                    month_map = {
                        'january': 'Jan', 'januari': 'Jan', 'jan': 'Jan',
                        'february': 'Feb', 'februari': 'Feb', 'feb': 'Feb',
                        'march': 'Mar', 'maret': 'Mar', 'mar': 'Mar',
                        'april': 'Apr', 'apr': 'Apr',
                        'may': 'May', 'mei': 'May',
                        'june': 'Jun', 'juni': 'Jun', 'jun': 'Jun',
                        'july': 'Jul', 'juli': 'Jul', 'jul': 'Jul',
                        'august': 'Aug', 'agustus': 'Aug', 'aug': 'Aug',
                        'september': 'Sep', 'september': 'Sep', 'sep': 'Sep',
                        'october': 'Oct', 'oktober': 'Oct', 'oct': 'Oct',
                        'november': 'Nov', 'november': 'Nov', 'nov': 'Nov',
                        'december': 'Dec', 'desember': 'Dec', 'dec': 'Dec'
                    }
                    month_name = match.group(1)
                    year = match.group(2)
                    normalized_month = month_map.get(month_name.lower(), month_name.title())
                    return f"{normalized_month} {year}"

        # Default if no period found
        return datetime.now().strftime("%b %Y")

    @classmethod
    def extract_financial_data_from_text(cls, text: str, document_name: str) -> StructuredFinancialData:
        """Extract financial data points from text content.

        Args:
            text: Text content to extract financial data from.
            document_name: Name of the source document.

        Returns:
            StructuredFinancialData containing extracted financial data points.
        """
        data_points = []
        period = cls.extract_period_from_text(text)

        for metric_type, patterns in cls.FINANCIAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # Extract and clean the numeric value
                        value_str = match.group(1).replace(',', '').replace('.', '')
                        value = float(value_str)

                        # Extract the raw text that matched
                        raw_text = match.group(0)

                        # Create financial data point
                        data_point = FinancialDataPoint(
                            metric_type=metric_type,
                            value=value,
                            period=period,
                            currency="IDR",  # Default to IDR for Indonesian documents
                            confidence=0.9,  # High confidence for regex matches
                            raw_text=raw_text
                        )
                        data_points.append(data_point)

                    except (ValueError, IndexError) as e:
                        # Skip invalid numeric values
                        continue

        # Calculate overall confidence score
        if data_points:
            avg_confidence = sum(dp.confidence for dp in data_points) / len(data_points)
        else:
            avg_confidence = 0.0

        return StructuredFinancialData(
            data_points=data_points,
            document_name=document_name,
            extraction_method="regex",
            confidence_score=avg_confidence,
            extraction_timestamp=datetime.now().isoformat()
        )

    @classmethod
    def aggregate_financial_data_by_metric(cls, financial_data_list: List[StructuredFinancialData], metric_type: str) -> dict:
        """Aggregate financial data points by metric type across multiple chunks.

        Args:
            financial_data_list: List of StructuredFinancialData objects.
            metric_type: The metric type to aggregate (e.g., "total_assets").

        Returns:
            Dictionary with aggregated data ready for visualization.
        """
        periods = []
        values = []

        # Sort by period to maintain chronological order
        all_data_points = []
        for financial_data in financial_data_list:
            for data_point in financial_data.data_points:
                if data_point.metric_type == metric_type:
                    all_data_points.append(data_point)

        # Sort by period
        all_data_points.sort(key=lambda x: x.period)

        # Extract unique periods and corresponding values
        seen_periods = set()
        for data_point in all_data_points:
            if data_point.period not in seen_periods:
                periods.append(data_point.period)
                values.append(data_point.value)
                seen_periods.add(data_point.period)

        return {
            "periods": periods,
            "values": values,
            "metric": metric_type.replace('_', ' ').title(),
            "currency": "IDR"
        }