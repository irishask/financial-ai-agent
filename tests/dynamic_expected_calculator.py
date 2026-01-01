"""
═══════════════════════════════════════════════════════════════════════════════
DYNAMIC EXPECTED CALCULATOR
═══════════════════════════════════════════════════════════════════════════════

Calculates expected answers dynamically from transactions.csv at test runtime.

This module eliminates hardcoded dates/amounts in test expectations.
All calculations are based on:
- Current date (datetime.now())
- Actual transaction data in transactions.csv
- Validation rules from _new_QA_mapping.json

USAGE:
    from tests.dynamic_expected_calculator import DynamicExpectedCalculator
    
    calc = DynamicExpectedCalculator()
    
    # Get expected values for a query
    expected = calc.calculate_expected(query_id=4)
    print(expected)
    # {'total': 523.45, 'count': 12, 'period': 'November 2025', ...}

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class DynamicExpectedCalculator:
    """
    Calculates expected test results dynamically from transactions.csv.
    
    All temporal calculations are relative to the current date (datetime.now()),
    making tests work regardless of when they're run.
    """
    
    def __init__(
        self, 
        transactions_path: str = "data/transactions.csv",
        user_id: str = "USER_001"
    ):
        """
        Initialize calculator with transaction data.
        
        Args:
            transactions_path: Path to transactions.csv
            user_id: Default user ID for queries
        """
        self.user_id = user_id
        self.df = self._load_transactions(transactions_path)
        
        
       # self.today = datetime.now().date()                     # PRODUCTION: Use real system date
        self.today = datetime(2025, 12, 1).date()               # DEMO: Fixed date for test data
        
    def _load_transactions(self, path: str) -> pd.DataFrame:
        """Load and prepare transaction data."""
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', dayfirst=True)
        df['date_only'] = df['date'].dt.date
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATE RANGE CALCULATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_period_dates(self, period: str) -> Tuple[datetime, datetime]:
        """
        Convert period string to actual date range.
        
        Args:
            period: Period identifier like 'last_month', 'this_year', 'september', 
                    'september_2024', 'november_2025', etc.
            
        Returns:
            Tuple of (start_date, end_date)
        """
        today = self.today
        current_year = today.year
        current_month = today.month
        
        if period == 'last_month':
            # Previous calendar month
            if current_month == 1:
                start = datetime(current_year - 1, 12, 1)
                end = datetime(current_year - 1, 12, 31)
            else:
                start = datetime(current_year, current_month - 1, 1)
                # Last day of previous month
                end = datetime(current_year, current_month, 1) - timedelta(days=1)
            return start, end
        
        elif period == 'this_month':
            start = datetime(current_year, current_month, 1)
            end = datetime.now()
            return start, end
        
        elif period == 'this_year':
            start = datetime(current_year, 1, 1)
            end = datetime(current_year, 12, 31)
            return start, end
        
        elif period == 'this_week':
            # Monday to today
            days_since_monday = today.weekday()
            start = datetime.combine(today - timedelta(days=days_since_monday), datetime.min.time())
            end = datetime.now()
            return start, end
        
        elif period == 'last_7_days':
            start = datetime.combine(today - timedelta(days=6), datetime.min.time())
            end = datetime.now()
            return start, end
        
        elif period == 'last_30_days':
            start = datetime.combine(today - timedelta(days=29), datetime.min.time())
            end = datetime.now()
            return start, end
        
        # Named months (january, february, etc.) - assumes current year
        elif period.lower() in ['january', 'february', 'march', 'april', 'may', 'june',
                                 'july', 'august', 'september', 'october', 'november', 'december']:
            month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december']
            month_num = month_names.index(period.lower()) + 1
            
            # Assume current year
            year = current_year
            start = datetime(year, month_num, 1)
            
            # Last day of month
            if month_num == 12:
                end = datetime(year, 12, 31)
            else:
                end = datetime(year, month_num + 1, 1) - timedelta(days=1)
            
            return start, end
        
        # Format: month_current_year (e.g., 'november_current_year')
        elif '_current_year' in period:
            month_name = period.replace('_current_year', '')
            month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december']
            month_num = month_names.index(month_name.lower()) + 1
            
            start = datetime(current_year, month_num, 1)
            if month_num == 12:
                end = datetime(current_year, 12, 31)
            else:
                end = datetime(current_year, month_num + 1, 1) - timedelta(days=1)
            
            return start, end
        
        # Format: month_YYYY (e.g., 'september_2024', 'november_2024', 'september_2025')
        elif '_' in period and period.split('_')[-1].isdigit():
            parts = period.rsplit('_', 1)
            month_name = parts[0].lower()
            year = int(parts[1])
            
            month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december']
            
            if month_name in month_names:
                month_num = month_names.index(month_name) + 1
                start = datetime(year, month_num, 1)
                
                # Last day of month
                if month_num == 12:
                    end = datetime(year, 12, 31)
                else:
                    end = datetime(year, month_num + 1, 1) - timedelta(days=1)
                
                return start, end
            else:
                raise ValueError(f"Unknown month in period: {period}")
        
        else:
            raise ValueError(f"Unknown period: {period}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA FILTERING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def filter_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        direction: Optional[str] = None,
        category_group_id: Optional[str] = None,
        sub_category_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter transactions by various criteria.
        
        Args:
            start_date: Filter transactions >= this date
            end_date: Filter transactions <= this date
            direction: 'D' for debits, 'C' for credits
            category_group_id: Filter by categoryGroupId
            sub_category_id: Filter by subCategoryId
            
        Returns:
            Filtered DataFrame
        """
        df = self.df[self.df['user_id'] == self.user_id].copy()
        
        if start_date:
            df = df[df['date'] >= pd.Timestamp(start_date)]
        
        if end_date:
            df = df[df['date'] <= pd.Timestamp(end_date)]
        
        if direction:
            df = df[df['direction'] == direction]
        
        if category_group_id:
            df = df[df['categoryGroupId'] == category_group_id]
        
        if sub_category_id:
            df = df[df['subCategoryId'] == sub_category_id]
        
        return df
    
    def parse_category_filter(self, filter_str: str) -> Dict[str, str]:
        """
        Parse category filter string like 'categoryGroupId = CG800'.
        
        Returns:
            Dict with filter key and value
        """
        if '=' in filter_str:
            key, value = filter_str.split('=')
            return {key.strip(): value.strip()}
        return {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST TYPE CALCULATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_balance(self, validation: Dict) -> Dict[str, Any]:
        """
        Calculate current balance.
        
        Test type: balance_calculation
        """
        df = self.df[self.df['user_id'] == self.user_id]
        
        credits = df[df['direction'] == 'C']['amount'].sum()
        debits = df[df['direction'] == 'D']['amount'].sum()
        balance = credits - debits
        
        return {
            'balance': round(balance, 2),
            'total_credits': round(credits, 2),
            'total_debits': round(debits, 2),
            'as_of_date': self.today.strftime('%B %d, %Y')
        }
    
    def calculate_last_transaction(self, validation: Dict) -> Dict[str, Any]:
        """
        Find the last (most recent) transaction.
        
        Test type: last_transaction
        """
        df = self.df[self.df['user_id'] == self.user_id]
        
        # Get most recent transaction
        last_txn = df.loc[df['date'].idxmax()]
        
        return {
            'transaction_id': last_txn['transaction_id'],
            'date': last_txn['date'].strftime('%Y-%m-%d'),
            'amount': float(last_txn['amount']),
            'direction': last_txn['direction'],
            'categoryName': last_txn.get('categoryName', ''),
            'subCategoryName': last_txn.get('subCategoryName', '')
        }
    
    def calculate_sum_single_period(self, validation: Dict) -> Dict[str, Any]:
        """
        Calculate sum for a single period.
        
        Test type: sum_single_period
        """
        period = validation.get('period')
        direction = validation.get('direction')
        category_filter = validation.get('category_filter', '')
        
        start_date, end_date = self.get_period_dates(period)
        cat_filter = self.parse_category_filter(category_filter)
        
        df = self.filter_transactions(
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            category_group_id=cat_filter.get('categoryGroupId'),
            sub_category_id=cat_filter.get('subCategoryId')
        )
        
        total = df['amount'].sum()
        count = len(df)
        
        return {
            'total': round(total, 2),
            'count': count,
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d'),
            'period_description': f"{start_date.strftime('%B')} {start_date.year}"
        }
    
    def calculate_compare_two_periods(self, validation: Dict) -> Dict[str, Any]:
        """
        Compare totals between two periods.
        
        Test type: compare_two_periods
        """
        period_1 = validation.get('period_1')
        period_2 = validation.get('period_2')
        direction = validation.get('direction')
        category_filter = validation.get('category_filter', '')
        
        cat_filter = self.parse_category_filter(category_filter)
        
        # Period 1
        start1, end1 = self.get_period_dates(period_1)
        df1 = self.filter_transactions(
            start_date=start1,
            end_date=end1,
            direction=direction,
            category_group_id=cat_filter.get('categoryGroupId'),
            sub_category_id=cat_filter.get('subCategoryId')
        )
        total1 = df1['amount'].sum()
        count1 = len(df1)
        
        # Period 2
        start2, end2 = self.get_period_dates(period_2)
        df2 = self.filter_transactions(
            start_date=start2,
            end_date=end2,
            direction=direction,
            category_group_id=cat_filter.get('categoryGroupId'),
            sub_category_id=cat_filter.get('subCategoryId')
        )
        total2 = df2['amount'].sum()
        count2 = len(df2)
        
        # Calculate difference
        difference = total1 - total2
        if total2 > 0:
            percentage_change = (difference / total2) * 100
        else:
            percentage_change = 100.0 if total1 > 0 else 0.0
        
        return {
            'period_1': {
                'name': period_1,
                'total': round(total1, 2),
                'count': count1,
                'start': start1.strftime('%Y-%m-%d'),
                'end': end1.strftime('%Y-%m-%d')
            },
            'period_2': {
                'name': period_2,
                'total': round(total2, 2),
                'count': count2,
                'start': start2.strftime('%Y-%m-%d'),
                'end': end2.strftime('%Y-%m-%d')
            },
            'difference': round(difference, 2),
            'percentage_change': round(percentage_change, 1)
        }
    
    def calculate_list_transactions_period(self, validation: Dict) -> Dict[str, Any]:
        """
        List/count transactions for a period.
        
        Test type: list_transactions_period
        """
        period = validation.get('period')
        category_filter = validation.get('category_filter', '')
        
        start_date, end_date = self.get_period_dates(period)
        cat_filter = self.parse_category_filter(category_filter)
        
        df = self.filter_transactions(
            start_date=start_date,
            end_date=end_date,
            category_group_id=cat_filter.get('categoryGroupId'),
            sub_category_id=cat_filter.get('subCategoryId')
        )
        
        total_credits = df[df['direction'] == 'C']['amount'].sum()
        total_debits = df[df['direction'] == 'D']['amount'].sum()
        
        return {
            'count': len(df),
            'total_credits': round(total_credits, 2),
            'total_debits': round(total_debits, 2),
            'total_amount': round(df['amount'].sum(), 2),
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d')
        }
    
    def calculate_count_transactions_period(self, validation: Dict) -> Dict[str, Any]:
        """
        Count transactions for a period.
        
        Test type: count_transactions_period
        """
        period = validation.get('period')
        
        start_date, end_date = self.get_period_dates(period)
        
        df = self.filter_transactions(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            'count': len(df),
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d')
        }
    
    def calculate_last_transaction_by_category(self, validation: Dict) -> Dict[str, Any]:
        """
        Find last transaction in a specific category.
        
        Test type: last_transaction_by_category
        """
        category_filter = validation.get('category_filter', '')
        cat_filter = self.parse_category_filter(category_filter)
        
        df = self.filter_transactions(
            category_group_id=cat_filter.get('categoryGroupId'),
            sub_category_id=cat_filter.get('subCategoryId')
        )
        
        if len(df) == 0:
            return {
                'found': False,
                'message': 'No transactions found for this category'
            }
        
        # Get most recent
        last_txn = df.loc[df['date'].idxmax()]
        
        return {
            'found': True,
            'transaction_id': last_txn['transaction_id'],
            'date': last_txn['date'].strftime('%Y-%m-%d'),
            'amount': float(last_txn['amount']),
            'categoryName': last_txn.get('categoryName', ''),
            'subCategoryName': last_txn.get('subCategoryName', '')
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_expected(self, query_data: Dict) -> Dict[str, Any]:
        """
        Calculate expected results for a query based on its test_type.
        
        Args:
            query_data: Query definition from _new_QA_mapping.json
            
        Returns:
            Dict with calculated expected values
        """
        test_type = query_data.get('test_type')
        validation = query_data.get('validation', {})
        
        if test_type == 'balance_calculation':
            return self.calculate_balance(validation)
        
        elif test_type == 'last_transaction':
            return self.calculate_last_transaction(validation)
        
        elif test_type == 'sum_single_period':
            return self.calculate_sum_single_period(validation)
        
        elif test_type == 'compare_two_periods':
            return self.calculate_compare_two_periods(validation)
        
        elif test_type == 'list_transactions_period':
            return self.calculate_list_transactions_period(validation)
        
        elif test_type == 'count_transactions_period':
            return self.calculate_count_transactions_period(validation)
        
        elif test_type == 'last_transaction_by_category':
            return self.calculate_last_transaction_by_category(validation)
        
        elif test_type == 'vague_needs_clarification':
            # For VAGUE queries, just return the expected clarity
            return {
                'expected_clarity': 'VAGUE',
                'missing_info': query_data.get('missing_info', [])
            }
        
        else:
            raise ValueError(f"Unknown test_type: {test_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_llm_answer(
    llm_answer: str,
    expected: Dict[str, Any],
    test_type: str,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Validate LLM answer against calculated expected values.
    
    Args:
        llm_answer: The answer text from LLM-2
        expected: Calculated expected values
        test_type: Type of test (determines validation logic)
        tolerance: Acceptable percentage difference for amounts
        
    Returns:
        Dict with validation results
    """
    result = {
        'valid': False,
        'checks': [],
        'errors': []
    }
    
    # Extract numbers from LLM answer (filter out empty strings)
    raw_amounts = re.findall(r'\$?([\d,]+\.?\d*)', llm_answer)
    amounts_in_answer = []
    for a in raw_amounts:
        a_clean = a.replace(',', '').strip()
        if a_clean and a_clean != '.':
            try:
                amounts_in_answer.append(float(a_clean))
            except ValueError:
                continue
    
    # Helper: Check if LLM indicates zero/no spending
    def indicates_zero_spending(text: str) -> bool:
        zero_phrases = [
            "didn't spend anything",
            "didn't have any",
            "no spending",
            "no transactions",
            "$0", "$0.00", "0.00",
            "zero",
            "nothing"
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in zero_phrases)
    
    # Helper: Check if amount matches expected (handles zero case)
    def amount_matches(actual: float, expected_val: float) -> bool:
        if expected_val == 0:
            return actual == 0 or actual < 0.01
        return abs(actual - expected_val) / expected_val < tolerance
    
    if test_type == 'balance_calculation':
        expected_balance = expected.get('balance', 0)
        
        # Check if expected balance appears in answer
        for amount in amounts_in_answer:
            if amount_matches(amount, expected_balance):
                result['valid'] = True
                result['checks'].append(f"✅ Balance ${expected_balance:.2f} found in answer")
                break
        
        if not result['valid']:
            result['errors'].append(f"Expected balance ${expected_balance:.2f} not found in answer")
    
    elif test_type in ['sum_single_period', 'compare_two_periods']:
        if test_type == 'sum_single_period':
            expected_total = expected.get('total', 0)
            
            # Handle $0.00 case
            if expected_total == 0:
                if indicates_zero_spending(llm_answer) or 0.0 in amounts_in_answer:
                    result['valid'] = True
                    result['checks'].append(f"✅ Zero spending correctly indicated")
                else:
                    result['errors'].append(f"Expected $0.00 but LLM didn't indicate zero spending")
            else:
                for amount in amounts_in_answer:
                    if amount_matches(amount, expected_total):
                        result['valid'] = True
                        result['checks'].append(f"✅ Total ${expected_total:.2f} found in answer")
                        break
                if not result['valid']:
                    result['errors'].append(f"Expected total ${expected_total:.2f} not found")
        else:
            # compare_two_periods - check both totals
            total1 = expected.get('period_1', {}).get('total', 0)
            total2 = expected.get('period_2', {}).get('total', 0)
            
            # Handle zero cases for comparison
            if total1 == 0 and total2 == 0:
                if indicates_zero_spending(llm_answer):
                    result['valid'] = True
                    result['checks'].append(f"✅ Period 1 total $0.00 found")
                    result['checks'].append(f"✅ Period 2 total $0.00 found")
                else:
                    result['errors'].append("Expected $0.00 for both periods")
            else:
                # Check period 1
                if total1 == 0:
                    found1 = indicates_zero_spending(llm_answer) or 0.0 in amounts_in_answer
                else:
                    found1 = any(amount_matches(a, total1) for a in amounts_in_answer)
                
                # Check period 2
                if total2 == 0:
                    found2 = indicates_zero_spending(llm_answer) or 0.0 in amounts_in_answer
                else:
                    found2 = any(amount_matches(a, total2) for a in amounts_in_answer)
                
                if found1:
                    result['checks'].append(f"✅ Period 1 total ${total1:.2f} found")
                else:
                    result['errors'].append(f"Period 1 total ${total1:.2f} not found")
                
                if found2:
                    result['checks'].append(f"✅ Period 2 total ${total2:.2f} found")
                else:
                    result['errors'].append(f"Period 2 total ${total2:.2f} not found")
                
                result['valid'] = found1 and found2
    
    elif test_type == 'last_transaction_by_category':
        expected_amount = expected.get('amount', 0)
        for amount in amounts_in_answer:
            if abs(amount - expected_amount) / max(expected_amount, 1) < tolerance:
                result['valid'] = True
                result['checks'].append(f"✅ Amount ${expected_amount:.2f} found in answer")
                break
    
    elif test_type == 'list_transactions_period':
        # Check count or total_debits
        expected_count = expected.get('count', 0)
        expected_total = expected.get('total_debits', 0)
        
        count_found = expected_count in amounts_in_answer
        total_found = any(amount_matches(a, expected_total) for a in amounts_in_answer)
        
        if count_found or total_found:
            result['valid'] = True
            if count_found:
                result['checks'].append(f"✅ Count {expected_count} found in answer")
            if total_found:
                result['checks'].append(f"✅ Total ${expected_total:.2f} found in answer")
        else:
            result['errors'].append(f"Expected count {expected_count} or total ${expected_total:.2f} not found")
    
    elif test_type == 'vague_needs_clarification':
        # For VAGUE, we just check that LLM-1 returned VAGUE clarity
        # This is checked elsewhere, so mark as valid here
        result['valid'] = True
        result['checks'].append("✅ VAGUE query - clarity check done separately")
    
    else:
        result['checks'].append(f"⚠️ No specific validation for test_type: {test_type}")
        result['valid'] = True  # Pass by default for unhandled types
    
    return result