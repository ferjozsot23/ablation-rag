"""
Tool: calculate

Safely evaluates mathematical expressions.
Supports basic operations, growth rates, ratios, etc.
"""

import re
from typing import Dict


class Calculate:
    """
    Tool for safe mathematical calculations.

    Only digits, operators, parentheses and whitespace are accepted before the
    expression is evaluated.
    """

    def __init__(self):
        """Initializes the calculator."""
        # Allowed operators
        self.allowed_operators = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else float('inf'),
            '**': lambda a, b: a ** b,
            '%': lambda a, b: a % b if b != 0 else float('inf')
        }

    def calculate(self, expression: str) -> Dict:
        """
        Evaluates a mathematical expression.

        Args:
            expression: mathematical expression (e.g. "100 * 1.5")

        Returns:
            Dict with the result and metadata
        """
        try:
            expr = expression.strip()

            # Only numbers, operators and whitespace are allowed
            if not re.match(r'^[\d\s\+\-\*\/\%\(\)\.]+ *$', expr):
                return {
                    'expression': expression,
                    'result': None,
                    'error': 'Expression contains disallowed characters'
                }

            # eval is safe here: the input was validated above and builtins are disabled
            result = eval(expr, {"__builtins__": {}}, {})

            return {
                'expression': expression,
                'result': float(result),
                'error': None
            }

        except ZeroDivisionError:
            return {
                'expression': expression,
                'result': None,
                'error': 'Division by zero'
            }
        except Exception as e:
            return {
                'expression': expression,
                'result': None,
                'error': str(e)
            }

    def growth_rate(self, old_value: float, new_value: float) -> Dict:
        """
        Computes the growth rate.

        Args:
            old_value: previous value
            new_value: new value

        Returns:
            Dict with the result
        """
        if old_value == 0:
            return {
                'old_value': old_value,
                'new_value': new_value,
                'growth_rate': None,
                'growth_pct': None,
                'error': 'Previous value is zero'
            }

        growth = (new_value - old_value) / old_value
        growth_pct = growth * 100

        return {
            'old_value': old_value,
            'new_value': new_value,
            'growth_rate': growth,
            'growth_pct': growth_pct,
            'error': None
        }

    def ratio(self, numerator: float, denominator: float) -> Dict:
        """
        Computes a ratio.

        Args:
            numerator: numerator
            denominator: denominator

        Returns:
            Dict with the result
        """
        if denominator == 0:
            return {
                'numerator': numerator,
                'denominator': denominator,
                'ratio': None,
                'error': 'Denominator is zero'
            }

        return {
            'numerator': numerator,
            'denominator': denominator,
            'ratio': numerator / denominator,
            'error': None
        }

    def difference(self, value1: float, value2: float) -> Dict:
        """
        Computes the absolute difference.

        Args:
            value1: first value
            value2: second value

        Returns:
            Dict with the result
        """
        return {
            'value1': value1,
            'value2': value2,
            'difference': value1 - value2,
            'absolute_difference': abs(value1 - value2),
            'error': None
        }

    def format_result(self, result: Dict) -> str:
        """
        Formats a result for the agent.

        Args:
            result: calculation result

        Returns:
            Formatted string
        """
        if result.get('error'):
            return f"Error: {result['error']}"

        output = []

        # Simple calculation
        if 'expression' in result:
            output.append(f"Expression: {result['expression']}")
            output.append(f"Result: {result['result']}")

        # Growth rate
        elif 'growth_rate' in result:
            output.append(f"Previous value: {result['old_value']:,.2f}")
            output.append(f"New value: {result['new_value']:,.2f}")
            output.append(f"Growth: {result['growth_pct']:.2f}%")

        # Ratio
        elif 'ratio' in result:
            output.append(f"{result['numerator']:,.2f} / {result['denominator']:,.2f}")
            output.append(f"Ratio: {result['ratio']:.4f}")

        # Difference
        elif 'difference' in result:
            output.append(f"{result['value1']:,.2f} - {result['value2']:,.2f}")
            output.append(f"Difference: {result['difference']:,.2f}")

        return "\n".join(output)


# Singleton
_calc_tool = None


def get_calc_tool() -> Calculate:
    """Returns the singleton instance."""
    global _calc_tool

    if _calc_tool is None:
        _calc_tool = Calculate()

    return _calc_tool


def calculate(expression: str) -> Dict:
    """
    Convenience function for calculations.

    Args:
        expression: mathematical expression

    Returns:
        Result
    """
    tool = get_calc_tool()
    return tool.calculate(expression)


def main():
    """Tool demo (run with `python -m src.tools.calculate`)."""
    print("\n" + "="*60)
    print("TOOL: calculate")
    print("="*60)

    tool = get_calc_tool()

    # Test 1: Simple expression
    print("\n1. Simple expression: '100 * 1.5 + 20'")
    result = tool.calculate("100 * 1.5 + 20")
    print(tool.format_result(result))

    # Test 2: Growth rate
    print("\n2. Growth rate: 365.8B → 383.3B")
    result = tool.growth_rate(365.8e9, 383.3e9)
    print(tool.format_result(result))

    # Test 3: Ratio
    print("\n3. Ratio: 383.3B / 365.8B")
    result = tool.ratio(383.3e9, 365.8e9)
    print(tool.format_result(result))

    # Test 4: Difference
    print("\n4. Difference: 383.3B - 365.8B")
    result = tool.difference(383.3e9, 365.8e9)
    print(tool.format_result(result))

    print("\n" + "="*60)
    print("Demo completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
