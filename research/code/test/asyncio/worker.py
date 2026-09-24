# worker.py
import sys
import json

class Calculator:
    def power_list(self, base: int, exponents: list[int]) -> list[int]:
        """Raises the base to each exponent in the collection."""
        return [base ** exp for exp in exponents]

if __name__ == "__main__":
    # Read arguments passed from the command line
    # sys.argv[1] is the base, sys.argv[2] is a JSON string of exponents
    base_arg = int(sys.argv[1])
    exponents_arg = json.loads(sys.argv[2])
    
    # Initialize the class and run the function
    calc = Calculator()
    result_collection = calc.power_list(base_arg, exponents_arg)
    
    # Print the result as JSON to stdout so the parent process can read it
    print(json.dumps(result_collection))
