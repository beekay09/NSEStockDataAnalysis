"""
analyze_opportunities.py

Generates the opportunity_report.html (and text report) by running the integrated 
stock analysis pipeline. 

This file is maintained as a dedicated entry point for generating the opportunity 
reports, as requested, so you can continue running `python analyze_opportunities.py`.
"""

import sys
import run_filters

def main():
    print("==========================================================")
    print("   Starting Opportunity Analysis & Report Generation")
    print("==========================================================")
    print("This will execute the integrated pipeline from run_filters.py")
    print("to score stocks, check filters, and generate:")
    print("  -> opportunity_report.html")
    print("  -> opportunity_report.txt")
    print("  -> filter_results.xlsx")
    print("==========================================================\n")
    
    # We delegate to the main function of run_filters which contains 
    # all the logic for scoring and HTML generation.
    try:
        run_filters.main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
