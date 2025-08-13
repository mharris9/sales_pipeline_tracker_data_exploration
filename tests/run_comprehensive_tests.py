"""
Comprehensive Test Runner for Phase 3
Combines all existing tests with new comprehensive tests
"""
import pytest
import sys
import os
import time
import subprocess
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test_suite(test_files, description):
    """Run a specific test suite and report results"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Run pytest on the specified test files
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show top 10 slowest tests
        "--maxfail=5",  # Stop after 5 failures
    ] + test_files
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"Test suite timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running test suite: {e}")
        return False

def run_browser_tests():
    """Run browser-based tests using browsermcp"""
    print(f"\n{'='*60}")
    print("Running Browser-based UI Tests")
    print(f"{'='*60}")
    
    # Check if Streamlit app is running
    try:
        import requests
        response = requests.get("http://localhost:8502", timeout=5)
        if response.status_code == 200:
            print("✓ Streamlit app is running on localhost:8502")
        else:
            print("✗ Streamlit app is not responding correctly")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Streamlit app: {e}")
        print("Please start the app with: streamlit run main.py --server.port 8502")
        return False
    
    # Run browser tests
    browser_test_file = "tests/test_browser_ui_validation.py"
    if os.path.exists(browser_test_file):
        return run_test_suite([browser_test_file], "Browser UI Validation Tests")
    else:
        print(f"Browser test file not found: {browser_test_file}")
        return False

def main():
    """Main test runner"""
    print("Sales Pipeline Data Explorer - Comprehensive Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test suites
    test_suites = [
        {
            "name": "Existing Core Tests",
            "files": [
                "tests/services/test_state_manager.py",
                "tests/services/test_state_manager_extended.py",
                "tests/services/test_phase1.py",
                "tests/services/test_phase2_data_handler.py",
                "tests/services/test_phase2_filter_manager.py",
                "tests/services/test_phase2_feature_engine.py",
                "tests/services/test_phase2_outlier_manager.py",
                "tests/services/test_phase2_report_engine.py",
            ]
        },
        {
            "name": "Existing Calculation Accuracy Tests",
            "files": [
                "tests/services/test_calculation_accuracy.py",
                "tests/services/test_comprehensive_validation.py",
                "tests/services/test_edge_cases_controlled.py",
                "tests/services/test_report_accuracy.py",
            ]
        },
        {
            "name": "Existing Integration Tests",
            "files": [
                "tests/services/test_integration.py",
                "tests/services/test_data_handler.py",
                "tests/services/test_feature_engine.py",
                "tests/services/test_report_engine.py",
            ]
        },
        {
            "name": "Phase 3 UI Component Tests",
            "files": [
                "tests/test_phase3_ui_components.py",
            ]
        },
        {
            "name": "Phase 3 Comprehensive Tests",
            "files": [
                "tests/test_phase3_comprehensive_updated.py",
            ]
        },
        {
            "name": "Streamlit Best Practices Tests",
            "files": [
                "tests/test_streamlit_best_practices.py",
            ]
        },
    ]
    
    # Track results
    results = {}
    total_tests = len(test_suites)
    passed_tests = 0
    
    # Run each test suite
    for suite in test_suites:
        # Check if all test files exist
        existing_files = [f for f in suite["files"] if os.path.exists(f)]
        if not existing_files:
            print(f"\n⚠️  No test files found for {suite['name']}")
            results[suite["name"]] = "SKIPPED"
            continue
        
        if len(existing_files) < len(suite["files"]):
            missing_files = [f for f in suite["files"] if not os.path.exists(f)]
            print(f"\n⚠️  Some test files missing for {suite['name']}: {missing_files}")
        
        success = run_test_suite(existing_files, suite["name"])
        results[suite["name"]] = "PASSED" if success else "FAILED"
        
        if success:
            passed_tests += 1
    
    # Run browser tests if requested
    if "--browser" in sys.argv:
        browser_success = run_browser_tests()
        results["Browser UI Tests"] = "PASSED" if browser_success else "FAILED"
        total_tests += 1
        if browser_success:
            passed_tests += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for suite_name, result in results.items():
        status_icon = "✓" if result == "PASSED" else "✗" if result == "FAILED" else "⚠️"
        print(f"{status_icon} {suite_name}: {result}")
    
    print(f"\nOverall Results:")
    print(f"Passed: {passed_tests}/{total_tests} test suites")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All test suites passed!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} test suite(s) failed")
        return 1

def run_specific_test(test_file):
    """Run a specific test file"""
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return 1
    
    print(f"Running specific test: {test_file}")
    return run_test_suite([test_file], f"Specific Test: {test_file}")

if __name__ == "__main__":
    # Check for specific test file argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        test_file = sys.argv[1]
        exit_code = run_specific_test(test_file)
    else:
        exit_code = main()
    
    sys.exit(exit_code)
