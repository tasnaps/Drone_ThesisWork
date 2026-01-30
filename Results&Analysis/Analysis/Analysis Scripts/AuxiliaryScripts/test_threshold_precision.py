import pandas as pd

# Test 1: String preservation
print("Test 1: String value preservation")
test_val = '0.3040691912174225'
df = pd.DataFrame({'aggregation_threshold': [test_val]*3})
df.to_csv('test1.csv', index=False)
df2 = pd.read_csv('test1.csv', dtype=str)
result = df2['aggregation_threshold'].iloc[0]
print(f"  Original: {test_val}")
print(f"  After CSV: {result}")
print(f"  Match: {result == test_val}")

# Test 2: repr() of float
print("\nTest 2: repr() of float")
test_float = 0.3040691912174225
df = pd.DataFrame({'aggregation_threshold': [repr(test_float)]*3})
df.to_csv('test2.csv', index=False)
df2 = pd.read_csv('test2.csv', dtype=str)
result = df2['aggregation_threshold'].iloc[0]
print(f"  Original float: {test_float}")
print(f"  repr(): {repr(test_float)}")
print(f"  After CSV: {result}")

# Cleanup
import os
os.remove('test1.csv')
os.remove('test2.csv')
