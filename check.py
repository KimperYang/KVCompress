import matplotlib.pyplot as plt
import numpy as np

# Example list of integer numbers
data = [3, 5, 2, 5, 7, 5, 6, 3, 5, 4,
        5, 5, 4, 2, 5, 6, 6, 5, 5, 4]

# Compute statistics for reference (optional)
mean_val = np.mean(data)
median_val = np.median(data)

# Set up bins so each integer has its own bin
# +2 on max(data) because range() is exclusive at the upper boundary
bins = range(min(data), max(data) + 2)

plt.hist(data, bins=bins, align='left', alpha=0.8)
plt.axvline(mean_val, linestyle='--', label=f"Mean = {mean_val:.2f}")
plt.axvline(median_val, linestyle=':', label=f"Median = {median_val:.2f}")

plt.title("Distribution of Integer Data")
plt.xlabel("Integer Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.legend()
plt.savefig("test.png")  
