"""
Matplotlib Cheatsheet
Author: Rishi Sharma
Description: This file contains a comprehensive Matplotlib cheatsheet with essential functions and operations.
"""
import matplotlib.pyplot as plt
import numpy as np

# Basic Line Plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Line Plot')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, color='red', marker='o', label='Data points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.legend()
plt.show()

# Bar Chart
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 1, 8]
plt.bar(categories, values, color='blue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, color='green', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Pie Chart
labels = ['Apple', 'Banana', 'Cherry', 'Dates']
sizes = [30, 20, 35, 15]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'yellow', 'pink', 'brown'])
plt.title('Pie Chart')
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y, 'b')
axes[0, 0].set_title('Line Plot')
axes[0, 1].scatter(x, y, color='r')
axes[0, 1].set_title('Scatter Plot')
axes[1, 0].bar(categories, values, color='g')
axes[1, 0].set_title('Bar Chart')
axes[1, 1].hist(data, bins=20, color='purple')
axes[1, 1].set_title('Histogram')
plt.tight_layout()
plt.show()

# Saving Figures
plt.plot(x, y)
plt.title('Saved Figure Example')
plt.savefig('plot.png', dpi=300)
plt.show()
