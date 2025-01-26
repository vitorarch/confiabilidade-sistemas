import matplotlib.pyplot as plt
import numpy as np
import time

# Enable interactive mode
plt.ion()

# Create a figure
fig = plt.figure()

# Data range
x = np.linspace(0, 2 * np.pi, 100)

for i in range(100):
    # Clear the figure
    plt.clf()
    
    # Generate new data
    y = np.sin(x + i * 0.1)
    
    # Redraw the plot
    plt.plot(x, y, label=f"Iteration {i+1}")
    plt.legend()
    
    # Add title and labels
    plt.title("Dynamic Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    # Draw the updated plot
    plt.draw()
    
    # Pause to refresh the figure
    plt.pause(0.1)

# Keep the plot open after the loop
plt.ioff()
plt.show()
