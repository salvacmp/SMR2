import numpy as np
import matplotlib.pyplot as plt

# Input data (camera coordinates)
# [x, y, 1]
A = np.array([
    [498, 227, 1],
    [495, 30, 1],
    [198, 38, 1],
    [207, 248, 1],
    [221, 444, 1,],
    [506, 432, 1],
])

# Target data (robot coordinates)
# [x, y]
B = np.array([
    [-0.29003, -0.40826],
    [-0.50619, -0.41537],
    [-0.49599, -0.74428],
    [-0.26745, -0.72710],
    [-0.04279, -0.71209],
    [-0.05393, -0.39335],
])

# Solve the least squares problem
coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

# Print each coefficient
print("Coefficients:")
print(f"a = {coefficients[0, 0]}")
print(f"b = {coefficients[1, 0]}")
print(f"c = {coefficients[2, 0]}")
print(f"d = {coefficients[0, 1]}")
print(f"e = {coefficients[1, 1]}")
print(f"f = {coefficients[2, 1]}")

# Apply the transformation to the camera coordinates
transformed_coords = A @ coefficients

# Compare the transformed coordinates with the robot coordinates
print("\nTransformed coordinates vs. Actual robot coordinates:")
for i in range(len(B)):
    print(f"Transformed: {transformed_coords[i]} vs. Actual: {B[i]}")

# Calculate the Mean Squared Error (MSE)
mse = np.mean((B - transformed_coords) ** 2)
print(f"\nMean Squared Error (MSE): {mse}")

# Calculate the accuracy as a percentage
max_value = np.max(B)
accuracy = 100 - (mse / max_value) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Calculate Euclidean Distance
distances = np.linalg.norm(B - transformed_coords, axis=1)
average_distance = np.mean(distances)
print(f"Average Euclidean Distance: {average_distance:.4f}")

# Plotting the results
fig, ax = plt.subplots()
ax.scatter(B[:,0], B[:,1], c='r', marker='o', label='Actual robot coords')
ax.scatter(transformed_coords[:,0], transformed_coords[:,1], c='b', marker='^', label='Transformed coords')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()

plt.show()
