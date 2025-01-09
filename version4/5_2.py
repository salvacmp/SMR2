import numpy as np
import matplotlib.pyplot as plt

# Input data (camera coordinates)
# [x, y, z, 1]
A = np.array([
    [161, 38, 0.683, 1],
    [494, 27, 0.683, 1],
    [519, 452, 0.683, 1],
    [180, 459, 0.683, 1],
])

# Target data (robot coordinates)
# [x, y, z]
B = np.array([
    [-0.52507, -0.38959, 0.79847],
    [-0.51195, -0.76239, 0.79847],
    [-0.04092, -0.74408, 0.79847],
    [-0.04831, -0.36217, 0.79847],
])

# Solve the least squares problem
coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

# Print each coefficient
print("Coefficients:")
print(f"a = {coefficients[0, 0]}")
print(f"b = {coefficients[1, 0]}")
print(f"c = {coefficients[2, 0]}")
print(f"d = {coefficients[3, 0]}")
print(f"e = {coefficients[0, 1]}")
print(f"f = {coefficients[1, 1]}")
print(f"g = {coefficients[2, 1]}")
print(f"h = {coefficients[3, 1]}")
print(f"i = {coefficients[0, 2]}")
print(f"j = {coefficients[1, 2]}")
print(f"k = {coefficients[2, 2]}")
print(f"l = {coefficients[3, 2]}")

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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(B[:,0], B[:,1], B[:,2], c='r', marker='o', label='Actual robot coords')
ax.scatter(transformed_coords[:,0], transformed_coords[:,1], transformed_coords[:,2], c='b', marker='^', label='Transformed coords')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()

plt.show()
