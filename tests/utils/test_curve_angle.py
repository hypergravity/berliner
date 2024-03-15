import numpy as np

# Define two vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Calculate the length of the vectors
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)

# Calculate the dot product
dot_product = np.dot(a, b)

# Calculate the cosine of the angle
cos_theta = dot_product / (norm_a * norm_b)

# Calculate the angle
theta = np.arccos(cos_theta)

# Convert angle from radians to degrees (optional)
theta_degrees = np.degrees(theta)

print("Angle in radians:", theta)
print("Angle in degrees:", theta_degrees)
