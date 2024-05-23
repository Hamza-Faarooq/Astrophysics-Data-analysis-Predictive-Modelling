import pandas as pd
import numpy as np

# Creating a mock DataFrame
data = {
    'u': np.random.uniform(14, 25, 100),
    'g': np.random.uniform(13, 25, 100),
    'r': np.random.uniform(12, 24, 100),
    'i': np.random.uniform(11, 23, 100),
    'z': np.random.uniform(10, 22, 100),
    'redshift': np.random.uniform(0, 2, 100)
}

df = pd.DataFrame(data)

# Saving DataFrame to CSV
df.to_csv('sdss_data.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of redshift
plt.hist(data['redshift'], bins=30, edgecolor='k')
plt.title('Redshift Distribution')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.show()

# Pairplot of features
sns.pairplot(data[['u', 'g', 'r', 'i', 'z']])
plt.show()

# Correlation matrix
corr_matrix = data[['u', 'g', 'r', 'i', 'z', 'redshift']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()