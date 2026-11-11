# %% [markdown]
# # Group Practical Assignment: Clustering Techniques (Unsupervised Learning)
# **Course:** Machine Learning / Artificial Intelligence Practical  
# **Group:** Group 1  
# **Dataset:** Mall Customers Dataset (`Mall_Customers.csv`)  
# **Student:** Monero Ashley Mwasa  
# **Topic:** K-Means vs. Hierarchical Clustering Comparison  
# **Environment:** Python 3.14 / VS Code  
# 
# ---
# 
# ### What this notebook covers:
# In this assignment, our group is working with the **Mall Customers** dataset (200 customers). Our main goal is to find natural customer groups based on how much they earn and how much they spend at the mall.
# 
# We compare two main clustering methods we learned in class:
# 1. **K-Means Clustering** (using $k$-means++ initialization)
# 2. **Hierarchical Clustering** (comparing Ward linkage and Complete linkage)
# 
# Here is our workflow:
# - **Part A:** Load and clean the data, check for duplicates and missing values, and scale features.
# - **Part B:** First Exploratory Data Analysis (summary stats, histograms, boxplots, pairplot, and correlation).
# - **Part C:** Feature engineering (Income-to-Spend ratio and Age Groups) and testing a log transform on Income.
# - **Part D:** Finding the best number of clusters ($k$) using 4 metrics (Elbow/Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz), and comparing K-Means with Hierarchical dendrograms.
# - **Part E:** Profiling the cluster centroids in original units and running statistical hypothesis tests (ANOVA, Kruskal-Wallis, and Chi-Square).
# - **Part F:** Summary of customer personas, model limitations, proposed improvements (K-Prototypes for mixed data), and a cluster stability test across 50 random seeds.

# %%
# Step 0: Import the libraries we need for data analysis, clustering, and plotting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn tools for scaling, K-Means, and cluster evaluation metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# SciPy tools for hierarchical clustering and hypothesis tests
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy import stats

# Set a clean plot style for VS Code
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'

# Work out the folder this script lives in, so it runs from any computer.
# (__file__ is not defined in notebook / interactive cells, so fall back to the working folder.)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# Make sure our figures directory exists (next to this script)
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
print("All libraries imported successfully!")

# %% [markdown]
# ---
# ## Part A: Data Loading & Preprocessing (15 points)
# In this section, we:
# 1. Load the dataset from CSV and check the first 10 rows and data types.
# 2. Check for duplicate rows and remove them if any exist.
# 3. Check for missing values and set up median/mode imputation.
# 4. Save a reproducible copy with `random_state=42`.
# 5. Scale numerical features with `StandardScaler` and explain why we keep `Gender` separate from the clustering distance.

# %%
# 1. Load the dataset and look at the first 10 rows
# Look for the CSV next to this script first, then in the current working folder
data_path = os.path.join(BASE_DIR, "Mall_Customers.csv")
if not os.path.exists(data_path):
    data_path = os.path.join(os.getcwd(), "Mall_Customers.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(
        "Mall_Customers.csv not found. Put it in the same folder as this script: " + BASE_DIR
    )

df_raw = pd.read_csv(data_path)
print("=== FIRST 10 ROWS OF RAW DATA ===")
print(df_raw.head(10))

print("\n=== COLUMN DATA TYPES ===")
print(df_raw.dtypes)

# %%
# 2. Check for exact duplicate rows
duplicate_count = df_raw.duplicated().sum()
print(f"Number of duplicate rows found: {duplicate_count}")

# Remove duplicates if any were found
df_dedup = df_raw.drop_duplicates()
print(f"Rows before: {len(df_raw)}, Rows after deduplication: {len(df_dedup)}")

# 3. Check for missing values
missing_per_col = df_dedup.isnull().sum()
print("\n=== MISSING VALUES PER COLUMN ===")
print(missing_per_col)

# We set up a rule: drop rows with >50% missing values, and impute the rest (median for numbers, mode for Gender)
df_cleaned = df_dedup.copy()
row_missing_threshold = 0.5 * df_cleaned.shape[1]
rows_before = len(df_cleaned)
df_cleaned = df_cleaned.dropna(thresh=int(row_missing_threshold))
print(f"Rows dropped due to >50% missing values: {rows_before - len(df_cleaned)}")

# Impute numerical features using median (if any are missing)
for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    if df_cleaned[col].isnull().sum() > 0:
        med = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(med)
        print(f"Imputed missing values in '{col}' using median: {med}")

# Impute Gender using mode (if missing)
if df_cleaned['Gender'].isnull().sum() > 0:
    mode_val = df_cleaned['Gender'].mode()[0]
    df_cleaned['Gender'] = df_cleaned['Gender'].fillna(mode_val)
    print(f"Imputed missing values in 'Gender' using mode: {mode_val}")

print("\nMissing values after check:")
print(df_cleaned.isnull().sum())

# 4. Set random_state=42 for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
df = df_cleaned.copy()
print(f"\nCleaned reproducible dataset ready! Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

# %% [markdown]
# ### Preprocessing Decisions: Feature Scaling & Why We Exclude Gender
# 
# 1. **Why we use StandardScaler:**  
#    Both K-Means and Hierarchical clustering calculate similarity using straight-line Euclidean distance:
#    $$d(x, y) = \sqrt{\sum (x_i - y_i)^2}$$
#    In our dataset, `Annual Income (k$)` goes from 15 up to 137, while `Spending Score` goes from 1 to 100, and `Age` is 18 to 70. Because Annual Income has a much wider raw numerical spread (variance of ~690), it would naturally dominate the distance calculations if we left it unscaled. By using `StandardScaler`, we transform every feature to have a mean of 0 and a standard deviation of 1 ($z = \frac{x - \mu}{\sigma}$). This puts all features on an equal playing field.
# 
# 2. **Why we exclude Gender from the K-Means distance calculation:**  
#    `Gender` is a binary categorical variable (Male / Female). If we encode it as 0 and 1, it creates an artificial gap of 1.0 standard unit in continuous Euclidean space. K-Means assumes continuous, round clusters, so adding a 0/1 variable often forces the algorithm to split people by gender rather than their actual shopping behavior.  
#    Because our goal is customer segmentation based on spending habits, we cluster on the continuous behavioral features (`Annual Income` and `Spending Score`) and keep `Gender` aside. In Part E, we run a Chi-Square test of independence to see if gender actually affects cluster membership.

# %%
# 5. Standardize numeric features using StandardScaler
numeric_features = ['Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_features])

# We also prepare a scaled 3D version including Age for comparison
scaler_3d = StandardScaler()
X_scaled_3d = scaler_3d.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

print(f"Scaled 2D feature matrix shape: {X_scaled.shape}")
print(f"Feature means after scaling: {X_scaled.mean(axis=0).round(4)} (close to 0)")
print(f"Feature std dev after scaling: {X_scaled.std(axis=0).round(4)} (equal to 1)")

# %% [markdown]
# ---
# ## Part B: First Exploratory Data Analysis (20 points)
# In this section, we explore what the data looks like:
# 1. Calculate descriptive statistics (Mean, Median, Standard Deviation, IQR, Skewness, Outliers).
# 2. Plot histograms and boxplots to check for skewness and unusual data points.
# 3. Create a pairplot colored by Gender to spot natural groupings.
# 4. Calculate the correlation matrix and display a heatmap.

# %%
# 1. Descriptive statistics for Age, Annual Income, and Spending Score
summary_list = []
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for col in numeric_cols:
    mean_val = df[col].mean()
    med_val = df[col].median()
    std_val = df[col].std()
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr_val = q75 - q25
    skew_val = df[col].skew()
    
    # Check for outliers using the standard 1.5 * IQR rule
    lower_limit = q25 - 1.5 * iqr_val
    upper_limit = q75 + 1.5 * iqr_val
    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)][col].tolist()
    
    summary_list.append({
        'Feature': col,
        'Mean': round(mean_val, 2),
        'Median': round(med_val, 2),
        'Std Dev': round(std_val, 2),
        'Q1 (25%)': round(q25, 2),
        'Q3 (75%)': round(q75, 2),
        'IQR': round(iqr_val, 2),
        'Skewness': round(skew_val, 2),
        'Outliers Count': len(outliers),
        'Outlier Values': outliers
    })

summary_table = pd.DataFrame(summary_list)
print("=== DESCRIPTIVE STATISTICS TABLE ===")
print(summary_table.to_string(index=False))

# %% [markdown]
# ### Comments on Skewness and Outliers
# - **Age:** Mean is 38.85 and median is 36.00, with a skewness of +0.49 (slightly right-skewed). There are no outliers; customers range from 18 to 70 years old.
# - **Annual Income (k$):** Mean is $60.56k and median is $61.50k, with a mild positive skewness of +0.32. There are 2 outliers at $137k (Customers #199 and #200), because $Q3 + 1.5 \times IQR = 78 + (1.5 \times 36.5) = 132.75k$. These represent wealthy shoppers, which are realistic and important for a mall segmentation study, so we keep them in the dataset.
# - **Spending Score (1-100):** Mean is 50.20 and median is 50.00, with skewness of -0.05. This feature is almost perfectly symmetrical and has zero outliers.

# %%
# 2. Histograms and Boxplots for numerical features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
colors = ['#2b5c8f', '#2a9d8f', '#e76f51']

for i, col in enumerate(numeric_cols):
    # Histogram with KDE curve
    sns.histplot(df[col], kde=True, ax=axes[0, i], color=colors[i], bins=15, edgecolor='black', alpha=0.6)
    axes[0, i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {df[col].mean():.1f}")
    axes[0, i].axvline(df[col].median(), color='black', linestyle=':', linewidth=1.5, label=f"Median: {df[col].median():.1f}")
    axes[0, i].set_title(f"{col} Distribution\n(Skewness: {df[col].skew():.2f})", fontsize=11, fontweight='bold')
    axes[0, i].set_xlabel(col, fontsize=10)
    axes[0, i].set_ylabel("Count", fontsize=10)
    axes[0, i].legend(fontsize=9, loc='upper right')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[1, i], color=colors[i], flierprops=dict(marker='o', markersize=6, markerfacecolor='red'))
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr = q75 - q25
    axes[1, i].set_title(f"{col} Boxplot\n(IQR: {iqr:.1f}, Q1: {q25:.1f}, Q3: {q75:.1f})", fontsize=11, fontweight='bold')
    axes[1, i].set_xlabel(col, fontsize=10)

plt.suptitle("Figure 1: Histograms and Boxplots for Numeric Features", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%
# 3. Pairwise scatter plots (pairplot) colored by Gender
pair_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']].copy()
pairplot = sns.pairplot(pair_df, hue='Gender', palette={'Male': '#2b5c8f', 'Female': '#e76f51'},
                        diag_kind='kde', plot_kws={'alpha': 0.75, 's': 45, 'edgecolor': 'k', 'linewidth': 0.5})
pairplot.fig.subplots_adjust(top=0.93)
pairplot.fig.suptitle("Figure 2: Pairwise Scatter Plots by Gender", fontsize=13, fontweight='bold')
plt.show()

# %% [markdown]
# ### Observations from the Pairplot
# 1. When we look at **Annual Income vs. Spending Score**, we can clearly see **5 distinct clusters** in the scatter plot:
#    - Bottom-left: Low income, low spending
#    - Top-left: Low income, high spending
#    - Center: Average income, average spending
#    - Bottom-right: High income, low spending
#    - Top-right: High income, high spending
# 2. Men and women are evenly mixed across all these regions. There are no separate "male only" or "female only" clusters.
# 3. Looking at **Age vs. Spending Score**, customers over 40 almost never have a spending score above 60. Most high spenders (score > 70) are young adults under 40.

# %%
# 4. Correlation Matrix and Heatmap
plt.figure(figsize=(7, 5))
corr_matrix = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-0.5, vmax=1.0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 11, "weight": "bold"})
plt.title("Figure 3: Correlation Matrix Heatmap", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()

print("=== CORRELATION VALUES ===")
print(corr_matrix.round(4))

# %% [markdown]
# ### Correlation Summary
# - **Age vs. Spending Score:** Moderate negative correlation ($r = -0.327, p < 0.001$). As age goes up, mall spending tends to go down.
# - **Annual Income vs. Spending Score:** Almost zero linear correlation ($r = 0.0099$). This was surprising at first, but it makes sense because income and spending have a non-linear relationship: some high earners spend a lot, while others save a lot! Linear correlation can't see these patterns, but clustering can.
# - **Age vs. Annual Income:** Negligible correlation ($r = -0.0124$).

# %% [markdown]
# ---
# ## Part C: Feature Engineering & Transformations (15 points)
# In this section, we create new features:
# 1. `Income_to_Spend`: Ratio of Annual Income to Spending Score.
# 2. `AgeGroup`: Customer life-stages (`<25`, `25-35`, `36-50`, `>50`).
# 3. Test a log transform on Income to see if it fixes skewness.
# 4. Finalize the feature set for our clustering models.

# %%
# 1. Feature 1: Income-to-Spend Ratio
# We add +1 to the denominator so we never divide by zero
df['Income_to_Spend'] = df['Annual Income (k$)'] / (df['Spending Score (1-100)'] + 1.0)
print("=== INCOME TO SPEND RATIO SUMMARY ===")
print(df['Income_to_Spend'].describe().round(3))

# 2. Feature 2: Age Group Bins
age_bins = [0, 25, 35, 50, 100]
age_labels = ['<25', '25-35', '36-50', '>50']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
print("\n=== CUSTOMERS IN EACH AGE GROUP ===")
print(df['AgeGroup'].value_counts().sort_index())

# 3. Test Log Transformation on Annual Income
raw_skew = df['Annual Income (k$)'].skew()
log_income = np.log(df['Annual Income (k$)'])
log_skew = log_income.skew()

print(f"\nRaw Income Skewness: {raw_skew:.4f}")
print(f"Log-Transformed Income Skewness: {log_skew:.4f}")

# %% [markdown]
# ### Feature Engineering Discussion & Decision
# - **Testing Log Transformation:**  
#   The raw Annual Income had a skewness of **+0.3218** (already quite balanced, between -0.5 and +0.5). When we took the natural log, the skewness became **-0.7596** (moderately left-skewed). The log transform actually over-corrected and introduced more skewness than we started with. Because of this, **we decided not to use the log transform**, and instead kept the original linear scale with standard scaling.
# - **How the engineered features help:**  
#   - `Income_to_Spend` shows who is frugal vs. who is a big spender relative to their paycheck. High-income savers have high ratios (up to 23.3), while young impulse buyers have very low ratios (around 0.3).  
#   - `AgeGroup` translates continuous age into practical retail cohorts: Students/Gen Z (<25), Young Professionals (25-35), Families (36-50), and Seniors (>50).
# - **Final Feature Set for Clustering:**  
#   `['Annual Income (k$)', 'Spending Score (1-100)']` scaled with `StandardScaler`. This 2D space gives 5 clear, interpretable customer segments without adding extra noise.

# %% [markdown]
# ---
# ## Part D: Clustering Modelling & Selection (25 points)
# In this section, we:
# 1. Run K-Means for $k = 2$ through $8$ with `n_init=15` and calculate 4 evaluation metrics:
#    - Inertia (Elbow method)
#    - Silhouette Coefficient (higher is better)
#    - Davies-Bouldin Index (lower is better)
#    - Calinski-Harabasz Index (higher is better)
# 2. Choose the optimal $k$ and explain why.
# 3. Run Hierarchical Clustering with **Ward** and **Complete** linkages, plot both dendrograms, and compute their cophenetic correlation.
# 4. Compare Hierarchical cuts ($k=2..6$) against our K-Means results.

# %%
# 1. Run K-Means for k = 2..8 and compute all 4 metrics
k_range = list(range(2, 9))
kmeans_metrics = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    
    kmeans_metrics.append({
        'k': k,
        'Inertia': round(inertia, 2),
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'Calinski-Harabasz': round(ch, 2)
    })

metrics_table = pd.DataFrame(kmeans_metrics)
print("=== K-MEANS METRIC RESULTS (k=2..8) ===")
print(metrics_table.to_string(index=False))

# %%
# Plot the 4 evaluation metrics vs k
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) Inertia
axes[0, 0].plot(metrics_table['k'], metrics_table['Inertia'], 'o-', color='#2b5c8f', linewidth=2, markersize=7)
axes[0, 0].axvline(5, color='crimson', linestyle='--', linewidth=1.5, label='Elbow at k=5')
axes[0, 0].set_title("(a) Inertia / Elbow Method (Look for the bend)", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel("Number of Clusters (k)", fontsize=10)
axes[0, 0].set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, linestyle=':', alpha=0.6)

# (b) Silhouette Score
axes[0, 1].plot(metrics_table['k'], metrics_table['Silhouette'], 's-', color='#2a9d8f', linewidth=2, markersize=7)
axes[0, 1].axvline(5, color='crimson', linestyle='--', linewidth=1.5, label='Peak at k=5 (0.555)')
axes[0, 1].set_title("(b) Silhouette Score (Higher is better)", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("Number of Clusters (k)", fontsize=10)
axes[0, 1].set_ylabel("Silhouette Score", fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, linestyle=':', alpha=0.6)

# (c) Davies-Bouldin Index
axes[1, 0].plot(metrics_table['k'], metrics_table['Davies-Bouldin'], '^-', color='#e76f51', linewidth=2, markersize=7)
axes[1, 0].axvline(5, color='crimson', linestyle='--', linewidth=1.5, label='Minimum at k=5 (0.572)')
axes[1, 0].set_title("(c) Davies–Bouldin Index (Lower is better)", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Number of Clusters (k)", fontsize=10)
axes[1, 0].set_ylabel("Davies–Bouldin Index", fontsize=10)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, linestyle=':', alpha=0.6)

# (d) Calinski-Harabasz Index
axes[1, 1].plot(metrics_table['k'], metrics_table['Calinski-Harabasz'], 'd-', color='#9b5de5', linewidth=2, markersize=7)
axes[1, 1].axvline(5, color='crimson', linestyle='--', linewidth=1.5, label='Peak at k=5 (248.65)')
axes[1, 1].set_title("(d) Calinski–Harabasz Index (Higher is better)", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Number of Clusters (k)", fontsize=10)
axes[1, 1].set_ylabel("Calinski–Harabasz Score", fontsize=10)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Figure 4: Evaluation Metrics for K-Means (k = 2 to 8)", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Why k = 5 is the Optimal Choice
# Every single metric pointed directly to **$k = 5$**:
# 1. **Elbow Method (Inertia):** Drops sharply from $k=2$ (270.7) down to $k=5$ (65.6). After $k=5$, it levels off ($k=6$ is 55.1). This creates a clear elbow at $k=5$.
# 2. **Silhouette Score:** Hits its highest value (**0.5547**) at $k=5$.
# 3. **Davies–Bouldin Index:** Hits its lowest (best) score (**0.5722**) at $k=5$.
# 4. **Calinski–Harabasz Index:** Reaches its highest peak (**248.65**) at $k=5$.
# 5. **Real-world Sense:** $k=5$ matches the 5 natural customer groups we saw earlier in the scatter plot (Low/Low, Low/High, Average/Average, High/Low, High/High).

# %%
# 3. Hierarchical Clustering: Ward vs. Complete Linkage
pdist_matrix = pdist(X_scaled)

# Ward linkage
Z_ward = linkage(X_scaled, method='ward')
c_ward, _ = cophenet(Z_ward, pdist_matrix)

# Complete linkage
Z_complete = linkage(X_scaled, method='complete')
c_complete, _ = cophenet(Z_complete, pdist_matrix)

print(f"Cophenetic Correlation (Ward Linkage):     {c_ward:.4f}")
print(f"Cophenetic Correlation (Complete Linkage): {c_complete:.4f}")

# Plot both dendrograms
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

dendrogram(Z_ward, ax=axes[0], truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8, color_threshold=5.5)
axes[0].axhline(y=5.5, color='crimson', linestyle='--', linewidth=1.2, label='k=5 Cut Line (height 5.5)')
axes[0].set_title(f"Ward Linkage Dendrogram\n(Cophenetic Corr: {c_ward:.4f})", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Truncated Cluster Index", fontsize=10)
axes[0].set_ylabel("Distance", fontsize=10)
axes[0].legend(fontsize=9, loc='upper right')

dendrogram(Z_complete, ax=axes[1], truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8, color_threshold=2.8)
axes[1].axhline(y=2.8, color='crimson', linestyle='--', linewidth=1.2, label='k=5 Cut Line (height 2.8)')
axes[1].set_title(f"Complete Linkage Dendrogram\n(Cophenetic Corr: {c_complete:.4f})", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Truncated Cluster Index", fontsize=10)
axes[1].set_ylabel("Distance", fontsize=10)
axes[1].legend(fontsize=9, loc='upper right')

plt.suptitle("Figure 5: Hierarchical Dendrograms (Ward vs. Complete Linkage)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%
# 4. Compare cuts for k = 2..6 between Hierarchical and K-Means
comparison_rows = []

for k in range(2, 7):
    # Ward cut
    lbl_w = fcluster(Z_ward, k, criterion='maxclust')
    sil_w = silhouette_score(X_scaled, lbl_w)
    db_w = davies_bouldin_score(X_scaled, lbl_w)
    
    # Complete cut
    lbl_c = fcluster(Z_complete, k, criterion='maxclust')
    sil_c = silhouette_score(X_scaled, lbl_c)
    db_c = davies_bouldin_score(X_scaled, lbl_c)
    
    # K-Means scores for comparison
    km_row = metrics_table[metrics_table['k'] == k].iloc[0]
    
    comparison_rows.append({
        'k': k,
        'Ward Sil': round(sil_w, 4),
        'Ward DB': round(db_w, 4),
        'Complete Sil': round(sil_c, 4),
        'Complete DB': round(db_c, 4),
        'K-Means Sil': km_row['Silhouette'],
        'K-Means DB': km_row['Davies-Bouldin']
    })

comparison_df = pd.DataFrame(comparison_rows)
print("=== HIERARCHICAL CUTS VS. K-MEANS COMPARISON ===")
print(comparison_df.to_string(index=False))

# %% [markdown]
# ### Hierarchical vs. K-Means Takeaways
# 1. **Cophenetic Correlation:** Ward linkage scored **0.7209**, while Complete linkage scored **0.6670**. Ward preserved the actual Euclidean distances between customers better.
# 2. **Comparing Cuts:** At $k=5$, Ward linkage gave a Silhouette score of **0.5538** and Davies-Bouldin of **0.5779**, which is almost identical to K-Means (**0.5547** and **0.5722**). Complete linkage also matched closely (**0.5531** and **0.5680**).
# 3. Because these different algorithms arrived at almost the exact same 5 clusters, we can be confident these customer segments are genuine patterns in the data.

# %% [markdown]
# ---
# ## Part E: Second EDA & Statistical Inference (15 points)
# In this section, we:
# 1. Fit the final K-Means model ($k=5$) and calculate the cluster centroids (means) in original units ($k and score).
# 2. Test if the clusters are significantly different on numeric features using ANOVA and Kruskal-Wallis.
# 3. Test if categorical features (`Gender` and `AgeGroup`) are related to the clusters using Chi-Square tests of independence.
# 4. Describe the 5 customer personas.

# %%
# 1. Fit the final K-Means model with k=5 and tabulate centroids in original units
kmeans_final = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=RANDOM_STATE)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Calculate means and counts for each cluster in original units
centroids_tab = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Age': ['mean', 'std', 'count']
}).round(2)

print("=== CLUSTER CENTROIDS IN ORIGINAL UNITS ===")
print(centroids_tab)

# %%
# Plot the 5 clusters with their centroids
cluster_names = {
    0: "Middle-Class Essentials (Cluster 0)",
    1: "VIP Spenders (Cluster 1)",
    2: "Young Impulsive Shoppers (Cluster 2)",
    3: "Careful Savers (Cluster 3)",
    4: "Budget Shoppers (Cluster 4)"
}
df['Persona'] = df['Cluster'].map(cluster_names)
palette = ['#2b5c8f', '#2a9d8f', '#e76f51', '#e63946', '#9b5de5']

plt.figure(figsize=(10, 7))
for c_id in range(5):
    sub = df[df['Cluster'] == c_id]
    plt.scatter(sub['Annual Income (k$)'], sub['Spending Score (1-100)'],
                s=70, color=palette[c_id], label=cluster_names[c_id], alpha=0.85, edgecolors='k', linewidth=0.5)

# Convert the scaled centroids back to original units and plot as yellow stars
centroids_orig = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_orig[:, 0], centroids_orig[:, 1], s=250, c='yellow', marker='*',
            edgecolors='black', linewidth=1.5, label='Centroids (Cluster Means)', zorder=5)

for i, (cx, cy) in enumerate(centroids_orig):
    plt.text(cx, cy + 3.2, f"C{i}", fontsize=11, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))

plt.title("Figure 6: The 5 Customer Clusters with Centroids", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Annual Income (k$)", fontsize=11, fontweight='bold')
plt.ylabel("Spending Score (1–100)", fontsize=11, fontweight='bold')
plt.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.9)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# 2. Statistical Tests: ANOVA and Kruskal-Wallis for numerical features
test_features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Income_to_Spend']
test_records = []

for feat in test_features:
    groups = [df[df['Cluster'] == c][feat].values for c in range(5)]
    f_stat, p_f = stats.f_oneway(*groups)
    h_stat, p_h = stats.kruskal(*groups)
    
    test_records.append({
        'Feature': feat,
        'ANOVA F-stat': round(f_stat, 2),
        'ANOVA p-val': f"{p_f:.3e}",
        'Kruskal H-stat': round(h_stat, 2),
        'Kruskal p-val': f"{p_h:.3e}",
        'Significant?': 'Yes (p < 0.05)' if p_f < 0.05 else 'No'
    })

stat_table = pd.DataFrame(test_records)
print("=== NUMERICAL SIGNIFICANCE TESTS ===")
print(stat_table.to_string(index=False))

# %%
# 3. Categorical Tests: Chi-Square Test of Independence
# (a) Gender vs. Cluster
ct_gender = pd.crosstab(df['Cluster'], df['Gender'])
chi2_g, p_g, dof_g, _ = stats.chi2_contingency(ct_gender)

print("\n=== CROSSTAB: GENDER VS. CLUSTER ===")
print(ct_gender)
print(f"Chi-Square: {chi2_g:.4f}, p-value: {p_g:.4f}, df: {dof_g}")

# (b) AgeGroup vs. Cluster
ct_age = pd.crosstab(df['Cluster'], df['AgeGroup'])
chi2_a, p_a, dof_a, _ = stats.chi2_contingency(ct_age)

print("\n=== CROSSTAB: AGE GROUP VS. CLUSTER ===")
print(ct_age)
print(f"Chi-Square: {chi2_a:.4f}, p-value: {p_a:.4e}, df: {dof_a}")

# %% [markdown]
# ### Statistical Interpretation
# 1. **Numeric Features:**
#    - Both Annual Income ($F = 187.80, p < 0.001$) and Spending Score ($F = 351.65, p < 0.001$) had tiny p-values, showing huge differences between the clusters.
#    - Interestingly, **Age** was also significantly different between clusters ($F = 12.37, p < 0.001$), even though we did not use Age to build the 2D clusters! Young shoppers make up the high-spending clusters (mean age 25 in Cluster 2, and 32 in Cluster 1), while older shoppers make up the low-spending clusters (mean age 41 in Cluster 3, and 45 in Cluster 4).
# 2. **Gender Independence:**
#    - The Chi-Square test for Gender gave $\chi^2 = 2.23$ with a p-value of **0.693**. Because $p > 0.05$, we fail to reject the null hypothesis. This means cluster membership does not depend on gender at all in this mall. This confirmed that our choice to leave Gender out of the distance metric was the right call!
# 3. **Age Group Dependency:**
#    - The Chi-Square test for AgeGroup gave $\chi^2 = 102.28$ with $p < 0.001$. This shows that customer life stages (students, young pros, families, retirees) are strongly tied to their cluster.

# %% [markdown]
# ---
# ## Part F: Presentation, Reflection & Stability (10 points)
# In this section, we:
# 1. Summarize our 5 customer personas and marketing strategies.
# 2. Reflect on limitations of our clustering algorithms.
# 3. Propose an improvement: **K-Prototypes** for mixed data.
# 4. **Optional Bonus:** Check cluster stability across 50 random initializations (`n_init=1` vs `n_init=10`).

# %%
# 4. Optional Task: Test Cluster Stability across 50 Random Seeds
seeds = list(range(50))
scores_ninit1 = []
scores_ninit10 = []

for s in seeds:
    # Single-start K-Means
    km1 = KMeans(n_clusters=5, init='k-means++', n_init=1, random_state=s)
    l1 = km1.fit_predict(X_scaled)
    scores_ninit1.append(silhouette_score(X_scaled, l1))
    
    # Standard multi-start K-Means
    km10 = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=s)
    l10 = km10.fit_predict(X_scaled)
    scores_ninit10.append(silhouette_score(X_scaled, l10))

scores_ninit1 = np.array(scores_ninit1)
scores_ninit10 = np.array(scores_ninit10)

print("=== CLUSTER STABILITY (50 SEEDS) ===")
print(f"n_init=1:  Mean Silhouette = {scores_ninit1.mean():.4f} ± {scores_ninit1.std():.4f} (Min: {scores_ninit1.min():.4f}, Max: {scores_ninit1.max():.4f})")
print(f"n_init=10: Mean Silhouette = {scores_ninit10.mean():.4f} ± {scores_ninit10.std():.4f} (Min: {scores_ninit10.min():.4f}, Max: {scores_ninit10.max():.4f})")

# Plot stability comparison
plt.figure(figsize=(9, 5))
plt.plot(seeds, scores_ninit1, 'o--', color='#e76f51', alpha=0.7, label=f"n_init=1 (Mean: {scores_ninit1.mean():.4f} ± {scores_ninit1.std():.4f})")
plt.plot(seeds, scores_ninit10, 's-', color='#2b5c8f', linewidth=2, label=f"n_init=10 (Mean: {scores_ninit10.mean():.4f} ± {scores_ninit10.std():.4f})")
plt.axhline(0.5547, color='black', linestyle=':', label="Optimal Score (0.5547)")
plt.title("Figure 7: Cluster Stability Across 50 Seeds (n_init=1 vs. n_init=10)", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Random Seed Index", fontsize=10)
plt.ylabel("Silhouette Score", fontsize=10)
plt.ylim(0.38, 0.58)
plt.legend(fontsize=9, loc='lower right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Final Reflections & Business Insights
# 
# #### 1. The 5 Customer Personas & Marketing Plan
# - **Cluster 0: Middle-Class Essentials (40.5% of shoppers, $n=81$):**  
#   *Profile:* Average income ($55.3k), average spend (49.5), mean age 42.7.  
#   *Strategy:* They are the bread-and-butter of mall foot traffic. Keep them engaged with grocery rewards, anchor department store sales, and family weekend events.
# - **Cluster 1: VIP Spenders (19.5% of shoppers, $n=39$):**  
#   *Profile:* High income ($86.5k), high spend (82.1), young average age 32.7.  
#   *Strategy:* Top revenue generators. Treat them to exclusive perks: private luxury previews, personal shopping assistants, VIP lounge access, and concierge service.
# - **Cluster 2: Young Impulsive Shoppers (11.0% of shoppers, $n=22$):**  
#   *Profile:* Low income ($25.7k), high spend (79.4), young average age 25.3.  
#   *Strategy:* Trend-driven youth and students. Engage them through TikTok/Instagram ads, trendy pop-up stores, Buy-Now-Pay-Later (BNPL) payment options, and experiential dining.
# - **Cluster 3: Careful Savers (17.5% of shoppers, $n=35$):**  
#   *Profile:* High income ($88.2k), low spend (17.1), mean age 41.1.  
#   *Strategy:* High purchasing power but frugal. Convince them with quality guarantees, durable investment goods, clear warranties, and financial planning seminars.
# - **Cluster 4: Budget Shoppers (11.5% of shoppers, $n=23$):**  
#   *Profile:* Low income ($26.3k), low spend (20.9), older average age 45.2.  
#   *Strategy:* Very price-sensitive. Attract them with clearance sales, discount coupons, and value essentials.
# 
# #### 2. Limitations We Encountered
# - **Spherical cluster assumption:** K-Means assumes clusters are round and similar in spread. It would struggle if the data formed elongated or crescent shapes.
# - **Sensitivity to scaling:** Euclidean distance depends entirely on how features are scaled.
# - **Hierarchical clustering cannot undo steps:** Once two customers or groups are merged in agglomerative clustering, they stay together forever.
# - **Sample size:** With only 200 customers, our findings should be tested on larger mall datasets before making huge commercial investments.
# 
# #### 3. Proposed Improvement: K-Prototypes for Mixed Data
# In standard K-Means, we had to leave out `Gender` because turning a category into 0/1 distorts Euclidean distance. To fix this, we recommend using **K-Prototypes** (Huang, 1998) or **PAM (Partitioning Around Medoids) with Gower's distance**.  
# K-Prototypes calculates distance using Euclidean distance for numbers and simple matching for categories:
# $$E = \sum \text{squared distance (numeric)} + \gamma \sum \text{matching mismatch (categorical)}$$
# This would let us cluster on income, spending, age, and gender together naturally without awkward dummy encoding.
# 
# #### 4. Cluster Stability Finding
# In our 50-seed test:
# - With `n_init=10`, K-Means was **100% stable** (Silhouette score of 0.5547 every single run, std = 0.0000).
# - With `n_init=1`, it varied widely (mean 0.5386, dropping down to 0.4101 in some runs).  
# This proves why using multiple starts (`n_init >= 10`) is essential so the algorithm doesn't get stuck in a bad local minimum.
# 
# ---
# **Assignment completed.**
