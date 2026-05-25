import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
n = 4521

age = np.random.randint(18, 70, n)
job = np.random.choice(['admin','technician','services','management',
                        'retired','student','unemployed'], n)
marital = np.random.choice(['married','single','divorced'], n, p=[0.56,0.32,0.12])
education = np.random.choice(['primary','secondary','tertiary'], n, p=[0.15,0.52,0.33])
balance = np.random.normal(1362, 3000, n).astype(int)
duration = np.random.exponential(250, n).astype(int)
campaign = np.random.randint(1, 10, n)
poutcome = np.random.choice(['success','failure','unknown'], n, p=[0.11,0.39,0.50])

prob = (
    0.08 +
    (age > 55) * 0.12 +
    (education == 'tertiary') * 0.10 +
    (duration > 300) * 0.25 +
    (balance > 2000) * 0.10 +
    (poutcome == 'success') * 0.30 +
    (marital == 'single') * 0.05 -
    (campaign > 5) * 0.05
).clip(0.02, 0.95)

y = np.array([np.random.binomial(1, p) for p in prob])

df = pd.DataFrame({
    'age': age, 'job': job, 'marital': marital,
    'education': education, 'balance': balance,
    'duration': duration, 'campaign': campaign,
    'poutcome': poutcome, 'y': y
})

# Encode
le = LabelEncoder()
df_enc = df.copy()
for col in ['job','marital','education','poutcome']:
    df_enc[col] = le.fit_transform(df_enc[col])

X = df_enc.drop('y', axis=1)
y = df_enc['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, min_samples_split=50,
                              min_samples_leaf=20, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['No','Yes']))

# ── Visualization ────────────────────────────────────────────────────────────
BG='#0f0f1a'; PANEL='#1a1a2e'; BORDER='#2a2a4a'
WHITE='#e8e8f0'; MUTED='#8080a0'
PURPLE='#7c6af7'; PINK='#f472b6'; GREEN='#34d399'; AMBER='#f59e0b'

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle('Decision Tree Classifier — Bank Marketing Dataset',
             color=WHITE, fontsize=17, fontweight='bold', y=0.98)
fig.text(0.5, 0.955, 'Prodigy InfoTech | Data Science Internship | Task 03',
         ha='center', color=MUTED, fontsize=10)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# Plot 1: Decision Tree
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(PANEL)
ax1.set_title('1. Decision Tree Structure (max_depth=4)', color=WHITE,
              fontsize=11, fontweight='bold', pad=8)
for spine in ax1.spines.values():
    spine.set_edgecolor(BORDER)
plot_tree(clf, feature_names=X.columns, class_names=['No','Yes'],
          filled=True, rounded=True, fontsize=6, ax=ax1,
          impurity=False, proportion=False)

# Plot 2: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL)
ax2.set_title('2. Confusion Matrix', color=WHITE, fontsize=11, fontweight='bold', pad=8)
cm = confusion_matrix(y_test, y_pred)
im = ax2.imshow(cm, cmap='Purples')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i,j]), ha='center', va='center',
                color=WHITE, fontsize=14, fontweight='bold')
ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
ax2.set_xticklabels(['Predicted No','Predicted Yes'], color=MUTED, fontsize=8)
ax2.set_yticklabels(['Actual No','Actual Yes'], color=MUTED, fontsize=8)

# Plot 3: Feature Importance
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, '3. Feature Importance')
fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
colors = [PURPLE if v < 0.1 else AMBER if v < 0.2 else GREEN for v in fi.values]
bars = ax3.barh(fi.index, fi.values, color=colors, edgecolor=BG, height=0.6)
ax3.set_xlabel('Importance', color=MUTED, fontsize=9)
ax3.xaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
ax3.yaxis.grid(False)

# Plot 4: Accuracy metrics bar
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, '4. Model Performance Metrics')
report = classification_report(y_test, y_pred, target_names=['No','Yes'], output_dict=True)
metrics = ['Precision\n(No)', 'Recall\n(No)', 'Precision\n(Yes)', 'Recall\n(Yes)', 'Accuracy']
values = [report['No']['precision'], report['No']['recall'],
          report['Yes']['precision'], report['Yes']['recall'], acc]
bar_colors = [PURPLE, PURPLE, GREEN, GREEN, AMBER]
bars = ax4.bar(metrics, [v*100 for v in values], color=bar_colors, edgecolor=BG, width=0.5)
for bar, val in zip(bars, values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val*100:.1f}%', ha='center', color=WHITE, fontsize=9, fontweight='bold')
ax4.set_ylabel('Score (%)', color=MUTED, fontsize=9)
ax4.set_ylim(0, 115)
ax4.tick_params(axis='x', labelsize=7)

# Plot 5: Purchase rate by duration bucket
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, '5. Purchase Rate by Call Duration')
df['dur_bucket'] = pd.cut(df['duration'], bins=[0,100,200,300,500,2000],
                           labels=['<100s','100-200s','200-300s','300-500s','>500s'])
dur_rate = df.groupby('dur_bucket', observed=True)['y'].mean() * 100
bars = ax5.bar(dur_rate.index, dur_rate.values,
               color=[PINK, PURPLE, AMBER, GREEN, '#60a5fa'], edgecolor=BG, width=0.6)
for bar, val in zip(bars, dur_rate.values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}%', ha='center', color=WHITE, fontsize=9, fontweight='bold')
ax5.set_ylabel('Purchase Rate (%)', color=MUTED, fontsize=9)
ax5.set_xlabel('Call Duration', color=MUTED, fontsize=9)
ax5.tick_params(axis='x', labelsize=8)

plt.savefig('/mnt/user-data/outputs/prodigy_ds_task03.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
print("Saved!")
