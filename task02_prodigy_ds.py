import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from io import StringIO

# ── 1. Create realistic Titanic-like dataset ─────────────────────────────────
np.random.seed(42)
n = 891

pclass = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
sex = np.random.choice(['male', 'female'], size=n, p=[0.65, 0.35])

ages = []
for p in pclass:
    if p == 1: ages.append(np.random.normal(38, 14))
    elif p == 2: ages.append(np.random.normal(30, 12))
    else: ages.append(np.random.normal(25, 12))
age = np.clip(ages, 1, 80)
age_with_nan = age.copy()
age_with_nan[np.random.choice(n, 177, replace=False)] = np.nan

fare_base = {1: 84, 2: 20, 3: 13}
fare = np.array([np.random.exponential(fare_base[p]) for p in pclass])

# survival based on class and sex
surv_prob = []
for i in range(n):
    if sex[i] == 'female':
        p = [0.97, 0.92, 0.50][pclass[i]-1]
    else:
        p = [0.37, 0.16, 0.14][pclass[i]-1]
    surv_prob.append(p)
survived = np.array([np.random.binomial(1, p) for p in surv_prob])

embarked = np.random.choice(['S', 'C', 'Q', None], size=n, p=[0.72, 0.19, 0.086, 0.004])
sibsp = np.random.choice([0,1,2,3,4,5], size=n, p=[0.68,0.23,0.05,0.02,0.01,0.01])
parch = np.random.choice([0,1,2,3], size=n, p=[0.76,0.13,0.09,0.02])

df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived': survived,
    'Pclass': pclass,
    'Sex': sex,
    'Age': age_with_nan,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
})

# ── 2. Data Cleaning ──────────────────────────────────────────────────────────
print("=== BEFORE CLEANING ===")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}\n")

df['Age'] = df['Age'].fillna(df.groupby(['Pclass','Sex'])['Age'].transform('median'))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                         labels=['Child','Teen','Young Adult','Adult','Senior'])

print("=== AFTER CLEANING ===")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}\n")
print(f"Survival Rate: {df['Survived'].mean()*100:.1f}%")

# ── 3. Visualization ──────────────────────────────────────────────────────────
BG = '#0f0f1a'
PANEL = '#1a1a2e'
BORDER = '#2a2a4a'
WHITE = '#e8e8f0'
MUTED = '#8080a0'
PURPLE = '#7c6af7'
PINK = '#f472b6'
GREEN = '#34d399'
AMBER = '#f59e0b'

surv_colors = [PINK, GREEN]

fig = plt.figure(figsize=(16, 12), facecolor=BG)
fig.suptitle('Titanic Dataset — Data Cleaning & Exploratory Data Analysis',
             color=WHITE, fontsize=17, fontweight='bold', y=0.98)
fig.text(0.5, 0.955, 'Prodigy InfoTech | Data Science Internship | Task 02',
         ha='center', color=MUTED, fontsize=10)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# Plot 1: Survival Count
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, '1. Survival Count')
sv = df['Survived'].value_counts()
bars = ax1.bar(['Not Survived', 'Survived'], sv.values, color=[PINK, GREEN],
               edgecolor=BG, width=0.5)
for bar, val in zip(bars, sv.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(val),
             ha='center', color=WHITE, fontsize=10, fontweight='bold')
ax1.set_ylabel('Count', color=MUTED, fontsize=9)
ax1.set_ylim(0, max(sv.values)*1.15)

# Plot 2: Survival by Sex
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, '2. Survival by Gender')
sex_surv = df.groupby('Sex')['Survived'].mean() * 100
bars = ax2.bar(sex_surv.index, sex_surv.values, color=[PURPLE, AMBER], width=0.4, edgecolor=BG)
for bar, val in zip(bars, sex_surv.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}%', ha='center', color=WHITE, fontsize=10, fontweight='bold')
ax2.set_ylabel('Survival Rate (%)', color=MUTED, fontsize=9)
ax2.set_ylim(0, 110)

# Plot 3: Survival by Class
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, '3. Survival by Class')
cls_surv = df.groupby('Pclass')['Survived'].mean() * 100
bars = ax3.bar(['1st Class', '2nd Class', '3rd Class'], cls_surv.values,
               color=[GREEN, AMBER, PINK], width=0.5, edgecolor=BG)
for bar, val in zip(bars, cls_surv.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}%', ha='center', color=WHITE, fontsize=10, fontweight='bold')
ax3.set_ylabel('Survival Rate (%)', color=MUTED, fontsize=9)
ax3.set_ylim(0, 110)

# Plot 4: Age Distribution by Survival
ax4 = fig.add_subplot(gs[1, :2])
style_ax(ax4, '4. Age Distribution by Survival Status')
for surv, color, label in [(0, PINK, 'Not Survived'), (1, GREEN, 'Survived')]:
    data = df[df['Survived']==surv]['Age'].dropna()
    ax4.hist(data, bins=25, alpha=0.65, color=color, label=label, edgecolor=BG)
ax4.axvline(df['Age'].mean(), color=AMBER, linestyle='--', linewidth=1.5,
            label=f'Mean Age: {df["Age"].mean():.1f}')
ax4.set_xlabel('Age', color=MUTED, fontsize=9)
ax4.set_ylabel('Count', color=MUTED, fontsize=9)
ax4.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9)

# Plot 5: Fare Distribution
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, '5. Fare Distribution by Class')
colors_cls = [GREEN, AMBER, PINK]
for i, cls in enumerate([1,2,3]):
    data = df[df['Pclass']==cls]['Fare']
    ax5.hist(data, bins=20, alpha=0.7, color=colors_cls[i],
             label=f'Class {cls}', edgecolor=BG)
ax5.set_xlabel('Fare (£)', color=MUTED, fontsize=9)
ax5.set_ylabel('Count', color=MUTED, fontsize=9)
ax5.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9)
ax5.set_xlim(0, 300)

# Plot 6: Survival by Age Group
ax6 = fig.add_subplot(gs[2, 0])
style_ax(ax6, '6. Survival by Age Group')
age_surv = df.groupby('AgeGroup', observed=True)['Survived'].mean() * 100
bars = ax6.bar(age_surv.index, age_surv.values,
               color=[PURPLE, AMBER, GREEN, PINK, '#60a5fa'], edgecolor=BG, width=0.6)
for bar, val in zip(bars, age_surv.values):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.0f}%', ha='center', color=WHITE, fontsize=9, fontweight='bold')
ax6.set_ylabel('Survival Rate (%)', color=MUTED, fontsize=9)
ax6.tick_params(axis='x', labelsize=7)
ax6.set_ylim(0, 110)

# Plot 7: Family Size vs Survival
ax7 = fig.add_subplot(gs[2, 1])
style_ax(ax7, '7. Family Size vs Survival Rate')
fam_surv = df.groupby('FamilySize')['Survived'].mean() * 100
ax7.plot(fam_surv.index, fam_surv.values, color=PURPLE, marker='o',
         linewidth=2, markersize=7, markerfacecolor=AMBER)
ax7.fill_between(fam_surv.index, fam_surv.values, alpha=0.15, color=PURPLE)
ax7.set_xlabel('Family Size', color=MUTED, fontsize=9)
ax7.set_ylabel('Survival Rate (%)', color=MUTED, fontsize=9)
ax7.set_ylim(0, 110)

# Plot 8: Missing values heatmap / correlation
ax8 = fig.add_subplot(gs[2, 2])
style_ax(ax8, '8. Correlation Heatmap')
num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
corr = df[num_cols].corr()
im = ax8.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax8.set_xticks(range(len(num_cols)))
ax8.set_yticks(range(len(num_cols)))
short = ['Surv', 'Cls', 'Age', 'Sib', 'Par', 'Fare', 'Fam']
ax8.set_xticklabels(short, color=MUTED, fontsize=7, rotation=45)
ax8.set_yticklabels(short, color=MUTED, fontsize=7)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax8.text(j, i, f'{corr.values[i,j]:.1f}', ha='center', va='center',
                color='black' if abs(corr.values[i,j]) < 0.5 else 'white', fontsize=7)

plt.savefig('/mnt/user-data/outputs/prodigy_ds_task02.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
print("Done! Saved.")
