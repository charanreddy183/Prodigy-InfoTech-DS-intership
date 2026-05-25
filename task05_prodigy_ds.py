import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

np.random.seed(42)
n = 5000

# ── 1. Generate realistic US accident dataset ─────────────────────────────────
hours = np.random.choice(range(24), n, p=[
    0.0202,0.0152,0.0101,0.0081,0.0081,0.0121,0.0304,0.0709,0.0810,0.0506,
    0.0405,0.0405,0.0455,0.0455,0.0506,0.0557,0.0658,0.0759,0.0709,0.0557,
    0.0455,0.0405,0.0354,0.0253])

weather = np.random.choice(
    ['Clear','Cloudy','Rain','Fog','Snow','Windy'],
    n, p=[0.45, 0.25, 0.15, 0.07, 0.05, 0.03])

road_condition = np.random.choice(
    ['Dry','Wet','Snow/Ice','Muddy'],
    n, p=[0.60, 0.25, 0.10, 0.05])

severity = []
for i in range(n):
    base = 1.5
    if weather[i] in ['Rain','Fog']: base += 0.5
    if weather[i] in ['Snow','Fog']: base += 0.8
    if road_condition[i] == 'Wet': base += 0.3
    if road_condition[i] == 'Snow/Ice': base += 1.0
    if hours[i] in [0,1,2,3,22,23]: base += 0.4
    if hours[i] in [7,8,17,18]: base += 0.2
    sev = int(np.clip(np.random.normal(base, 0.6), 1, 4))
    severity.append(sev)

states = np.random.choice(
    ['CA','TX','FL','NY','PA','OH','IL','GA','NC','MI'],
    n, p=[0.18,0.15,0.12,0.09,0.07,0.07,0.07,0.07,0.09,0.09])

day_of_week = np.random.choice(
    ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
    n, p=[0.14,0.14,0.14,0.14,0.16,0.14,0.14])

light = np.where(
    (np.array(hours) >= 6) & (np.array(hours) <= 19),
    'Daylight', np.where(
    (np.array(hours) >= 20) | (np.array(hours) <= 5),
    'Night', 'Dawn/Dusk'))

months = np.random.choice(range(1,13), n, p=[
    0.09,0.08,0.08,0.08,0.08,0.07,0.07,0.07,0.08,0.09,0.10,0.11])
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

df = pd.DataFrame({
    'hour': hours, 'weather': weather, 'road_condition': road_condition,
    'severity': severity, 'state': states, 'day_of_week': day_of_week,
    'light_condition': light, 'month': months
})

print(f"Dataset: {len(df)} accidents")
print(df['severity'].value_counts().sort_index())

# ── 2. Visualization ──────────────────────────────────────────────────────────
BG='#0f0f1a'; PANEL='#1a1a2e'; BORDER='#2a2a4a'
WHITE='#e8e8f0'; MUTED='#8080a0'
PURPLE='#7c6af7'; PINK='#f472b6'; GREEN='#34d399'; AMBER='#f59e0b'; BLUE='#60a5fa'
SEV_COLORS = {1: GREEN, 2: AMBER, 3: PINK, 4: '#ff4444'}

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle('US Traffic Accident Analysis — Patterns & Contributing Factors',
             color=WHITE, fontsize=17, fontweight='bold', y=0.98)
fig.text(0.5, 0.955, 'Prodigy InfoTech | Data Science Internship | Task 05',
         ha='center', color=MUTED, fontsize=10)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# Plot 1: Accidents by Hour of Day
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, '1. Accidents by Hour of Day')
hourly = df.groupby('hour').size()
colors_h = [PINK if h in [7,8,17,18,19] else PURPLE if h in [0,1,2,3,22,23] else BLUE
            for h in hourly.index]
ax1.bar(hourly.index, hourly.values, color=colors_h, edgecolor=BG, width=0.8)
ax1.set_xlabel('Hour of Day', color=MUTED, fontsize=9)
ax1.set_ylabel('Accidents', color=MUTED, fontsize=9)
ax1.set_xticks(range(0,24,3))
rush = mpatches.Patch(color=PINK, label='Rush Hour')
night = mpatches.Patch(color=PURPLE, label='Night')
other = mpatches.Patch(color=BLUE, label='Regular')
ax1.legend(handles=[rush, night, other], facecolor=PANEL,
           edgecolor=BORDER, labelcolor=WHITE, fontsize=7)

# Plot 2: Accidents by Weather
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, '2. Accidents by Weather Condition')
weath = df.groupby('weather').size().sort_values(ascending=False)
w_colors = [GREEN,BLUE,AMBER,PURPLE,PINK,'#ff4444'][:len(weath)]
bars = ax2.bar(weath.index, weath.values, color=w_colors, edgecolor=BG, width=0.6)
for bar, val in zip(bars, weath.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
             str(val), ha='center', color=WHITE, fontsize=8, fontweight='bold')
ax2.set_ylabel('Accidents', color=MUTED, fontsize=9)
ax2.tick_params(axis='x', labelsize=8)

# Plot 3: Severity Distribution
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, '3. Accident Severity Distribution')
sev = df['severity'].value_counts().sort_index()
sev_labels = ['Mild\n(1)', 'Moderate\n(2)', 'Serious\n(3)', 'Critical\n(4)']
sev_c = [SEV_COLORS[i] for i in sev.index]
bars = ax3.bar(sev_labels[:len(sev)], sev.values, color=sev_c, edgecolor=BG, width=0.55)
for bar, val in zip(bars, sev.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
             f'{val}\n({val/n*100:.1f}%)', ha='center', color=WHITE,
             fontsize=8, fontweight='bold')
ax3.set_ylabel('Count', color=MUTED, fontsize=9)

# Plot 4: Heatmap - Hour vs Day of Week
ax4 = fig.add_subplot(gs[1, 0:2])
style_ax(ax4, '4. Accident Heatmap — Hour vs Day of Week')
days_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
pivot = df.groupby(['day_of_week','hour']).size().unstack(fill_value=0)
pivot = pivot.reindex(days_order)
cmap = LinearSegmentedColormap.from_list('custom', ['#1a1a2e','#7c6af7','#f472b6','#ff4444'])
im = ax4.imshow(pivot.values, aspect='auto', cmap=cmap, interpolation='nearest')
ax4.set_yticks(range(len(days_order)))
ax4.set_yticklabels(days_order, color=MUTED, fontsize=8)
ax4.set_xticks(range(0,24,2))
ax4.set_xticklabels(range(0,24,2), color=MUTED, fontsize=7)
ax4.set_xlabel('Hour of Day', color=MUTED, fontsize=9)
plt.colorbar(im, ax=ax4, label='Accident Count').ax.yaxis.label.set_color(MUTED)

# Plot 5: Road condition vs avg severity + monthly trend
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, '5. Monthly Accident Trend')
monthly = df.groupby('month').size()
ax5.plot([month_names[m-1] for m in monthly.index], monthly.values,
         color=PURPLE, marker='o', linewidth=2.5, markersize=6, markerfacecolor=AMBER)
ax5.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color=PURPLE)
ax5.set_xticks(range(len(monthly)))
ax5.set_xticklabels([month_names[m-1] for m in monthly.index],
                     rotation=45, fontsize=7, color=MUTED)
ax5.set_ylabel('Accidents', color=MUTED, fontsize=9)
ax5.set_xlabel('Month', color=MUTED, fontsize=9)

plt.savefig('/mnt/user-data/outputs/prodigy_ds_task05.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
print("Done! Saved.")
