import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import re
from collections import Counter

np.random.seed(42)

# ── 1. Simulate realistic social media dataset ────────────────────────────────
topics = ['iPhone 15', 'Tesla', 'ChatGPT', 'Netflix', 'Twitter/X']

positive_words = ['amazing', 'love', 'awesome', 'great', 'excellent', 'fantastic',
    'wonderful', 'best', 'brilliant', 'happy', 'excited', 'perfect', 'incredible',
    'outstanding', 'superb', 'delightful', 'impressive', 'phenomenal', 'satisfied']

negative_words = ['terrible', 'hate', 'awful', 'worst', 'horrible', 'disgusting',
    'disappointing', 'boring', 'useless', 'broken', 'frustrated', 'annoying',
    'pathetic', 'overpriced', 'garbage', 'failed', 'poor', 'disaster', 'regret']

neutral_words = ['okay', 'average', 'decent', 'normal', 'standard', 'regular',
    'fine', 'acceptable', 'moderate', 'typical', 'basic', 'usual', 'ordinary']

positive_templates = [
    "This {topic} is absolutely {w1}! {w2} experience ever.",
    "Just got the new {topic} and it's {w1}. Totally {w2}!",
    "The {topic} update is {w1}. Feeling {w2} about this!",
    "{topic} never disappoints. {w1} and {w2} as always.",
    "Wow {topic} is {w1}! Best decision ever. So {w2}!"
]
negative_templates = [
    "The {topic} is absolutely {w1}. Such a {w2} product.",
    "Regretting buying {topic}. It's {w1} and {w2}.",
    "{topic} keeps crashing. This is {w1} and {w2}!",
    "Never buying {topic} again. {w1} quality, so {w2}.",
    "My {topic} broke already. {w1} experience, totally {w2}."
]
neutral_templates = [
    "{topic} is {w1}. Nothing special, pretty {w2}.",
    "Using {topic} for a week. It's {w1}, feels {w2}.",
    "{topic} works {w1}. Not bad, not great. Quite {w2}.",
    "Tried {topic} today. It's {w1} at best. Very {w2}.",
    "My thoughts on {topic}: {w1} design, {w2} performance."
]

records = []
for topic in topics:
    n_pos = np.random.randint(180, 260)
    n_neg = np.random.randint(80, 150)
    n_neu = np.random.randint(60, 120)
    for sentiment, templates, words, count in [
        ('Positive', positive_templates, positive_words, n_pos),
        ('Negative', negative_templates, negative_words, n_neg),
        ('Neutral',  neutral_templates,  neutral_words,  n_neu)
    ]:
        for _ in range(count):
            w1, w2 = np.random.choice(words, 2, replace=False)
            tmpl = np.random.choice(templates)
            text = tmpl.format(topic=topic, w1=w1, w2=w2)
            likes = (np.random.poisson(80) if sentiment=='Positive'
                     else np.random.poisson(30) if sentiment=='Negative'
                     else np.random.poisson(20))
            records.append({
                'text': text, 'topic': topic,
                'sentiment': sentiment, 'likes': likes,
                'retweets': max(0, likes // 4 + np.random.randint(-5, 10)),
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 90))
            })

df = pd.DataFrame(records)
df['month'] = df['date'].dt.to_period('M')

print(f"Dataset: {len(df)} tweets | Topics: {df['topic'].nunique()}")
print(df['sentiment'].value_counts())

# ── 2. Word frequency per sentiment ──────────────────────────────────────────
def get_top_words(df_sub, n=15):
    all_text = ' '.join(df_sub['text'].str.lower())
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    stop = {'this','that','with','have','just','never','such','been','about',
            'best','very','totally','keeps','tried','using','week','thoughts',
            'works','feels','broke','already','buying','again','always','only'}
    words = [w for w in words if w not in stop]
    return Counter(words).most_common(n)

# ── 3. Visualization ──────────────────────────────────────────────────────────
BG='#0f0f1a'; PANEL='#1a1a2e'; BORDER='#2a2a4a'
WHITE='#e8e8f0'; MUTED='#8080a0'
PURPLE='#7c6af7'; PINK='#f472b6'; GREEN='#34d399'; AMBER='#f59e0b'; BLUE='#60a5fa'
SENT_COLORS = {'Positive': GREEN, 'Negative': PINK, 'Neutral': AMBER}

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle('Social Media Sentiment Analysis — Brand & Topic Monitoring',
             color=WHITE, fontsize=17, fontweight='bold', y=0.98)
fig.text(0.5, 0.955, 'Prodigy InfoTech | Data Science Internship | Task 04',
         ha='center', color=MUTED, fontsize=10)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.yaxis.grid(True, color=BORDER, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# Plot 1: Overall Sentiment Distribution (Donut)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
ax1.set_title('1. Overall Sentiment Distribution', color=WHITE, fontsize=11, fontweight='bold', pad=8)
for spine in ax1.spines.values(): spine.set_edgecolor(BORDER)
sv = df['sentiment'].value_counts()
wedge_colors = [SENT_COLORS[s] for s in sv.index]
wedges, texts, autotexts = ax1.pie(
    sv.values, labels=sv.index, colors=wedge_colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
    textprops=dict(color=WHITE, fontsize=9))
for at in autotexts: at.set_color(BG); at.set_fontweight('bold'); at.set_fontsize(8)
ax1.text(0, 0, f'{len(df)}\ntweets', ha='center', va='center',
         color=WHITE, fontsize=10, fontweight='bold')

# Plot 2: Sentiment by Topic (stacked bar)
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, '2. Sentiment by Topic')
topic_sent = df.groupby(['topic','sentiment']).size().unstack(fill_value=0)
bottom = np.zeros(len(topic_sent))
for sent in ['Positive','Neutral','Negative']:
    if sent in topic_sent.columns:
        vals = topic_sent[sent].values
        ax2.bar(topic_sent.index, vals, bottom=bottom,
                color=SENT_COLORS[sent], label=sent, edgecolor=BG, linewidth=0.5)
        bottom += vals
ax2.set_ylabel('Tweet Count', color=MUTED, fontsize=9)
ax2.tick_params(axis='x', labelsize=7, rotation=15)
ax2.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8, loc='upper right')

# Plot 3: Sentiment trend over time
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, '3. Sentiment Trend Over Time')
trend = df.groupby(['month','sentiment']).size().unstack(fill_value=0)
months_str = [str(m) for m in trend.index]
for sent in ['Positive','Neutral','Negative']:
    if sent in trend.columns:
        ax3.plot(months_str, trend[sent].values, color=SENT_COLORS[sent],
                 marker='o', linewidth=2, markersize=5, label=sent)
ax3.set_ylabel('Tweet Count', color=MUTED, fontsize=9)
ax3.tick_params(axis='x', labelsize=7, rotation=20)
ax3.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)

# Plot 4: Top positive words (manual bar chart as word cloud substitute)
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, '4. Top Positive Keywords')
pos_words = get_top_words(df[df['sentiment']=='Positive'], 10)
words, counts = zip(*pos_words)
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(words)))
ax4.barh(words, counts, color=colors, edgecolor=BG)
ax4.set_xlabel('Frequency', color=MUTED, fontsize=9)
ax4.yaxis.grid(False)
ax4.invert_yaxis()

# Plot 5: Top negative words
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, '5. Top Negative Keywords')
neg_words = get_top_words(df[df['sentiment']=='Negative'], 10)
words, counts = zip(*neg_words)
colors = plt.cm.RdPu(np.linspace(0.4, 0.9, len(words)))
ax5.barh(words, counts, color=colors, edgecolor=BG)
ax5.set_xlabel('Frequency', color=MUTED, fontsize=9)
ax5.yaxis.grid(False)
ax5.invert_yaxis()

# Plot 6: Avg likes by sentiment per topic
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, '6. Avg Engagement (Likes) by Sentiment')
eng = df.groupby(['topic','sentiment'])['likes'].mean().unstack()
x = np.arange(len(eng.index))
w = 0.25
for i, sent in enumerate(['Positive','Neutral','Negative']):
    if sent in eng.columns:
        bars = ax6.bar(x + i*w, eng[sent].values, width=w,
                       color=SENT_COLORS[sent], label=sent, edgecolor=BG, alpha=0.9)
ax6.set_xticks(x + w)
ax6.set_xticklabels(eng.index, rotation=15, fontsize=7)
ax6.set_ylabel('Avg Likes', color=MUTED, fontsize=9)
ax6.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=8)

plt.savefig('/mnt/user-data/outputs/prodigy_ds_task04.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
print("Done! Saved.")
