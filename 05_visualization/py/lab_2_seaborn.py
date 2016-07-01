# lab 2 - seaborn

# 1. Regression plot
fig, ax = plt.subplots(figsize=(6,6))
sns.lmplot(x="total_bill", y="tip", data=tips, ax = ax);
ax.set_title('Tip vs Total Bill')
fig.savefig('seaborn_01.png')

# 2. Regression plot categorical value, jitter
fig, ax = plt.subplots(figsize=(6,6))
sns.regplot(x="size", y="tip", data=tips, ax = ax, x_jitter=.1);
ax.set_title('Tip vs Size')
fig.savefig('seaborn_02.png')

# 3.

fig, ax = plt.subplots(figsize=(6,6))
tips["big_tip"] = (tips.tip / tips.total_bill) > .15
sns.lmplot(x="total_bill", y="big_tip", data=tips, y_jitter=.03);
fig.savefig('seaborn_03.png')

# 3.1

fig, ax = plt.subplots(figsize=(6,6))
tips["big_tip"] = (tips.tip / tips.total_bill) > .15
sns.regplot(x="total_bill", y="big_tip", data=tips, logistic=True, y_jitter=.03);
fig.savefig('seaborn_031.png')

# 7.
fig, ax = plt.subplots(1,2,  figsize=(18,9))

sns.boxplot(x="day", y="total_bill", hue='time', data=tips[tips.sex=='Female'], ax = ax[0]);
ax[0].set_title('Tips from Women')
sns.boxplot(x="day", y="total_bill", hue='time', data=tips[tips.sex=='Male'], ax = ax[1]);
ax[1].set_title('Tips from Men')
fig.savefig('seaborn_07.png')