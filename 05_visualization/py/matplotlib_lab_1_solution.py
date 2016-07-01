fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

df[['Age','Embarked']].boxplot(by = 'Embarked', ax = ax1, showmeans= True )
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.set_axis_bgcolor('white')
ax1.set_ylabel('Age')
ax1.set_xlabel('')
ax1.set_title('Port of embarkation')
ax1.grid(False)
ax1.set_ylim(-5,100)

df[['Age','Pclass']].boxplot(by = 'Pclass', ax = ax2, showmeans= True )
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none')
ax2.set_axis_bgcolor('white')
ax2.set_ylim(-5,100)

plt.setp(ax2.get_yticklabels(), visible=False)
ax2.set_xlabel('')
ax2.set_title('Class')
ax2.grid(False)

plt.tight_layout(pad=1.0)
fig.savefig('age_in_titanic.png')
