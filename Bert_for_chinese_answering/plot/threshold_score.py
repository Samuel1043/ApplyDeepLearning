import json
import matplotlib.pyplot as plt
import sys

threshold=[0.1,0.3,0.5,0.7,0.9]
overall=[]
answerable=[]
unanswerable=[]

overall_f1=[]
answerable_f1=[]
unanswerable_f1=[]

for i in threshold: 
    with open(sys.argv[1]+'performance_max30_able'+str(i)+'.json','r') as  r:
        line=json.loads(r.readline())
        overall.append(line['overall']['em'])
        answerable.append(line['answerable']['em'])
        unanswerable.append(line['unanswerable']['em'])
        overall_f1.append(line['overall']['f1'])
        answerable_f1.append(line['answerable']['f1'])
        unanswerable_f1.append(line['unanswerable']['f1'])

        
fig=plt.figure(figsize=(10,6))

fig.suptitle('performance on different threshold')
fig.text(0.5, 0.04, 'Answerable threshold', ha='center')
ax1=plt.subplot(1,2,1)
plt.title('EM')
plt.plot(range(len(threshold)),overall,'o-',label='overall')
plt.plot(range(len(threshold)),answerable,'o-',label='answerable')
plt.plot(range(len(threshold)),unanswerable,'o-',label='unanswerable')



plt.xticks(range(len(threshold)),threshold)
plt.subplot(1,2,2,sharey=ax1)
plt.title('F1')
p1=plt.plot(range(len(threshold)),overall_f1,'o-')
p2=plt.plot(range(len(threshold)),answerable_f1,'o-')
p3=plt.plot(range(len(threshold)),unanswerable_f1,'o-')
plt.xticks(range(len(threshold)),threshold)


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.savefig(sys.argv[2])