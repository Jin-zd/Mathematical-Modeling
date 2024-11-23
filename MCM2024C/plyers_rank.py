import matplotx
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('out/set1_p1.csv')
x = data['p1_point']
y1 = data['p1_score']
y2 = data['sub1_mom']

with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, label='Score', color='#A07EE7')
    ax1.set_xlabel('Points')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, 100])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Momentum')
    ax2.plot(x, y2, label='Momentum', c='#3FEDCF')
    ax2.set_ylim([0, 1])

    ax2.legend()
    plt.savefig('out/set31_p1.png', dpi=300)