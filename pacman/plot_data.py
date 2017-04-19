import matplotlib.pyplot as plt


def plot(df, room, usage):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_ylim(-0.1,1.1)
    df["int_temperature"].plot(ax=ax1, color='blue')
    df["status_est"].plot(ax=ax2)
    plt.savefig("../Results/Estimation/Figures/Estimated/PT_" + room + "_" + str(usage) + ".png")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_ylim(-0.1,1.1)
    df["power"].plot(ax=ax1, color='blue')
    df["status_est"].plot(ax=ax2)
    plt.savefig("../Results/Estimation/Figures/Estimated/PS_" + room + "_" + str(usage) + ".png")
