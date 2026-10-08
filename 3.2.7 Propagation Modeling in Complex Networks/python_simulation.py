import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Patch

N_NODES = 100
K_NEIGHBORS = 6
REWIRING_PROB = 0.1
ITERATIONS = 50
INFECTION_PROB = 0.3
RECOVERY_PROB = 0.1

np.random.seed(42)

G = nx.watts_strogatz_graph(
    N_NODES, K_NEIGHBORS, REWIRING_PROB, seed=42
)

states = np.zeros(N_NODES, dtype=int)
states[np.random.randint(N_NODES)] = 1

pos = nx.spring_layout(G, seed=42)

# Filas: 50 iteraciones. Columnas: S, I, R.
metrics = np.zeros((ITERATIONS, 3), dtype=int)

# Evita ejecutar dos veces una misma actualización.
processed = np.zeros(ITERATIONS, dtype=bool)
history = np.zeros((ITERATIONS, N_NODES), dtype=int)
chart_created = False

colors = {
    0: "#2ecc71",
    1: "#e74c3c",
    2: "#95a5a6"
}

fig, ax = plt.subplots(figsize=(9, 7))


def draw_network(current_states, title):
    ax.clear()

    nx.draw_networkx(
        G,
        pos,
        node_color=[colors[s] for s in current_states],
        node_size=100,
        edge_color="lightgray",
        with_labels=False,
        ax=ax
    )

    ax.set_title(title)
    ax.legend(handles=[
        Patch(color=colors[0], label="Susceptibles"),
        Patch(color=colors[1], label="Infectados"),
        Patch(color=colors[2], label="Recuperados")
    ])
    ax.axis("off")


def initialize_animation():
    draw_network(states, "SIR — Estado inicial")


def update(frame):
    global chart_created

    if not processed[frame]:
        new_states = states.copy()

        for node in G.nodes():
            if states[node] == 1:
                for neighbor in G.neighbors(node):
                    if (
                        states[neighbor] == 0
                        and np.random.rand() < INFECTION_PROB
                    ):
                        new_states[neighbor] = 1

                if np.random.rand() < RECOVERY_PROB:
                    new_states[node] = 2

        states[:] = new_states

        # Registro después de cada actualización.
        metrics[frame] = np.bincount(states, minlength=3)
        history[frame] = states
        processed[frame] = True

        assert metrics[frame].sum() == N_NODES

    s, i, r = metrics[frame]

    draw_network(
        history[frame],
        f"SIR — Iteración {frame + 1}/{ITERATIONS}\n"
        f"S: {s} | I: {i} | R: {r}"
    )

    if frame == ITERATIONS - 1 and not chart_created:
        chart_created = True

        fig.savefig(
            "animacion_final_sir.png",
            dpi=200,
            bbox_inches="tight"
        )
        np.save("metricas_sir.npy", metrics)

        fig_curves, ax_curves = plt.subplots(figsize=(9, 5))
        steps = np.arange(1, ITERATIONS + 1)

        labels = [
            "Susceptibles (S)",
            "Infectados (I)",
            "Recuperados (R)"
        ]

        for column, label in enumerate(labels):
            ax_curves.plot(
                steps,
                metrics[:, column],
                color=colors[column],
                label=label,
                linewidth=2
            )

        ax_curves.set_title("Evolución temporal SIR")
        ax_curves.set_xlabel("Iteración")
        ax_curves.set_ylabel("Número de nodos")
        ax_curves.set_ylim(0, N_NODES)
        ax_curves.grid(alpha=0.3)
        ax_curves.legend()
        fig_curves.tight_layout()

        fig_curves.savefig("curvas_sir.png", dpi=200)
        fig_curves.show()

        print("Simulación terminada.")
        print("Dimensiones de las métricas:", metrics.shape)
        print(f"Estado final: S={s}, I={i}, R={r}")


ani = animation.FuncAnimation(
    fig,
    update,
    init_func=initialize_animation,
    frames=ITERATIONS,
    interval=300,
    repeat=False,
    blit=False,
    cache_frame_data=False
)

plt.show()