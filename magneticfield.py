import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

# Rozmiar obrazu
width, height = 500, 300


# Wczytanie wygenerowanego obrazu pola magnetycznego
image = Image.open("magnetic_field_cat.png").convert("L")
B_field = np.array(image) / 255.0  # Normalizacja pola do zakresu [0,1]

# Parametry symulacji
q = 1.0  # Ładunek cząstki
m = 1.0  # Masa cząstki
dt = 0.1  # Krok czasowy
steps = 1000  # Ilość kroków symulacji (zmniejszono do testów)

# Początkowe warunki cząstki
x, y = 250, 150  # Startowa pozycja (środek obrazu)
vx, vy = 15.0, 0.0  # Prędkość początkowa

# Listy do przechowywania trajektorii
x_traj, y_traj = [x], [y]

# Tworzenie figury
fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow(B_field, cmap="gray", origin="upper")
line, = ax.plot([], [], color="red", linewidth=1.5, label="Trajektoria cząstki")
dot, = ax.plot([], [], color="green", marker="o", label="Start")
ax.legend()
ax.set_title("Ruch cząstki w polu magnetycznym")

# Funkcja inicjująca animację
def init():
    line.set_data([x_traj[0]], [y_traj[0]])
    dot.set_data([x],[y])  # Hide dot at start
    return line, dot

# Funkcja aktualizująca animację
def update(frame):
    global x, y, vx, vy

    # Pobieramy wartość pola magnetycznego w aktualnym miejscu (zaokrąglamy indeksy)
    B = B_field[int(y) % B_field.shape[0], int(x) % B_field.shape[1]]

    # Siła Lorentza: F = q(v × B), przy czym B działa prostopadle do ruchu
    ax_ = q * vy * B / m
    ay_ = -q * vx * B / m  # Minus, bo działa prostopadle

    # Aktualizacja prędkości i pozycji
    vx += ax_ * dt
    vy += ay_ * dt
    x += vx * dt
    y += vy * dt

    # Debugging: Output current position and force (B)
    print(f"Frame {frame}: Position ({x:.2f}, {y:.2f}), Velocity ({vx:.2f}, {vy:.2f}), B = {B:.2f}")

    # Zapisujemy trajektorię
    x_traj.append(x)
    y_traj.append(y)

    # Ustawienie nowego data dla linii (trajektorii)
    line.set_data(x_traj, y_traj)
    dot.set_data([x], [y])  # Rysowanie cząstki

    return line, dot

# Tworzenie animacji z mniejszą ilością kroków
ani = animation.FuncAnimation(fig, update, frames=1000, init_func=init, blit=True, interval=dt * 1000)

# Wyświetlenie animacji
plt.show()
