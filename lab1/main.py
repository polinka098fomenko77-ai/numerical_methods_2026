import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=" \
      "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|" \
      "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|" \
      "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|" \
      "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|" \
      "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|" \
      "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|" \
      "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)  # надсилає запит до api
data = response.json()  # перетворює відповідь у формат json

results = data["results"]  # отримує список точок
n = len(results)  # визначає кількість точок

print("Кількість вузлів:", n)  # виводить кількість точок
print("\nТабуляція вузлів:")  # виводить заголовок
print("N° | Latitude | Longitude | Elevation (m)")  # виводить назви колонок

for i, point in enumerate(results):  # перебирає всі точки
    print(f"{i:2d} | {point['latitude']:.6f} | "
          f"{point['longitude']:.6f} | {point['elevation']:.2f}")  # виводить дані точки

def haversine(lat1, lon1, lat2, lon2):  # оголошує функцію відстанi
    R = 6371000  # задає радіус землі
    phi1, phi2 = np.radians(lat1), np.radians(lat2)  # переводить широти у радіани
    dphi = np.radians(lat2 - lat1)  # різниця широт у радіанах
    dlambda = np.radians(lon2 - lon1)  # різниця довгот у радіанах
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2  # формула
    return 2 * R * np.arcsin(np.sqrt(a))  # повертає відстань

coords = [(p["latitude"], p["longitude"]) for p in results]  # створює список координат
elevations = [p["elevation"] for p in results]  # створює список висот

distances = [0]  # створює список відстаней з нуля

for i in range(1, n):  # цикл по точках
    d = haversine(coords[i-1][0], coords[i-1][1],
                  coords[i][0], coords[i][1])  # обчислює відстань між точками
    distances.append(distances[-1] + d)  # додає кумулятивну відстань

print("\nТабуляція (відстань, висота):")  # виводить заголовок
print("N | Distance (m) | Elevation (m)")  # виводить назви колонок

for i in range(n):  # перебирає точки
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")  # виводить відстань і висоту

print("\nХарактеристики маршруту:")  # виводить заголовок

print("Загальна довжина маршруту (м):", distances[-1])  # виводить довжину маршруту

total_ascent = sum(max(elevations[i] - elevations[i-1], 0)
                   for i in range(1, n))  # обчислює сумарний підйом
print("Сумарний набір висоти (м):", total_ascent)  # виводить підйом

total_descent = sum(max(elevations[i-1] - elevations[i], 0)
                    for i in range(1, n))  # обчислює сумарний спуск
print("Сумарний спуск (м):", total_descent)  # виводить спуск

grad = np.gradient(elevations, distances) * 100  # обчислює градієнт

print("\nГрадієнт:")  # виводить заголовок
print("Максимальний підйом (%):", np.max(grad))  # виводить максимальний підйом
print("Максимальний спуск (%):", np.min(grad))  # виводить максимальний спуск
print("Середній градієнт (%):", np.mean(np.abs(grad)))  # виводить середній градієнт

mass = 80  # задає масу
g = 9.81  # задає прискорення вільного падіння

energy = mass * g * total_ascent  # обчислює енергію підйому

print("\nЕнергія підйому:")  # виводить заголовок
print("Механічна робота (Дж):", energy)  # виводить роботу в джоулях
print("Механічна робота (кДж):", energy / 1000)  # виводить роботу в кілоджоулях
print("Енергія (ккал):", energy / 4184)  # виводить енергію в ккал

plt.plot(distances, elevations)  # будує графік
plt.xlabel("Дистанція (m)")  # підпис осі x
plt.ylabel("Elevation (m)")  # підпис осі y
plt.title("Elevation Profile")  # назва графіка
plt.grid()  # додає сітку
plt.show()  # показує графік