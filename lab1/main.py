import numpy as np # імпорт бібліотеки для обчислень

import requests # бібліотека для запитів до api

import matplotlib.pyplot as plt # бібліотека для побудови графіків

# запит до open-elevation api
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.154214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url) # виконання get-запиту

data = response.json() # отримання даних у форматі json

results = data['results'] # вибірка списку результатів

#  результати табуляції
n = len(results) # кількість отриманих точок

print(f"кількість вузлів: {n}") # вивід кількості вузлів

print("\nтабуляція вузлів:") # заголовок таблиці

print(f"{'#':<3} | {'latitude':<10} | {'longitude':<10} | {'elevation [m]':<12}") # шапка таблиці

for i, point in enumerate(results): # цикл для виводу кожної точки
    print(f"{i:<3} | {point['latitude']:<10.6f} | {point['longitude']:<10.6f} | {point['elevation']:<12.1f}") # форматований вивід рядка

# обчислення кумулятивної відстані
def haversine(lat1, lon1, lat2, lon2): # функція розрахунку відстані на сфері
    r = 6371000 # середній радіус землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2) # перевід широти в радіани
    dphi = np.radians(lat2 - lat1) # різниця широт у радіанах
    dlambda = np.radians(lon2 - lon1) # різниця довгот у радіанах
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2 # проміжний розрахунок за формулою
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1-a)) # повернення відстані в метрах

coords = [(p['latitude'], p['longitude']) for p in results] # список координат x та у
elevations = [p['elevation'] for p in results] # список висот z

distances = [0] # початкова відстань
for i in range(1, n): # цикл для накопичення відстані
    d = haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]) # крок між сусідніми вузлами
    distances.append(distances[-1] + d) # додавання до загальної суми

#  розв'язок системи лінійних рівнянь методом прогонки
def solve_tridiagonal(a, b, c, d): # функція для тридіагональної матриці
    nf = len(d) # розмірність системи
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # копіювання масивів коефіцієнтів
    for i in range(1, nf): # прямий хід прогонки
        mc = ac[i-1]/bc[i-1] # обчислення множника
        bc[i] = bc[i] - mc*cc[i-1] # перерахунок діагоналі
        dc[i] = dc[i] - mc*dc[i-1] # перерахунок правої частини
    xc = bc # створення масиву для результату
    xc[-1] = dc[-1]/bc[-1] # знаходження останнього невідомого
    for il in range(nf-2, -1, -1): # зворотний хід прогонки
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il] # послідовне знаходження x
    return xc # повернення розв'язку системи

# обчислення коефіцієнтів сплайнів
def build_spline(x, y): # функція для побудови кубічного сплайна
    n = len(x) # кількість вузлів

    h = np.diff(x) # кроки між вузлами по осі x
    
    a_diag = 2 * (h[:-1] + h[1:]) # головна діагональ матриці
    
    a_sub = h[1:-1] # піддіагональ матриці
    
    a_super = h[1:-1] # наддіагональ матриці
    
    rhs = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1]) # права частина рівняння
    
    c_internal = solve_tridiagonal(a_sub, a_diag, a_super, rhs) # пошук коефіцієнтів c
    
    c = np.zeros(n) # масив для всіх c
    
    c[1:-1] = c_internal # запис знайдених значень
    
    a = y[:-1] # коефіцієнт a дорівнює значенню функції
    
    b = (y[1:] - y[:-1]) / h - (h * (c[1:] + 2 * c[:-1])) / 6 # розрахунок коефіцієнта b
    
    d = (c[1:] - c[:-1]) / (6 * h) # розрахунок коефіцієнта d
    
    c_final = c[:-1] / 2 # коригування коефіцієнта c
    
    return a, b, c_final, d # повернення набору коефіцієнтів

# функція для розрахунку наближеного значення
def eval_spline(x_nodes, a, b, c, d, x_target): # пошук y для заданого x
    
    idx = np.searchsorted(x_nodes, x_target) - 1 # пошук відповідного інтервалу
    
    idx = np.clip(idx, 0, len(a) - 1) # обмеження індексу
    
    dx = x_target - x_nodes[idx] # відстань від початку інтервалу
    
    return a[idx] + b[idx]*dx + c[idx]*(dx**2) + d[idx]*(dx**3) # значення полінома

# побудова графіків для різної кількості вузлів
plt.figure(figsize=(10, 6)) # створення вікна графіка

plt.scatter(distances, elevations, color='red', label='вихідні дані') # точки з api


for count in [10, 15, 20]: # цикл для різних наборів вузлів
    
    idx_step = np.linspace(0, n-1, count, dtype=int) # вибір рівномірних індексів
    
    x_sub, y_sub = np.array(distances)[idx_step], np.array(elevations)[idx_step] # вибірка точок
    
    sa, sb, sc, sd = build_spline(x_sub, y_sub) # побудова сплайна
    
    x_plot = np.linspace(x_sub[0], x_sub[-1], 200) # сітка для гладкого графіка
    
    y_plot = [eval_spline(x_sub, sa, sb, sc, sd, v) for v in x_plot] # розрахунок значень
    
    plt.plot(x_plot, y_plot, label=f'сплайн ({count} вузлів)') # малювання ліній

plt.title("профіль висоти маршруту") # назва графіка
plt.xlabel("відстань [м]") # підпис осі x
plt.ylabel("висота [м]") # підпис осі y
plt.legend() # вивід легенди
plt.show() # показ графіка

# характеристики маршруту
total_dist = distances[-1] # повна довжина шляху

total_ascent = sum(max(0, elevations[i]-elevations[i-1]) for i in range(1, n)) # сумарний підйом

print(f"загальна довжина маршруту: {total_dist:.1f} м") # вивід довжини

print(f"сумарний набір висоти: {total_ascent:.1f} м") # вивід набору

# механічна енергія підйому
mass, g = 80, 9.81 # маса тіла та прискорення вільного падіння
energy = mass * g * total_ascent # розрахунок роботи проти сили тяжіння
print(f"механічна енергія підйому: {energy/1000:.2f} кдж") # вивід енергії
