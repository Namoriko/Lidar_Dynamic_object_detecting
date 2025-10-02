import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.spatial import cKDTree
from scipy import ndimage
import colorsys
import cv2


def decim_and_average (data, n_decim):
    new_data = []
    time_trend = []
    N = len(data)
    new_data = np.convolve(data, np.ones(n_decim)/n_decim, 'valid')[::n_decim]
    for i in range(len(new_data)):
        time_trend.append(i*n_decim)
    return(new_data, time_trend)


def delet_emission(trend, base_data, delta):
    for i in range(len(trend)):
        if np.min(np.linalg.norm(base_data-trend[i],axis=1))>delta:
            
            trend[i] = -1e8
            print(i)
    trend = trend[trend[:,0]>0]
    return trend

def find_trend_point (indx_before, trend_data, L_need=100):
    indx=0
    L_summ=0
    while (L_summ < L_need) and (indx_before+indx+1 < len(trend_data)):
        indx+=1
        L_summ += np.linalg.norm(trend_data[indx_before+indx]-trend_data[indx_before+indx-1],axis=0)
    start = indx_before
    finish = indx_before + indx
    return (start, finish)




def ordering_trend(data, to_mach_length = 100, index = None):

    for i in range(len(data)-1):
        norm = np.linalg.norm(data[i+1:]-data[i],axis=1)
        index = np.argmin(norm) +i+1
        if index != i+1:
            # print(i, index)
            value = np.copy(data[i+1])
            data[i+1] = data[index]
            data[index] = value
    i = len(data)-1
    norm = np.linalg.norm(data[:]-data[i],axis=1)
    index = np.argmin(norm)
    while np.linalg.norm(data[i-1]-data[i])>to_mach_length:
        data = data[:-1]
        i = len(data)-1
        print(i, index)
        
    return data


# Функция нахождения диагональных углов разбиваемых данных: берём точки тренда, делаем отступ для захвата с запасом и находим
# длиныи и центра прямоугольника.

# point1 - первая точка тренда
# point2 - вторая точка тренда
# reserv - запас для предотвращения потерь данных
def create_rect (point1, point2, reserv=50):
    x_centr = (point1[0] + point2[0])/2
    x_l = abs(point1[0] - point2[0]) + 2*reserv
    y_centr = (point1[1] + point2[1])/2
    y_l = abs(point1[1] - point2[1]) + 2*reserv

    x_start = x_centr - x_l/2
    x_finish = x_centr + x_l/2
    y_start = y_centr - y_l/2
    y_finish = y_centr + y_l/2
    return(x_start, x_finish, y_start, y_finish)


# Функция рассчёта границ массива данных: проводим фильтрацию по границам.

# data - массив данных
# x_start, x_finish, y_start, y_finish - границы
def create_part_data (data, x_start, x_finish, y_start, y_finish):
    # Автоматически определяем минимальное и максимальное значение
    min_x, max_x = sorted([x_start, x_finish])
    min_y, max_y = sorted([y_start, y_finish])
    
    # Фильтруем данные
    filtered_data = data[(data[:, 0] >= min_x) & (data[:, 0] <= max_x) &
                         (data[:, 1] >= min_y) & (data[:, 1] <= max_y)]
    # filtered_indx = np.where(filtered_data)[0]
    return(filtered_data)


# Функция разбиения всего массива данных и сохранения в формате .pcd: пока остаются точкии тренда - продолжаем разбиение.

# TREND - массив линии тренда
# POINTS - массив разбиваемых данных
# POINTS_O3D - массив данных в формате PointCloud
# name - название сохраняемого файл (будет иметь вид "name_№.pcd")
# L_need - длина линиитренда на разбиваемом участке данных
# reserve - запас для предотвращения потерь данных
# save - команда сохранения данных

def partitioning_and_saving(TREND, POINTS, POINTS_O3D, name = "Points", L_need = 150, reserve = 50, save=True, dict_path={}):
    indx_before = 0
    flag = 0
    all_part_data = []
    dict_path['partioned_pcd']='partioned_pcd'
    while indx_before+1 < len(TREND):
        indx1, indx2 = find_trend_point(indx_before, TREND, L_need=L_need)
        x1, x2, y1, y2 = create_rect(TREND[indx1], TREND[indx2], reserv=reserve)
        part_data = create_part_data(POINTS, x1, x2, y1, y2)
        flag+=1
        indx_before = indx2
        all_part_data.append(part_data) # Массив данных numpy
        # Преобразуем numpy в PointCloud для сохранения
        part_points_o3d = o3d.geometry.PointCloud()
        part_points_o3d.points = o3d.utility.Vector3dVector(part_data)
        
        if save:
            if not os.path.isdir('partioned_pcd'):
                os.mkdir('partioned_pcd')
            dict_path['partioned_pcd']='partioned_pcd'
            # Формирование названия файла с номером
            filename = 'partioned_pcd/'+name + f'_part_{flag}.pcd'
            # Сохранение отобранных точек в файл
            o3d.io.write_point_cloud(filename, part_points_o3d)
    #if save:
    #    print(f"Сохранено {flag} массивов в PCD-файлы.")



# x, y, z - координаты массива точек
# degree - степень полиноминального признака
def ML_trendfiltr_ransac_background_removal(x, y, z, degree=2):
    """
    Использует RANSAC для robust оценки фона
    RANSAC игнорирует выбросы при построении модели
    """
    # Создаем полиномиальные признаки
    poly = PolynomialFeatures(degree=degree)
    features = poly.fit_transform(np.column_stack((x, y)))
    
    # Используем RANSAC - он автоматически игнорирует выбросы
    ransac = RANSACRegressor(
        residual_threshold=2,
        max_trials=100,
        stop_probability = 0.95
    )
    
    ransac.fit(features, z)
    
    # Предсказываем фон (только по inliers)
    z_background = ransac.predict(features)
    
    # Inliers маска (точки, которые считаются фоном)
    inlier_mask = ransac.inlier_mask_
    
    corrected = z - z_background
    
    return corrected, z_background, inlier_mask, ransac



# Стоит отметить, что каждое исходное значение может попасть сразу в несколько узлов сетки, тут также рассчитываются векторы нормали,
# используя срежнее значение для величиныи z/

# x, y, z - координаты массива точек
# grid_step - шаг сетки
# radius - радиус, в котором ищуся все точки вблизи узла сетки
# smoothing - величина размыввания для Гауссова фильтра по величине z
def create_grid_fast_binning(x, y, z, grid_step=0.5, radius=1, smoothing=1.0):

    # Создаем регулярную сетку
    x_grid = np.arange(x.min(), x.max() + grid_step, grid_step)
    y_grid = np.arange(y.min(), y.max() + grid_step, grid_step)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Определяем индексы бинов для каждой точки
    x_idx = np.floor((x - x.min()) / grid_step).astype(int)
    y_idx = np.floor((y - y.min()) / grid_step).astype(int)
    
    # Определяем радиус в терминах бинов
    bin_radius = max(1, int(radius / grid_step))
    
    # Инициализируем массивы
    Z_sum = np.zeros_like(X, dtype=float)
    count = np.zeros_like(X, dtype=float)
    
    # Заполняем бины с учетом радиуса
    for i in range(len(x)):
        # Определяем окно вокруг текущей точки
        x_min = max(0, x_idx[i] - bin_radius)
        x_max = min(X.shape[1], x_idx[i] + bin_radius + 1)
        y_min = max(0, y_idx[i] - bin_radius)
        y_max = min(X.shape[0], y_idx[i] + bin_radius + 1)
        
        # Вычисляем расстояния до всех ячеек в окне
        for xi in range(x_min, x_max):
            for yi in range(y_min, y_max):
                dist = np.sqrt((x[i] - X[yi, xi])**2 + (y[i] - Y[yi, xi])**2)
                if dist <= radius:
                    Z_sum[yi, xi] += z[i]
                    count[yi, xi] += 1
    
    # Вычисляем среднее
    Z_avg = np.zeros_like(Z_sum)
    mask = count > 0
    Z_avg[mask] = Z_sum[mask] / count[mask]
    
    # Остальная часть кода
    if smoothing > 0:
        Z_avg_smooth = ndimage.gaussian_filter(Z_avg, sigma=smoothing)
    else:
        Z_avg_smooth = Z_avg
    
    grad_x, grad_y = np.gradient(Z_avg_smooth, grid_step)
    nx = -grad_x
    ny = -grad_y
    nz = np.ones_like(grad_x)
    
    norm_length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / norm_length
    ny = ny / norm_length
    nz = nz / norm_length
    
    return X, Y, Z_sum, Z_avg, nx, ny, nz


# Создаём цветовую модель HSV

# nx, ny, nz - величины нормалей для точки
# intensity - значение z, отвечающая за яркость точки
# gamma - степень в гамма-корреляции
def normals_to_hsv(nx, ny, nz, intensity, gamma=0.15):
    """
    Преобразует нормали в HSV цветовое пространство:
    - H (оттенок) определяется азимутальным углом нормали
    - S (насыщенность) определяется углом наклона нормали
    - V (яркость) определяется интенсивностью (суммой Z)
    """
    # Вычисляем азимутальный угол (от 0 до 2π)
    azimuth = np.arctan2(ny, nx)  # от -π до π
    azimuth = (azimuth + 2 * np.pi) % (2 * np.pi)  # от 0 до 2π
    
    # Вычисляем угол наклона (от 0 до π/2)
    # nz = cos(θ), где θ - угол между нормалью и вертикалью
    inclination = np.arccos(np.clip(nz, -1, 1))  # от 0 до π
    
    # Нормализуем углы для HSV
    H = azimuth / (2 * np.pi)  # от 0 до 1
    S = inclination / (np.pi / 2)  # от 0 до 1 (ограничиваем максимум π/2)
    S = np.clip(S, 0, 1)  # на всякий случай ограничиваем
    
    # Нормализуем интенсивность для яркости
    # Используем логарифмическое масштабирование для лучшего восприятия
    V = intensity - intensity.min()
    if V.max() > 0:
        V = V / V.max()
    
    # Применяем гамма-коррекцию для лучшего визуального восприятия
    V = V ** gamma
    
    return H, S, V



# Переводим в цветовую модель RGB
def hsv_to_rgb(H, S, V):
    """
    Преобразует массивы HSV в RGB
    """
    # Создаем пустой RGB массив
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    
    # Векторизованное преобразование HSV to RGB
    # Адаптация алгоритма из colorsys для массивов
    h_i = (H * 6) % 6
    c = V * S
    x = c * (1 - np.abs(h_i % 2 - 1))
    m = V - c
    
    conditions = [
        h_i < 1,
        (h_i >= 1) & (h_i < 2),
        (h_i >= 2) & (h_i < 3),
        (h_i >= 3) & (h_i < 4),
        (h_i >= 4) & (h_i < 5),
        h_i >= 5
    ]
    
    choices_R = [c, x, 0, 0, x, c]
    choices_G = [x, c, c, x, 0, 0]
    choices_B = [0, 0, x, c, c, x]
    
    R = np.select(conditions, choices_R) + m
    G = np.select(conditions, choices_G) + m
    B = np.select(conditions, choices_B) + m
    
    return np.dstack((R, G, B))


# path - маршрут
# main_file - название основного файла
# input_degree - степень полиноминального признака при фильтрации дороги
# l_tree, l_road - уровни для удаления точек полотна дороги и крон деревьев
# input_grid_step - шаг сетки
# input_radius - радиус, в котором ищуся все точки вблизи узла сетки
# input_smoothing - величина размыввания для Гауссова фильтра по величине z
# input_gamma_cor - степень гамма-корреляции для корректирвки яркости цветов
# input_dpi - задаваемое разрешение
def make_pictures(main_file, input_degree =3, l_tree = 3.5, l_road = 0.5, input_grid_step=0.5, input_radius=1, input_smoothing=1.0, input_gamma_cor=0.15, input_dpi = 480, dict_path={}):
    name= main_file.split('.pcd')[0]
    dict_path['partioned_jpg'] = 'partioned_jpg'
    print('Processing')
    print(os.listdir(dict_path['partioned_pcd']))
    for file in os.listdir(dict_path['partioned_pcd']):
        if f'{name}_part' not in file or '.pcd' not in file: continue
        pcd_data = o3d.io.read_point_cloud(dict_path['partioned_pcd']+'/'+file)
        x,y,z = np.asarray(pcd_data.points).T
        #  Корректируем координату  Z
        z, z_background_ransac, inlier_mask, ransac_model  = ML_trendfiltr_ransac_background_removal(
        x, y, z, 
        degree=input_degree)
        # delete road and trees
        wtrees_x = []
        wtrees_y = []
        wtrees_z = []
        for i in range(len(z)):
            if (z[i] < l_tree) and (z[i] > l_road):
                wtrees_x.append(x[i])
                wtrees_y.append(y[i])
                wtrees_z.append(z[i])
        
        x=np.array(wtrees_x)
        y=np.array(wtrees_y)
        z=np.array(wtrees_z)
        X, Y, Z_sum, Z_avg, nx, ny, nz = create_grid_fast_binning(
        x, y, z, 
        grid_step=input_grid_step,
        radius=input_radius,
        smoothing=input_smoothing)
        # Преобразуем в RGB
        H, S, V = normals_to_hsv(nx, ny, nz, Z_sum, input_gamma_cor)
        RGB = hsv_to_rgb(H, S, V)
        file_name = file.split('.pcd')[0]
        plt.figure(figsize=(10, 8))
        plt.imshow(RGB, 
           extent=[x.min(), x.max(),  # xmin, xmax
                   y.min(), y.max()], # ymin, ymax
           origin='lower')
        plt.axis('off')
        if not os.path.isdir('partioned_jpg'):
            os.mkdir('partioned_jpg')
        plt.savefig(f'partioned_jpg/{file_name}.jpg', dpi=input_dpi , bbox_inches='tight', pad_inches=0)
        #plt.show()
        print(f'{file_name}.jpg saved')
        


################################
################################
# часть главного файла
################################
################################
def main(file_path, file_name):
    dict_path = {'full_pcd':file_path,'name': file_name.split('.pcd')[0]}
    pcd_data = o3d.io.read_point_cloud(file_path).uniform_down_sample(100)
    print('read')
    points_mat = np.asarray(pcd_data.points)  # Преобразование в массив NumPy
    print('read')
    Nomber_trends_point = 300 # Количество точек в линии тренда
    trend_x, N_trend = decim_and_average(points_mat[:,0], int(points_mat[:,0].shape[0]/Nomber_trends_point))
    trend_y, N_trend = decim_and_average(points_mat[:,1], int(points_mat[:,0].shape[0]/Nomber_trends_point))
    trend = np.array([trend_x,trend_y]).T
    print('trend')
    trend=delet_emission(trend, points_mat[:, 0:2], delta=5) 
    print('emission')
    pcd_data = o3d.io.read_point_cloud(file_path)
    print('read')
    points_mat = np.asarray(pcd_data.points)  # Преобразование в массив NumPy
    partitioning_and_saving(TREND = trend, POINTS = points_mat, POINTS_O3D=pcd_data, name = dict_path['name'], L_need = 150, reserve = 50, save=True, dict_path = dict_path)
    #print('saved')
    #print(dict_path)
    
    make_pictures(dict_path['name'], 
                  input_degree =4, l_tree = 3.125, l_road = 0.5, 
                  input_grid_step=0.07, input_radius=0.14, input_smoothing=0.2, 
                  input_gamma_cor=0.15, 
                  input_dpi = 720,
                  dict_path = dict_path)
    print('pictures')
    
