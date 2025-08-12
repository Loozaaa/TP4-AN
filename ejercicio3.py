import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy.integrate import simpson as simps
import warnings
from ejercicio1 import procesar_todas_imagenes

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calcular_propiedades_geometricas(df, escala=4.13e-6, densidad=7380):
    """Calcula propiedades geométricas y energía cinética con validación mejorada.

    Args:
        df (pd.DataFrame): DataFrame con datos del ejercicio 1
        escala (float): Factor de conversión de µm a metros (default: 4.13e-6 m/px)
        densidad (float): Densidad del material en kg/m³ (default: 7380)

    Returns:
        pd.DataFrame: DataFrame con las nuevas propiedades calculadas
    """
    # Validación de entrada
    required_cols = ['Contorno_x', 'Contorno_y', 'Tiempo (s)', 'Centroide_y (µm)']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame no contiene columnas requeridas: {required_cols}")

    if escala <= 0:
        raise ValueError("La escala debe ser positiva")

    if densidad <= 0:
        raise ValueError("La densidad debe ser positiva")

    # Convertir escala de µm/pixel a m/pixel
    escala_metros = escala * 1e-6

    # Inicializar listas para resultados
    resultados = {
        'Perimetro_izq (m)': [],
        'Perimetro_der (m)': [],
        'Simetria': [],
        'Factor_esparcimiento': [],
        'Area (m²)': [],
        'Volumen (m³)': [],
        'Energia_cinetica (J)': [],
        'Velocidad (m/s)': []
    }

    # Calcular velocidades (necesario para energía cinética)
    tiempos = df['Tiempo (s)'].values
    posiciones_y = df['Centroide_y (µm)'].values * 1e-6  # Convertir a metros
    velocidades = np.gradient(posiciones_y, tiempos)

    for idx, row in df.iterrows():
        try:
            # Recuperar contorno
            contorno_x = np.array(json.loads(row['Contorno_x']))
            contorno_y = np.array(json.loads(row['Contorno_y']))

            # Validar contorno
            if len(contorno_x) < 10 or len(contorno_y) < 10:
                raise ValueError("Contorno demasiado corto")

            # Combinar en array (y, x)
            contorno = np.column_stack((contorno_y, contorno_x))

            # 1. Cálculo de perímetros y simetría
            # Encontrar puntos de contacto izquierdo y derecho
            idx_izq = np.argmin(contorno[:, 1])  # Punto con x mínimo
            idx_der = np.argmax(contorno[:, 1])  # Punto con x máximo

            # Asegurar orden correcto
            if idx_izq > idx_der:
                idx_izq, idx_der = idx_der, idx_izq

            # Dividir contorno
            contorno_izq = contorno[idx_izq:idx_der + 1]
            contorno_der = np.vstack((contorno[idx_der:], contorno[:idx_izq + 1]))

            # Calcular perímetros (en metros)
            perim_izq = np.sum(
                np.sqrt(np.diff(contorno_izq[:, 1]) ** 2 + np.diff(contorno_izq[:, 0]) ** 2)) * escala_metros
            perim_der = np.sum(
                np.sqrt(np.diff(contorno_der[:, 1]) ** 2 + np.diff(contorno_der[:, 0]) ** 2)) * escala_metros

            # Calcular simetría (0-1, donde 1 es perfectamente simétrico)
            simetria = 1 - abs(perim_izq - perim_der) / max(perim_izq, perim_der)

            # 2. Factor de esparcimiento
            altura_max = np.max(contorno[:, 0]) * escala_metros
            diametro_base = (np.max(contorno[:, 1]) - np.min(contorno[:, 1])) * escala_metros
            factor_esp = diametro_base / (altura_max + 1e-10)

            # 3. Cálculo de área (método del shoelace)
            x = contorno[:, 1] * escala_metros
            y = contorno[:, 0] * escala_metros
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            # 4. Cálculo de volumen (método de los discos)
            # Usamos solo la mitad derecha para evitar duplicar
            mitad_idx = len(contorno) // 2
            mitad_contorno = contorno[mitad_idx:]
            x_vol = mitad_contorno[:, 1] * escala_metros
            y_vol = mitad_contorno[:, 0] * escala_metros

            # Ordenar por y para integración
            sort_idx = np.argsort(y_vol)
            y_vol = y_vol[sort_idx]
            x_vol = x_vol[sort_idx]

            # Calcular volumen (m³)
            volumen = np.pi * simps(x_vol ** 2, y_vol)

            # 5. Energía cinética
            masa = densidad * volumen
            energia = 0.5 * masa * (velocidades[idx] ** 2)

            # Guardar resultados
            resultados['Perimetro_izq (m)'].append(perim_izq)
            resultados['Perimetro_der (m)'].append(perim_der)
            resultados['Simetria'].append(simetria)
            resultados['Factor_esparcimiento'].append(factor_esp)
            resultados['Area (m²)'].append(area)
            resultados['Volumen (m³)'].append(volumen)
            resultados['Energia_cinetica (J)'].append(energia)
            resultados['Velocidad (m/s)'].append(velocidades[idx])

        except Exception as e:
            print(f"Error procesando imagen {row['Imagen']}: {str(e)}")
            for key in resultados:
                resultados[key].append(np.nan)

    # Añadir nuevas columnas al DataFrame
    for key, values in resultados.items():
        df[key] = values

    return df


def graficar_resultados_ej3(df):
    """Genera gráficos mejorados para los resultados del ejercicio 3."""
    try:
        plt.style.use('ggplot')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Análisis de Propiedades Geométricas y Energéticos', fontsize=16, y=1.02)

        # 1. Gráfico de perímetros
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(df['Tiempo (s)'], df['Perimetro_izq (m)'], 'b-', label='Perímetro izquierdo')
        ax1.plot(df['Tiempo (s)'], df['Perimetro_der (m)'], 'r--', label='Perímetro derecho')
        ax1.set_xlabel('Tiempo (s)', fontsize=12)
        ax1.set_ylabel('Perímetro (m)', fontsize=12)
        ax1.set_title('Evolución de los perímetros', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 2. Gráfico de simetría
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(df['Tiempo (s)'], df['Simetria'], 'g-')
        ax2.set_xlabel('Tiempo (s)', fontsize=12)
        ax2.set_ylabel('Coeficiente de simetría', fontsize=12)
        ax2.set_title('Simetría de la gota (1 = perfectamente simétrica)', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 1.1)

        # 3. Gráfico de factor de esparcimiento
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(df['Tiempo (s)'], df['Factor_esparcimiento'], 'm-')
        ax3.set_xlabel('Tiempo (s)', fontsize=12)
        ax3.set_ylabel('Factor de esparcimiento', fontsize=12)
        ax3.set_title('Relación diámetro base / altura máxima', fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.5)

        # 4. Gráfico de energía cinética
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(df['Tiempo (s)'], df['Energia_cinetica (J)'], 'c-', label='Energía cinética')
        ax4.set_xlabel('Tiempo (s)', fontsize=12)
        ax4.set_ylabel('Energía cinética (J)', fontsize=12)
        ax4.set_title('Energía cinética de la gota', fontsize=14)
        ax4.grid(True, linestyle='--', alpha=0.5)

        # Gráfico adicional de velocidad
        ax4b = ax4.twinx()
        ax4b.plot(df['Tiempo (s)'], df['Velocidad (m/s)'], 'k:', alpha=0.7, label='Velocidad')
        ax4b.set_ylabel('Velocidad (m/s)', fontsize=12)

        # Combinar leyendas
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig('resultados_ejercicio3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Gráficos del ejercicio 3 generados en 'resultados_ejercicio3.png'")

    except Exception as e:
        print(f"Error al generar gráficos: {str(e)}")


def generar_informe3(carpeta_imagenes=None, num_imagenes=126):
    """Función principal mejorada para el ejercicio 3."""
    print("\n--- Ejercicio 3: Análisis de propiedades geométricas ---")

    try:
        # Cargar datos
        if carpeta_imagenes:
            print("Procesando imágenes desde la carpeta...")
            if not os.path.exists(carpeta_imagenes):
                raise FileNotFoundError(f"Carpeta no encontrada: {carpeta_imagenes}")

            df = procesar_todas_imagenes(carpeta_imagenes, num_imagenes)
        else:
            print("Cargando datos del ejercicio 1 desde Excel...")
            if not os.path.exists('resultados_completos.xlsx'):
                raise FileNotFoundError("No se encontró 'resultados_completos.xlsx'")

            df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')

        # Validar datos cargados
        if df.empty:
            raise ValueError("No se encontraron datos válidos para procesar")

        # Calcular propiedades
        print("\nCalculando propiedades geométricas...")
        df = calcular_propiedades_geometricas(df)

        # Análisis de resultados
        print("\nResumen estadístico:")
        print(f"- Simetría promedio: {df['Simetria'].mean():.3f} ± {df['Simetria'].std():.3f}")
        print(f"- Factor de esparcimiento promedio: {df['Factor_esparcimiento'].mean():.3f}")
        print(f"- Energía cinética máxima: {df['Energia_cinetica (J)'].max():.2e} J")

        # Conservación de energía (análisis básico)
        energia_inicial = df['Energia_cinetica (J)'].iloc[0]
        energia_final = df['Energia_cinetica (J)'].iloc[-1]
        perdida_energia = 100 * (energia_inicial - energia_final) / energia_inicial
        print(f"\nAnálisis de energía:")
        print(f"- Energía inicial: {energia_inicial:.2e} J")
        print(f"- Energía final: {energia_final:.2e} J")
        print(f"- Pérdida porcentual: {perdida_energia:.1f}%")

        # Exportar resultados
        print("\nExportando resultados...")
        df.to_excel('resultados_completos3.xlsx', index=False)

        # Generar gráficos
        graficar_resultados_ej3(df)

        print("\nEjercicio 3 completado exitosamente!")
        return True

    except Exception as e:
        print(f"\nERROR durante el procesamiento: {str(e)}")
        return False