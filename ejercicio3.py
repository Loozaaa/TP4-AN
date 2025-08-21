import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy.interpolate import CubicSpline
from ejercicio1 import procesar_todas_imagenes
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def encontrar_puntos_contacto(contorno, tol_altura=5):
    """Encuentra puntos de contacto basado en proximidad al mínimo de Y."""
    y_min = np.min(contorno[:, 0])
    puntos_base_idx = np.where(np.abs(contorno[:, 0] - y_min) < tol_altura)[0]

    if len(puntos_base_idx) < 2:
        return np.argmin(contorno[:, 1]), np.argmax(contorno[:, 1])

    puntos_base = contorno[puntos_base_idx]
    idx_izq = puntos_base_idx[np.argmin(puntos_base[:, 1])]
    idx_der = puntos_base_idx[np.argmax(puntos_base[:, 1])]

    return idx_izq, idx_der

def calcular_perimetro(contorno, escala):
    """Calcula perímetro usando interpolación cúbica para contornos curvos."""
    try:
        t = np.linspace(0, 1, len(contorno))
        cs_x = CubicSpline(t, contorno[:, 1])
        cs_y = CubicSpline(t, contorno[:, 0])

        t_fine = np.linspace(0, 1, 5 * len(contorno))
        dx_dt = cs_x(t_fine, 1)
        dy_dt = cs_y(t_fine, 1)

        return np.trapz(np.sqrt(dx_dt ** 2 + dy_dt ** 2), t_fine) * escala
    except:
        diffs = np.sqrt(np.diff(contorno[:, 1]) ** 2 + np.diff(contorno[:, 0]) ** 2)
        return np.sum(diffs) * escala

def validar_energia(factor_esparcimiento, velocidad, densidad, altura_max, diametro_base):
    """Valida valores y calcula energía cinética aproximada usando factor de esparcimiento."""
    if np.isnan(factor_esparcimiento) or abs(velocidad) > 100:
        return np.nan
    # Aproximamos el volumen en base a geometría cilíndrica simplificada
    volumen_aprox = np.pi * (diametro_base / 2) ** 2 * altura_max
    return 0.5 * densidad * volumen_aprox * velocidad ** 2

def calcular_propiedades_geometricas(df, escala=4.13e-6, densidad=7380):
    """Calcula propiedades geométricas con las mejoras críticas."""
    required_cols = ['Contorno_x', 'Contorno_y', 'Tiempo (s)', 'Centroide_y (µm)', 'Centroide_x (µm)']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame no contiene columnas requeridas: {required_cols}")

    escala_metros = escala

    resultados = {
        'Perimetro_izq (m)': [], 'Perimetro_der (m)': [], 'Simetria': [],
        'Energia_cinetica (J)': [], 'Velocidad (m/s)': [], 'Altura_maxima (m)': [],
        'Diametro_base (m)': []
    }

    tiempos = df['Tiempo (s)'].values
    posiciones_y = df['Centroide_y (µm)'].values * 1e-6
    velocidades = np.gradient(posiciones_y, tiempos)

    for idx, row in df.iterrows():
        try:
            contorno_x = np.array(json.loads(row['Contorno_x']))
            contorno_y = np.array(json.loads(row['Contorno_y']))
            centro_x = row['Centroide_x (µm)']

            contorno = np.column_stack((contorno_y, contorno_x))

            # Puntos de contacto
            idx_izq, idx_der = encontrar_puntos_contacto(contorno)
            if idx_izq > idx_der:
                idx_izq, idx_der = idx_der, idx_izq

            contorno_izq = contorno[idx_izq:idx_der + 1]
            contorno_der = np.vstack((contorno[idx_der:], contorno[:idx_izq + 1]))

            # Perímetros
            perim_izq = calcular_perimetro(contorno_izq, escala_metros)
            perim_der = calcular_perimetro(contorno_der, escala_metros)
            simetria = 1 - abs(perim_izq - perim_der) / max(perim_izq, perim_der)

            # Geometría
            altura_max = np.max(contorno[:, 0]) * escala_metros
            diametro_base = (np.max(contorno[:, 1]) - np.min(contorno[:, 1])) * escala_metros

            # Factor de Esparcimiento (Excel del Ejercicio2)
            factor_esp = row['Factor_esparcimiento']

            # Energía cinética
            energia = validar_energia(factor_esp, velocidades[idx], densidad, altura_max, diametro_base)

            # Guardar
            resultados['Perimetro_izq (m)'].append(perim_izq)
            resultados['Perimetro_der (m)'].append(perim_der)
            resultados['Simetria'].append(simetria)
            resultados['Energia_cinetica (J)'].append(energia)
            resultados['Velocidad (m/s)'].append(velocidades[idx])
            resultados['Altura_maxima (m)'].append(altura_max)
            resultados['Diametro_base (m)'].append(diametro_base)

        except Exception as e:
            print(f"Error procesando imagen {row.get('Imagen', idx)}: {str(e)}")
            for key in resultados:
                resultados[key].append(np.nan)

    for key, values in resultados.items():
        df[key] = values

    return df

def graficar_resultados_ej3(df):
    try:
        plt.style.use('default')
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Propiedades Geométricas - Ejercicio 3', fontsize=16)

        # 1. Perímetros y simetría
        axs[0, 0].plot(df['Tiempo (s)'], df['Perimetro_izq (m)'], 'b-', label='Perímetro izquierdo', alpha=0.8)
        axs[0, 0].plot(df['Tiempo (s)'], df['Perimetro_der (m)'], 'r--', label='Perímetro derecho', alpha=0.8)
        axs[0, 0].set_xlabel('Tiempo (s)')
        axs[0, 0].set_ylabel('Perímetro (m)')
        axs[0, 0].grid(True, alpha=0.3)

        ax02 = axs[0, 0].twinx()
        ax02.plot(df['Tiempo (s)'], df['Simetria'], 'g:', label='Simetría', alpha=0.6)
        ax02.set_ylabel('Coeficiente de simetría')
        ax02.set_ylim(0, 1.1)

        lines, labels = axs[0, 0].get_legend_handles_labels()
        lines2, labels2 = ax02.get_legend_handles_labels()
        axs[0, 0].legend(lines + lines2, labels + labels2, loc='upper left')

        # 2. Factor de esparcimiento (desde Excel)
        axs[0, 1].plot(df['Tiempo (s)'], df['Factor_esparcimiento'], 'm-', alpha=0.8)
        axs[0, 1].set_xlabel('Tiempo (s)')
        axs[0, 1].set_ylabel('Factor de esparcimiento')
        axs[0, 1].set_title('Relación diámetro base / altura máxima')
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Energía cinética y velocidad
        axs[1, 0].plot(df['Tiempo (s)'], df['Energia_cinetica (J)'], 'c-', label='Energía cinética', alpha=0.8)
        axs[1, 0].set_xlabel('Tiempo (s)')
        axs[1, 0].set_ylabel('Energía cinética (J)')
        axs[1, 0].grid(True, alpha=0.3)

        ax10 = axs[1, 0].twinx()
        ax10.plot(df['Tiempo (s)'], df['Velocidad (m/s)'], 'k:', label='Velocidad', alpha=0.6)
        ax10.set_ylabel('Velocidad (m/s)')

        lines, labels = axs[1, 0].get_legend_handles_labels()
        lines2, labels2 = ax10.get_legend_handles_labels()
        axs[1, 0].legend(lines + lines2, labels + labels2, loc='upper left')

        # 4. Altura máxima y diámetro base
        axs[1, 1].plot(df['Tiempo (s)'], df['Altura_maxima (m)'], 'orange', label='Altura máxima', alpha=0.8)
        axs[1, 1].set_xlabel('Tiempo (s)')
        axs[1, 1].set_ylabel('Altura máxima (m)')
        axs[1, 1].grid(True, alpha=0.3)

        ax11 = axs[1, 1].twinx()
        ax11.plot(df['Tiempo (s)'], df['Diametro_base (m)'], 'purple', label='Diámetro base', alpha=0.6)
        ax11.set_ylabel('Diámetro base (m)')

        lines, labels = axs[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax11.get_legend_handles_labels()
        axs[1, 1].legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig('resultados_ejercicio3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Gráficos del ejercicio 3 generados en 'resultados_ejercicio3.png'")

    except Exception as e:
        print(f"Error al generar gráficos: {str(e)}")

def generar_informe3(carpeta_imagenes=None, num_imagenes=126):
    """Función principal optimizada para el ejercicio 3."""
    print("\n=== EJERCICIO 3: Análisis de propiedades geométricas ===\n")
    try:
        if carpeta_imagenes and os.path.exists(carpeta_imagenes):
            print("Procesando imágenes desde la carpeta...")
            df1 = procesar_todas_imagenes(carpeta_imagenes, num_imagenes)
        elif os.path.exists('resultados_completos.xlsx') and os.path.exists('resultados_completos2.xlsx'):
            print("Cargando datos desde Excel (ejercicio 1 y 2)...")
            df1 = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
            df2 = pd.read_excel('resultados_completos2.xlsx')
            df1 = pd.merge(df1, df2[['Imagen', 'Tiempo (s)', 'Factor_esparcimiento']], on=['Imagen', 'Tiempo (s)'], how='inner')
        else:
            raise FileNotFoundError("No se encontraron datos para procesar")

        if df1.empty:
            raise ValueError("DataFrame vacío - no hay datos válidos")

        print("Calculando propiedades geométricas mejoradas...")
        df = calcular_propiedades_geometricas(df1)

        print("\n--- RESULTADOS PRINCIPALES ---")
        print(f"Simetría promedio: {df['Simetria'].mean():.3f} ± {df['Simetria'].std():.3f}")
        print(f"Factor de esparcimiento promedio: {df['Factor_esparcimiento'].mean():.3f}")

        # Análisis de conservación de energía
        energia_valida = df['Energia_cinetica (J)'].dropna()
        if len(energia_valida) > 1:
            energia_inicial = energia_valida.iloc[0]
            energia_final = energia_valida.iloc[-1]
            perdida_porcentual = 100 * (energia_inicial - energia_final) / energia_inicial

            print(f"\n--- ANÁLISIS ENERGÉTICO ---")
            print(f"Energía cinética inicial: {energia_inicial:.2e} J")
            print(f"Energía cinética final: {energia_final:.2e} J")
            print(f"Pérdida porcentual: {perdida_porcentual:.1f}%")

        print("\nExportando resultados...")
        df.to_excel('resultados_completos3.xlsx', index=False)

        graficar_resultados_ej3(df)

        print("\nEJERCICIO 3 COMPLETADO EXITOSAMENTE")
        print("Resultados guardados en 'resultados_completos3.xlsx'")
        print("Gráficos guardados en 'resultados_ejercicio3.png'")

        return True

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False