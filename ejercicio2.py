import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd  # ¡Esta línea es crítica!
import json
import warnings
from scipy.integrate import simpson as simps
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

def ajustar_contornos(df, grado_polinomio=3, suavizado_spline=0.5, bins_y=None):
    """Ajusta contornos izquierdo y derecho construyendo perfiles x(y) por niveles de y."""
    resultados = []

    # Verificar columnas requeridas
    required_columns = ['Contorno_x', 'Contorno_y', 'Imagen', 'Tiempo (s)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("El DataFrame no contiene las columnas necesarias.")

    for _, row in df.iterrows():
        try:
            # Leer contornos robustamente
            x = np.array(json.loads(row['Contorno_x'])) if isinstance(row['Contorno_x'], str) else np.array(row['Contorno_x'])
            y = np.array(json.loads(row['Contorno_y'])) if isinstance(row['Contorno_y'], str) else np.array(row['Contorno_y'])

            # Filtrar inválidos
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            if len(x) < 10:
                # muy pocos puntos, lo reportamos y saltamos
                # print(f"[WARN] {row['Imagen']}: contorno con pocos puntos ({len(x)})")
                continue

            # Opcional: agrupar y en bins si el contorno es muy ruidoso
            if bins_y is None:
                # usar valores de y tal cual (float) pero redondear para agrupar puntos muy cercanos
                y_round = np.round(y, decimals=2)
            else:
                # crear bins uniformes
                y_round = np.digitize(y, bins_y)

            # Construir perfiles: para cada nivel de y, tomar min(x) -> izq, max(x) -> der
            unique_y = np.unique(y_round)
            x_left = []
            y_left = []
            x_right = []
            y_right = []

            for uy in unique_y:
                mask_level = (y_round == uy)
                xs = x[mask_level]
                ys = y[mask_level]
                if len(xs) == 0:
                    continue
                # usar la media de y del nivel para mantenerlo ordenado
                y_mean = float(np.mean(ys))
                x_left.append(np.min(xs))
                y_left.append(y_mean)
                x_right.append(np.max(xs))
                y_right.append(y_mean)

            x_left = np.array(x_left); y_left = np.array(y_left)
            x_right = np.array(x_right); y_right = np.array(y_right)

            # Ordenar por y ascendente
            if len(y_left) > 0:
                idx_l = np.argsort(y_left)
                x_left, y_left = x_left[idx_l], y_left[idx_l]
            if len(y_right) > 0:
                idx_r = np.argsort(y_right)
                x_right, y_right = x_right[idx_r], y_right[idx_r]

            # Validar conteo de puntos para spline
            spline_izq = None
            spline_der = None
            if len(y_left) > 3:
                # Ajuste adaptativo del suavizado basado en la cantidad de puntos y su distribución
                s_val = suavizado_spline
                if len(y_left) > 50:
                    s_val *= np.sqrt(len(y_left) / 50)  # Incremento más gradual
                # Ajustar según la variabilidad de los datos
                variability = np.std(np.diff(x_left)) / np.mean(np.diff(y_left))
                s_val *= max(1, variability)
                spline_izq = UnivariateSpline(y_left, x_left, s=s_val, k=3)  # k=3 para suavidad

            if len(y_right) > 3:
                s_val = suavizado_spline
                if len(y_right) > 50:
                    s_val *= np.sqrt(len(y_right) / 50)
                variability = np.std(np.diff(x_right)) / np.mean(np.diff(y_right))
                s_val *= max(1, variability)
                spline_der = UnivariateSpline(y_right, x_right, s=s_val, k=3)

            # Coeficientes polinomiales (opcional)
            coef_poly_izq = np.polyfit(y_left, x_left, grado_polinomio) if len(y_left) > grado_polinomio else None
            coef_poly_der = np.polyfit(y_right, x_right, grado_polinomio) if len(y_right) > grado_polinomio else None

            # Perímetros (usando los perfiles reconstruidos)
            perimetro_izq = np.sum(np.sqrt(np.diff(x_left) ** 2 + np.diff(y_left) ** 2)) if len(x_left) > 1 else np.nan
            perimetro_der = np.sum(np.sqrt(np.diff(x_right) ** 2 + np.diff(y_right) ** 2)) if len(x_right) > 1 else np.nan

            diametro_base = np.max(x) - np.min(x) if len(x) > 0 else np.nan
            altura_max = np.max(y) if len(y) > 0 else np.nan
            centro_x = np.mean(x)

            resultados.append({
                'Imagen': row['Imagen'],
                'Tiempo (s)': row['Tiempo (s)'],
                'spline_izq': spline_izq,
                'spline_der': spline_der,
                'coef_poly_izq': coef_poly_izq.tolist() if coef_poly_izq is not None else None,
                'coef_poly_der': coef_poly_der.tolist() if coef_poly_der is not None else None,
                'Perimetro_izq': perimetro_izq,
                'Perimetro_der': perimetro_der,
                'Asimetria_perimetro': abs(perimetro_izq - perimetro_der) / (perimetro_izq + perimetro_der)
                    if (perimetro_izq + perimetro_der) > 0 else np.nan,
                'Diametro_base': diametro_base,
                'Altura_max': altura_max,
                'Factor_esparcimiento': diametro_base / altura_max if altura_max > 0 else np.nan,
                'Area': simps(np.abs(x - centro_x), y) if len(x) > 1 else np.nan
            })

        except Exception as e:
            print(f"Error procesando {row.get('Imagen', 'desconocida')}: {str(e)}")
            continue

    return pd.DataFrame(resultados)

def calcular_angulo_contacto(df_ajustes, altura_contacto=50):
    """Calcula ángulos de contacto evaluando la derivada de los splines
       pero sólo dentro del dominio válido y comprobando finitud."""
    angulos = []
    densidad = 7380  # kg/m³

    if df_ajustes.empty:
        raise ValueError("DataFrame de ajustes está vacío.")

    for _, row in df_ajustes.iterrows():
        try:
            angulo_izq = np.nan
            angulo_der = np.nan

            # Para elegir puntos de evaluación usar el dominio donde el spline está definido
            if row['spline_izq'] is not None:
                try:
                    y_min, y_max = row['spline_izq'].get_knots()[0], row['spline_izq'].get_knots()[-1]
                    # Usar más puntos de evaluación y concentrarlos cerca de la base
                    y_eval = np.linspace(max(0, y_min), min(altura_contacto, y_max), 20)
                    # Concentrar más puntos cerca de la base usando una distribución exponencial
                    y_eval = y_min + (y_eval - y_min) ** 2 / (y_eval[-1] - y_min)
                    
                    # Calcular derivadas y filtrar valores atípicos
                    dxdy_izq = row['spline_izq'].derivative()(y_eval)
                    dxdy_izq = np.asarray(dxdy_izq, dtype=float)
                    
                    if np.any(np.isfinite(dxdy_izq)):
                        # Filtrar valores atípicos usando la mediana y la desviación absoluta mediana
                        valid_vals = dxdy_izq[np.isfinite(dxdy_izq)]
                        median = np.median(valid_vals)
                        mad = np.median(np.abs(valid_vals - median))
                        mask = np.abs(valid_vals - median) < 2.0 * mad
                        
                        if np.any(mask):
                            # Usar media ponderada dando más peso a los puntos cercanos a la base
                            weights = np.exp(-y_eval[np.isfinite(dxdy_izq)][mask] / altura_contacto)
                            angulo_izq = float(np.average(
                                np.degrees(np.arctan(valid_vals[mask])),
                                weights=weights
                            ))
                except Exception as e:
                    # print(f"[WARN] derivada izq fallo en {row['Imagen']}: {e}")
                    angulo_izq = np.nan

            if row['spline_der'] is not None:
                try:
                    y_min, y_max = row['spline_der'].get_knots()[0], row['spline_der'].get_knots()[-1]
                    # Usar más puntos de evaluación y concentrarlos cerca de la base
                    y_eval = np.linspace(max(0, y_min), min(altura_contacto, y_max), 20)
                    # Concentrar más puntos cerca de la base usando una distribución exponencial
                    y_eval = y_min + (y_eval - y_min) ** 2 / (y_eval[-1] - y_min)
                    
                    # Calcular derivadas y filtrar valores atípicos
                    dxdy_der = row['spline_der'].derivative()(y_eval)
                    dxdy_der = np.asarray(dxdy_der, dtype=float)
                    
                    if np.any(np.isfinite(dxdy_der)):
                        # Filtrar valores atípicos usando la mediana y la desviación absoluta mediana
                        valid_vals = dxdy_der[np.isfinite(dxdy_der)]
                        median = np.median(valid_vals)
                        mad = np.median(np.abs(valid_vals - median))
                        mask = np.abs(valid_vals - median) < 2.0 * mad
                        
                        if np.any(mask):
                            # Usar media ponderada dando más peso a los puntos cercanos a la base
                            weights = np.exp(-y_eval[np.isfinite(dxdy_der)][mask] / altura_contacto)
                            angulo_der = float(np.average(
                                np.degrees(np.arctan(valid_vals[mask])),
                                weights=weights
                            ))
                except Exception as e:
                    # print(f"[WARN] derivada der fallo en {row['Imagen']}: {e}")
                    angulo_der = np.nan

            # Calculamos el cambio en el factor de esparcimiento respecto al tiempo anterior
            factor_actual = row.get('Factor_esparcimiento', np.nan)
            tiempo_actual = row['Tiempo (s)']
            
            # Si es el primer punto o hay un cambio significativo en el factor de esparcimiento, es dinámico
            umbral_cambio = 0.05  # 5% de cambio como umbral
            es_dinamico = True
            
            if len(angulos) > 0:
                factor_anterior = angulos[-1].get('Factor_esparcimiento', np.nan)
                tiempo_anterior = angulos[-1].get('Tiempo (s)', 0)
                
                if not np.isnan(factor_actual) and not np.isnan(factor_anterior) and tiempo_actual > tiempo_anterior:
                    tasa_cambio = abs((factor_actual - factor_anterior) / (factor_anterior * (tiempo_actual - tiempo_anterior)))
                    es_dinamico = tasa_cambio > umbral_cambio

            angulos.append({
                'Imagen': row['Imagen'],
                'Tiempo (s)': row['Tiempo (s)'],
                'Angulo_izq': angulo_izq,
                'Angulo_der': angulo_der,
                'Diferencia_angular': abs(angulo_izq - angulo_der)
                    if not np.isnan(angulo_izq) and not np.isnan(angulo_der) else np.nan,
                'Perimetro_izq': row.get('Perimetro_izq', np.nan),
                'Perimetro_der': row.get('Perimetro_der', np.nan),
                'Asimetria_perimetro': row.get('Asimetria_perimetro', np.nan),
                'Factor_esparcimiento': factor_actual,
                'Energia_cinetica': np.nan,  # si quieres mantener cálculo, ajustarlo aquí
                'Tipo_angulo': 'Dinámico' if es_dinamico else 'Estático'
            })

        except Exception as e:
            print(f"Error calculando ángulos para {row.get('Imagen', 'desconocida')}: {str(e)}")
            angulos.append({
                'Imagen': row.get('Imagen', 'desconocida'),
                'Tiempo (s)': row.get('Tiempo (s)', np.nan),
                'Angulo_izq': np.nan,
                'Angulo_der': np.nan,
                'Diferencia_angular': np.nan,
                'Perimetro_izq': np.nan,
                'Perimetro_der': np.nan,
                'Asimetria_perimetro': np.nan,
                'Factor_esparcimiento': np.nan,
                'Energia_cinetica': np.nan,
                'Tipo_angulo': 'Desconocido'
            })
            continue

    return pd.DataFrame(angulos)

def graficar_resultados(df_angulos):
    """Grafica resultados con manejo de datos faltantes y mejora visual."""
    if df_angulos.empty:
        raise ValueError("No hay datos para graficar.")

    plt.style.use('default')  # Resetear al estilo default
    plt.figure(figsize=(15, 10))
    # Configurar un estilo más moderno manualmente
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'axes.labelsize': 10,
        'figure.figsize': (15, 10),
        'lines.linewidth': 2
    })

    # Gráfico 1: Ángulos
    ax1 = plt.subplot(2, 2, 1)
    if 'Angulo_izq' in df_angulos:
        plt.plot(df_angulos['Tiempo (s)'], df_angulos['Angulo_izq'], 'b-', label='Izquierdo', linewidth=2)
    if 'Angulo_der' in df_angulos:
        plt.plot(df_angulos['Tiempo (s)'], df_angulos['Angulo_der'], 'r-', label='Derecho', linewidth=2)
    plt.title('Evolución de Ángulos de Contacto', fontsize=12, pad=10)
    plt.xlabel('Tiempo (s)', fontsize=10)
    plt.ylabel('Ángulo (grados)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Agregar valor promedio como línea horizontal
    if 'Angulo_izq' in df_angulos and 'Angulo_der' in df_angulos:
        avg_izq = df_angulos['Angulo_izq'].mean()
        avg_der = df_angulos['Angulo_der'].mean()
        plt.axhline(y=avg_izq, color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=avg_der, color='r', linestyle='--', alpha=0.5)
        # Agregar anotaciones con los promedios
        plt.text(0.02, 0.98, f'Prom. Izq: {avg_izq:.1f}°', 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top', color='blue')
        plt.text(0.02, 0.93, f'Prom. Der: {avg_der:.1f}°', 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top', color='red')

    # Gráfico 2: Asimetría
    ax2 = plt.subplot(2, 2, 2)
    if 'Asimetria_perimetro' in df_angulos:
        plt.plot(df_angulos['Tiempo (s)'], df_angulos['Asimetria_perimetro'], 'g-', linewidth=2)
    plt.title('Asimetría del Perímetro', fontsize=12, pad=10)
    plt.xlabel('Tiempo (s)', fontsize=10)
    plt.ylabel('Asimetría', fontsize=10)
    plt.grid(True, alpha=0.3)
    if 'Asimetria_perimetro' in df_angulos:
        avg_asim = df_angulos['Asimetria_perimetro'].mean()
        plt.axhline(y=avg_asim, color='g', linestyle='--', alpha=0.5)
        plt.text(0.02, 0.98, f'Prom. Asimetría: {avg_asim:.3f}', 
                transform=ax2.transAxes, fontsize=9, verticalalignment='top', color='green')

    # Gráfico 3: Factor de esparcimiento
    ax3 = plt.subplot(2, 2, 3)
    if 'Factor_esparcimiento' in df_angulos:
        plt.plot(df_angulos['Tiempo (s)'], df_angulos['Factor_esparcimiento'], 'm-', linewidth=2)
    plt.title('Factor de Esparcimiento', fontsize=12, pad=10)
    plt.xlabel('Tiempo (s)', fontsize=10)
    plt.ylabel('Factor', fontsize=10)
    plt.grid(True, alpha=0.3)
    if 'Factor_esparcimiento' in df_angulos:
        avg_factor = df_angulos['Factor_esparcimiento'].mean()
        plt.axhline(y=avg_factor, color='m', linestyle='--', alpha=0.5)
        plt.text(0.02, 0.98, f'Prom. Factor: {avg_factor:.3f}', 
                transform=ax3.transAxes, fontsize=9, verticalalignment='top', color='magenta')

    # Gráfico 4: Histograma de ángulos
    ax4 = plt.subplot(2, 2, 4)
    if 'Angulo_izq' in df_angulos and 'Angulo_der' in df_angulos:
        plt.hist(df_angulos['Angulo_izq'].dropna(), bins=20, alpha=0.5, color='blue', label='Izquierdo')
        plt.hist(df_angulos['Angulo_der'].dropna(), bins=20, alpha=0.5, color='red', label='Derecho')
    plt.title('Distribución de Ángulos', fontsize=12, pad=10)
    plt.xlabel('Ángulo (grados)', fontsize=10)
    plt.ylabel('Frecuencia', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Agregar un título general
    plt.suptitle('Análisis de Ángulos de Contacto y Parámetros Relacionados', 
                fontsize=14, y=1.02)
    
    # Guardar con alta resolución
    plt.savefig('resultados_angulos.png', dpi=300, bbox_inches='tight')
    plt.savefig('resultados_ejercicio3.png', dpi=300, bbox_inches='tight')  # Guardar con ambos nombres
    plt.close()

def generar_informe2():
    print("\n--- Ejercicio 2: Análisis de ángulos de contacto ---")
    try:
        # 1. Validar existencia del archivo
        if not os.path.exists('resultados_completos.xlsx'):
            raise FileNotFoundError("No se encontró el archivo 'resultados_completos.xlsx'")

        # 2. Cargar datos con validación de columnas
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        required_cols = ['Contorno_x', 'Contorno_y', 'Imagen', 'Tiempo (s)']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"El archivo Excel no contiene las columnas requeridas: {required_cols}")

        # 3. Procesar con parámetros más robustos
        print("Procesando contornos...")
        df_ajustes = ajustar_contornos(df, grado_polinomio=3, suavizado_spline=1.0)

        # 4. Calcular ángulos con altura de contacto en µm (ajustar según necesidad)
        print("Calculando ángulos...")
        df_angulos = calcular_angulo_contacto(df_ajustes, altura_contacto=50)

        # 5. Exportar y graficar con más información
        df_angulos.to_excel('resultados_completos2.xlsx', index=False)
        graficar_resultados(df_angulos)

        # 6. Análisis mejorado
        print("\n--- Resultados ---")
        if not df_angulos.empty:
            print(
                f"Ángulos izquierdos: μ = {df_angulos['Angulo_izq'].mean():.1f}° ± {df_angulos['Angulo_izq'].std():.1f}°")
            print(
                f"Ángulos derechos: μ = {df_angulos['Angulo_der'].mean():.1f}° ± {df_angulos['Angulo_der'].std():.1f}°")

            # Calcular porcentaje de ángulos dinámicos vs estáticos
            if 'Tipo_angulo' in df_angulos:
                counts = df_angulos['Tipo_angulo'].value_counts(normalize=True)
                print(f"\nDistribución de tipos de ángulo:")
                for tipo, porcentaje in counts.items():
                    print(f"- {tipo}: {porcentaje:.1%}")

        return df_angulos

    except Exception as e:
        print(f"\nERROR en Ejercicio 2: {str(e)}")
        return None