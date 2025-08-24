import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from ejercicio1 import procesar_todas_imagenes
from ejercicio3 import encontrar_puntos_contacto, calcular_perimetro

def verificar_division_contorno(imagen_idx=0, mostrar_plot=True):
    """
    Verifica cómo se divide el contorno para calcular perímetros izquierdo y derecho.
    
    Args:
        imagen_idx: Índice de la imagen a analizar (0-based)
        mostrar_plot: Si mostrar el gráfico de verificación
    """
    print(f"\n=== VERIFICACIÓN DE DIVISIÓN DE CONTORNO - IMAGEN {imagen_idx + 1} ===")
    
    # Cargar datos
    try:
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
    except:
        print("No se encontró el archivo Excel. Procesando imágenes...")
        df = procesar_todas_imagenes('TP4_imagenes', 30)  # Solo primeras 30 para test
    
    if imagen_idx >= len(df):
        print(f"Error: Solo hay {len(df)} imágenes disponibles")
        return
    
    # Obtener datos de la imagen específica
    row = df.iloc[imagen_idx]
    contorno_x = np.array(json.loads(row['Contorno_x']))
    contorno_y = np.array(json.loads(row['Contorno_y']))
    centro_x = row['Centroide_x (µm)']
    centro_y = row['Centroide_y (µm)']
    
    # Crear contorno como matriz [y, x] (formato del ejercicio3)
    contorno = np.column_stack((contorno_y, contorno_x))
    
    # Encontrar puntos de contacto
    idx_izq, idx_der = encontrar_puntos_contacto(contorno)
    
    print(f"Puntos de contacto encontrados:")
    print(f"  - Índice izquierdo: {idx_izq} -> posición (y={contorno[idx_izq, 0]:.1f}, x={contorno[idx_izq, 1]:.1f})")
    print(f"  - Índice derecho: {idx_der} -> posición (y={contorno[idx_der, 0]:.1f}, x={contorno[idx_der, 1]:.1f})")
    
    # Verificar si los índices están en orden correcto
    if idx_izq > idx_der:
        print(f"¡ATENCIÓN! Los índices se intercambiaron: idx_izq ({idx_izq}) > idx_der ({idx_der})")
        idx_izq, idx_der = idx_der, idx_izq
        print(f"Después del intercambio: idx_izq={idx_izq}, idx_der={idx_der}")
    
    # División actual del código
    contorno_izq = contorno[idx_izq:idx_der + 1]
    contorno_der = np.vstack((contorno[idx_der:], contorno[:idx_izq + 1]))
    
    print(f"\nDivisión del contorno:")
    print(f"  - Contorno izquierdo: {len(contorno_izq)} puntos (índices {idx_izq} a {idx_der})")
    print(f"  - Contorno derecho: {len(contorno_der)} puntos (índices {idx_der} a fin + inicio a {idx_izq})")
    
    # Calcular perímetros
    escala = 4.13e-6
    try:
        perim_izq = calcular_perimetro(contorno_izq, escala)
        perim_der = calcular_perimetro(contorno_der, escala)
        
        print(f"\nPerímetros calculados:")
        print(f"  - Perímetro izquierdo: {perim_izq:.6f} m")
        print(f"  - Perímetro derecho: {perim_der:.6f} m")
        print(f"  - Diferencia relativa: {abs(perim_izq - perim_der) / max(perim_izq, perim_der) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error calculando perímetros: {e}")
        perim_izq = perim_der = np.nan
    
    # Verificar continuidad del contorno derecho
    if len(contorno_der) > 1:
        # Verificar salto entre el último punto de la primera parte y el primer punto de la segunda
        punto_union1 = contorno[idx_der]  # Último punto de contorno[idx_der:]
        punto_union2 = contorno[0]       # Primer punto de contorno[:idx_izq + 1]
        distancia_salto = np.sqrt((punto_union1[0] - punto_union2[0])**2 + 
                                 (punto_union1[1] - punto_union2[1])**2)
        print(f"\nVerificación de continuidad del contorno derecho:")
        print(f"  - Distancia del 'salto' en la unión: {distancia_salto:.2f} píxeles")
        if distancia_salto > 10:  # Umbral arbitrario
            print("  - ¡ADVERTENCIA! Posible discontinuidad grande en el contorno derecho")
    
    if mostrar_plot:
        visualizar_division_contorno(contorno, contorno_izq, contorno_der, 
                                   idx_izq, idx_der, centro_x, centro_y, imagen_idx)
    
    return {
        'imagen': imagen_idx,
        'idx_izq': idx_izq,
        'idx_der': idx_der,
        'perim_izq': perim_izq,
        'perim_der': perim_der,
        'puntos_izq': len(contorno_izq),
        'puntos_der': len(contorno_der)
    }

def visualizar_division_contorno(contorno, contorno_izq, contorno_der, 
                               idx_izq, idx_der, centro_x, centro_y, imagen_idx):
    """Visualiza cómo se divide el contorno."""
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Contorno completo con puntos de contacto
    plt.subplot(2, 2, 1)
    plt.plot(contorno[:, 1], contorno[:, 0], 'k-', alpha=0.7, label='Contorno completo')
    plt.plot(contorno[idx_izq, 1], contorno[idx_izq, 0], 'go', markersize=8, label=f'Punto izq (idx={idx_izq})')
    plt.plot(contorno[idx_der, 1], contorno[idx_der, 0], 'ro', markersize=8, label=f'Punto der (idx={idx_der})')
    plt.plot(centro_x, centro_y, 'b+', markersize=10, label='Centroide')
    plt.gca().invert_yaxis()  # Invertir Y para que coincida con coordenadas de imagen
    plt.title(f'Imagen {imagen_idx + 1}: Contorno y puntos de contacto')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Contorno izquierdo
    plt.subplot(2, 2, 2)
    if len(contorno_izq) > 0:
        plt.plot(contorno_izq[:, 1], contorno_izq[:, 0], 'b-o', markersize=3, alpha=0.7)
        plt.title(f'Contorno izquierdo ({len(contorno_izq)} puntos)')
    else:
        plt.title('Contorno izquierdo (vacío)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Contorno derecho
    plt.subplot(2, 2, 3)
    if len(contorno_der) > 0:
        plt.plot(contorno_der[:, 1], contorno_der[:, 0], 'r-o', markersize=3, alpha=0.7)
        plt.title(f'Contorno derecho ({len(contorno_der)} puntos)')
    else:
        plt.title('Contorno derecho (vacío)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Ambos contornos superpuestos
    plt.subplot(2, 2, 4)
    if len(contorno_izq) > 0:
        plt.plot(contorno_izq[:, 1], contorno_izq[:, 0], 'b-', linewidth=2, alpha=0.8, label='Izquierdo')
    if len(contorno_der) > 0:
        plt.plot(contorno_der[:, 1], contorno_der[:, 0], 'r-', linewidth=2, alpha=0.8, label='Derecho')
    plt.plot(centro_x, centro_y, 'k+', markersize=10, label='Centroide')
    plt.gca().invert_yaxis()
    plt.title('Superposición de contornos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'verificacion_contorno_img_{imagen_idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.show()

def verificar_metodo_alternativo(imagen_idx=0):
    """
    Propone un método alternativo para dividir el contorno basado en el centroide.
    """
    print(f"\n=== MÉTODO ALTERNATIVO PARA DIVISIÓN DE CONTORNO ===")
    
    # Cargar datos
    try:
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
    except:
        print("No se encontró el archivo Excel.")
        return
    
    row = df.iloc[imagen_idx]
    contorno_x = np.array(json.loads(row['Contorno_x']))
    contorno_y = np.array(json.loads(row['Contorno_y']))
    centro_x = row['Centroide_x (µm)']
    
    contorno = np.column_stack((contorno_y, contorno_x))
    
    # Método alternativo: dividir por posición X relativa al centroide
    indices_izq = contorno[:, 1] <= centro_x  # Puntos a la izquierda del centroide
    indices_der = contorno[:, 1] >= centro_x  # Puntos a la derecha del centroide
    
    contorno_izq_alt = contorno[indices_izq]
    contorno_der_alt = contorno[indices_der]
    
    # Ordenar por Y para asegurar continuidad
    contorno_izq_alt = contorno_izq_alt[np.argsort(contorno_izq_alt[:, 0])]
    contorno_der_alt = contorno_der_alt[np.argsort(contorno_der_alt[:, 0])]
    
    print(f"Método alternativo (división por centroide):")
    print(f"  - Contorno izquierdo: {len(contorno_izq_alt)} puntos")
    print(f"  - Contorno derecho: {len(contorno_der_alt)} puntos")
    
    # Calcular perímetros con método alternativo
    escala = 4.13e-6
    try:
        perim_izq_alt = calcular_perimetro(contorno_izq_alt, escala)
        perim_der_alt = calcular_perimetro(contorno_der_alt, escala)
        
        print(f"Perímetros con método alternativo:")
        print(f"  - Perímetro izquierdo: {perim_izq_alt:.6f} m")
        print(f"  - Perímetro derecho: {perim_der_alt:.6f} m")
        print(f"  - Diferencia relativa: {abs(perim_izq_alt - perim_der_alt) / max(perim_izq_alt, perim_der_alt) * 100:.2f}%")
        
        return perim_izq_alt, perim_der_alt
        
    except Exception as e:
        print(f"Error con método alternativo: {e}")
        return np.nan, np.nan

def analisis_completo_perimetros(num_imagenes=10):
    """
    Analiza múltiples imágenes para identificar patrones en el cálculo de perímetros.
    """
    print(f"\n=== ANÁLISIS COMPLETO DE PERÍMETROS ({num_imagenes} imágenes) ===")
    
    resultados = []
    
    for i in range(min(num_imagenes, 30)):  # Limitar para no sobrecargar
        try:
            resultado = verificar_division_contorno(i, mostrar_plot=False)
            if resultado:
                resultados.append(resultado)
                print(f"Imagen {i+1}: Perim_izq={resultado['perim_izq']:.6f}, Perim_der={resultado['perim_der']:.6f}")
        except Exception as e:
            print(f"Error en imagen {i+1}: {e}")
    
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        
        print(f"\n--- ESTADÍSTICAS GENERALES ---")
        print(f"Perímetro izquierdo promedio: {df_resultados['perim_izq'].mean():.6f} ± {df_resultados['perim_izq'].std():.6f}")
        print(f"Perímetro derecho promedio: {df_resultados['perim_der'].mean():.6f} ± {df_resultados['perim_der'].std():.6f}")
        
        diferencias_rel = abs(df_resultados['perim_izq'] - df_resultados['perim_der']) / np.maximum(df_resultados['perim_izq'], df_resultados['perim_der']) * 100
        print(f"Diferencia relativa promedio: {diferencias_rel.mean():.2f}% ± {diferencias_rel.std():.2f}%")
        
        # Identificar casos problemáticos
        casos_problematicos = diferencias_rel > 10  # Mayor a 10% de diferencia
        if casos_problematicos.any():
            print(f"\nCasos con diferencias > 10%:")
            for idx in df_resultados[casos_problematicos]['imagen']:
                print(f"  - Imagen {idx + 1}: {diferencias_rel.iloc[idx]:.1f}% de diferencia")

if __name__ == "__main__":
    # Verificar una imagen específica
    print("=== VERIFICACIÓN DE CÁLCULO DE PERÍMETROS ===")
    print("Analizando la primera imagen...")
    
    verificar_division_contorno(imagen_idx=0)
    
    print("\n" + "="*60)
    verificar_metodo_alternativo(imagen_idx=0)
    
    print("\n" + "="*60)
    respuesta = input("¿Desea analizar múltiples imágenes? (s/n): ")
    if respuesta.lower() == 's':
        num = int(input("¿Cuántas imágenes analizar? (recomendado: 10): ") or "10")
        analisis_completo_perimetros(num)
