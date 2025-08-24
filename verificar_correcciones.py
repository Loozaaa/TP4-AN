import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from ejercicio3 import (encontrar_puntos_contacto, calcular_perimetro, 
                       dividir_contorno_mejorado, calcular_propiedades_geometricas)

def test_correciones_perimetro():
    """
    Función principal para verificar las correcciones en el cálculo de perímetros.
    """
    print("=== VERIFICACIÓN DE CORRECCIONES EN CÁLCULO DE PERÍMETROS ===\n")
    
    try:
        # Cargar datos
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        df2 = pd.read_excel('resultados_completos2.xlsx')
        df = pd.merge(df, df2[['Imagen', 'Tiempo (s)', 'Factor_esparcimiento']], 
                     on=['Imagen', 'Tiempo (s)'], how='inner')
        
        print(f"Datos cargados: {len(df)} imágenes")
        
        # Verificar una imagen específica
        img_test = 5  # Imagen de prueba
        row = df.iloc[img_test]
        
        print(f"\n--- ANÁLISIS DETALLADO IMAGEN {img_test + 1} ---")
        
        contorno_x = np.array(json.loads(row['Contorno_x']))
        contorno_y = np.array(json.loads(row['Contorno_y']))
        contorno = np.column_stack((contorno_y, contorno_x))
        
        # Puntos de contacto
        idx_izq, idx_der = encontrar_puntos_contacto(contorno)
        print(f"Puntos de contacto: izq={idx_izq}, der={idx_der}")
        
        # División del contorno
        contorno_izq, contorno_der = dividir_contorno_mejorado(contorno, idx_izq, idx_der)
        print(f"Contorno izquierdo: {len(contorno_izq)} puntos")
        print(f"Contorno derecho: {len(contorno_der)} puntos")
        
        # Calcular perímetros
        escala = 4.13e-6
        perim_izq = calcular_perimetro(contorno_izq, escala)
        perim_der = calcular_perimetro(contorno_der, escala)
        
        print(f"Perímetro izquierdo: {perim_izq:.6f} m")
        print(f"Perímetro derecho: {perim_der:.6f} m")
        
        if max(perim_izq, perim_der) > 0:
            diferencia_rel = abs(perim_izq - perim_der) / max(perim_izq, perim_der) * 100
            simetria = 1 - abs(perim_izq - perim_der) / max(perim_izq, perim_der)
            print(f"Diferencia relativa: {diferencia_rel:.2f}%")
            print(f"Coeficiente de simetría: {simetria:.3f}")
        
        # Visualización
        visualizar_division_mejorada(contorno, contorno_izq, contorno_der, 
                                   idx_izq, idx_der, img_test)
        
        # Procesar todas las imágenes con el método corregido
        print(f"\n--- PROCESAMIENTO COMPLETO CON MÉTODO CORREGIDO ---")
        df_corregido = calcular_propiedades_geometricas(df.copy())
        
        # Estadísticas comparativas
        print(f"\nEstadísticas del método corregido:")
        print(f"Perímetro izquierdo promedio: {df_corregido['Perimetro_izq (m)'].mean():.6f} ± {df_corregido['Perimetro_izq (m)'].std():.6f}")
        print(f"Perímetro derecho promedio: {df_corregido['Perimetro_der (m)'].mean():.6f} ± {df_corregido['Perimetro_der (m)'].std():.6f}")
        print(f"Simetría promedio: {df_corregido['Simetria'].mean():.3f} ± {df_corregido['Simetria'].std():.3f}")
        
        # Verificar casos problemáticos
        diferencias = abs(df_corregido['Perimetro_izq (m)'] - df_corregido['Perimetro_der (m)']) / np.maximum(df_corregido['Perimetro_izq (m)'], df_corregido['Perimetro_der (m)']) * 100
        casos_problematicos = diferencias > 15  # Más de 15% de diferencia
        
        print(f"\nCasos con diferencias > 15%: {casos_problematicos.sum()}")
        if casos_problematicos.any():
            for idx in np.where(casos_problematicos)[0][:5]:  # Mostrar solo primeros 5
                print(f"  Imagen {idx + 1}: {diferencias.iloc[idx]:.1f}% de diferencia")
        
        # Guardar resultados corregidos
        df_corregido.to_excel('resultados_ejercicio3_corregido.xlsx', index=False)
        print(f"\nResultados guardados en 'resultados_ejercicio3_corregido.xlsx'")
        
        return True
        
    except Exception as e:
        print(f"Error en verificación: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualizar_division_mejorada(contorno, contorno_izq, contorno_der, 
                               idx_izq, idx_der, img_num):
    """Visualiza la división mejorada del contorno."""
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Contorno completo con puntos clave
    plt.subplot(2, 3, 1)
    plt.plot(contorno[:, 1], contorno[:, 0], 'k-', alpha=0.7, linewidth=2, label='Contorno completo')
    plt.plot(contorno[idx_izq, 1], contorno[idx_izq, 0], 'go', markersize=10, label=f'Punto izq ({idx_izq})')
    plt.plot(contorno[idx_der, 1], contorno[idx_der, 0], 'ro', markersize=10, label=f'Punto der ({idx_der})')
    
    # Marcar ápice
    idx_apice = np.argmin(contorno[:, 0])
    plt.plot(contorno[idx_apice, 1], contorno[idx_apice, 0], 'b^', markersize=10, label=f'Ápice ({idx_apice})')
    
    plt.gca().invert_yaxis()
    plt.title(f'Imagen {img_num + 1}: Puntos clave')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Contorno izquierdo
    plt.subplot(2, 3, 2)
    if len(contorno_izq) > 0:
        plt.plot(contorno_izq[:, 1], contorno_izq[:, 0], 'b-o', markersize=4, alpha=0.8, linewidth=2)
        plt.title(f'Contorno izquierdo\n({len(contorno_izq)} puntos)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Contorno derecho
    plt.subplot(2, 3, 3)
    if len(contorno_der) > 0:
        plt.plot(contorno_der[:, 1], contorno_der[:, 0], 'r-o', markersize=4, alpha=0.8, linewidth=2)
        plt.title(f'Contorno derecho\n({len(contorno_der)} puntos)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Superposición
    plt.subplot(2, 3, 4)
    if len(contorno_izq) > 0:
        plt.plot(contorno_izq[:, 1], contorno_izq[:, 0], 'b-', linewidth=3, alpha=0.8, label='Izquierdo')
    if len(contorno_der) > 0:
        plt.plot(contorno_der[:, 1], contorno_der[:, 0], 'r-', linewidth=3, alpha=0.8, label='Derecho')
    plt.gca().invert_yaxis()
    plt.title('Superposición')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Verificación de continuidad
    plt.subplot(2, 3, 5)
    # Calcular distancias entre puntos consecutivos
    if len(contorno_izq) > 1:
        diffs_izq = np.sqrt(np.diff(contorno_izq[:, 1])**2 + np.diff(contorno_izq[:, 0])**2)
        plt.plot(diffs_izq, 'b-', alpha=0.7, label='Izquierdo')
    
    if len(contorno_der) > 1:
        diffs_der = np.sqrt(np.diff(contorno_der[:, 1])**2 + np.diff(contorno_der[:, 0])**2)
        plt.plot(diffs_der, 'r-', alpha=0.7, label='Derecho')
    
    plt.title('Distancias entre puntos\n(continuidad)')
    plt.ylabel('Distancia (píxeles)')
    plt.xlabel('Índice de punto')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Estadísticas
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calcular estadísticas
    if len(contorno_izq) > 1 and len(contorno_der) > 1:
        escala = 4.13e-6
        perim_izq = calcular_perimetro(contorno_izq, escala)
        perim_der = calcular_perimetro(contorno_der, escala)
        
        if max(perim_izq, perim_der) > 0:
            diff_rel = abs(perim_izq - perim_der) / max(perim_izq, perim_der) * 100
            simetria = 1 - abs(perim_izq - perim_der) / max(perim_izq, perim_der)
        else:
            diff_rel = 0
            simetria = 0
        
        stats_text = f"""ESTADÍSTICAS:

Perímetro izquierdo:
{perim_izq:.6f} m

Perímetro derecho:
{perim_der:.6f} m

Diferencia relativa:
{diff_rel:.2f}%

Coeficiente simetría:
{simetria:.3f}

Puntos izquierda: {len(contorno_izq)}
Puntos derecha: {len(contorno_der)}
"""
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'verificacion_corregida_img_{img_num + 1}.png', dpi=150, bbox_inches='tight')
    plt.show()

def comparar_metodos():
    """Compara el método original vs el corregido."""
    print("\n=== COMPARACIÓN MÉTODO ORIGINAL VS CORREGIDO ===")
    
    try:
        # Cargar datos
        df = pd.read_excel('resultados_completos.xlsx', sheet_name='Datos Completos')
        df2 = pd.read_excel('resultados_completos2.xlsx')
        df = pd.merge(df, df2[['Imagen', 'Tiempo (s)', 'Factor_esparcimiento']], 
                     on=['Imagen', 'Tiempo (s)'], how='inner')
        
        # Método corregido
        df_corregido = calcular_propiedades_geometricas(df.copy())
        
        # Cargar resultados del método original (si existe)
        try:
            df_original = pd.read_excel('resultados_completos3.xlsx')
            
            print("Comparación estadística:")
            print(f"Método original - Simetría promedio: {df_original['Simetria'].mean():.3f}")
            print(f"Método corregido - Simetría promedio: {df_corregido['Simetria'].mean():.3f}")
            
            # Gráfico comparativo
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(df_original['Tiempo (s)'], df_original['Simetria'], 'r-', alpha=0.7, label='Original')
            plt.plot(df_corregido['Tiempo (s)'], df_corregido['Simetria'], 'b-', alpha=0.7, label='Corregido')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Simetría')
            plt.title('Comparación de Simetría')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(df_original['Tiempo (s)'], df_original['Perimetro_izq (m)'], 'r-', alpha=0.7, label='Original')
            plt.plot(df_corregido['Tiempo (s)'], df_corregido['Perimetro_izq (m)'], 'b-', alpha=0.7, label='Corregido')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Perímetro izquierdo (m)')
            plt.title('Comparación Perímetro Izquierdo')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.plot(df_original['Tiempo (s)'], df_original['Perimetro_der (m)'], 'r-', alpha=0.7, label='Original')
            plt.plot(df_corregido['Tiempo (s)'], df_corregido['Perimetro_der (m)'], 'b-', alpha=0.7, label='Corregido')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Perímetro derecho (m)')
            plt.title('Comparación Perímetro Derecho')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            diff_original = abs(df_original['Perimetro_izq (m)'] - df_original['Perimetro_der (m)']) / np.maximum(df_original['Perimetro_izq (m)'], df_original['Perimetro_der (m)']) * 100
            diff_corregido = abs(df_corregido['Perimetro_izq (m)'] - df_corregido['Perimetro_der (m)']) / np.maximum(df_corregido['Perimetro_izq (m)'], df_corregido['Perimetro_der (m)']) * 100
            
            plt.plot(df_original['Tiempo (s)'], diff_original, 'r-', alpha=0.7, label='Original')
            plt.plot(df_corregido['Tiempo (s)'], diff_corregido, 'b-', alpha=0.7, label='Corregido')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Diferencia relativa (%)')
            plt.title('Diferencia entre perímetros')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('comparacion_metodos.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except FileNotFoundError:
            print("No se encontró archivo con resultados del método original")
            print("Ejecutando solo análisis del método corregido...")
        
        return df_corregido
        
    except Exception as e:
        print(f"Error en comparación: {e}")
        return None

if __name__ == "__main__":
    print("VERIFICACIÓN DE CORRECCIONES EN EL EJERCICIO 3")
    print("=" * 50)
    
    # Ejecutar verificación
    if test_correciones_perimetro():
        print("\n✓ Verificación completada exitosamente")
        
        respuesta = input("\n¿Desea comparar con método original? (s/n): ")
        if respuesta.lower() == 's':
            comparar_metodos()
    else:
        print("\n✗ Error en la verificación")
