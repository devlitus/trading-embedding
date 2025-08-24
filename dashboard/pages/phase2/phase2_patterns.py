#!/usr/bin/env python3
"""
Datos de patrones para Fase 2 - Análisis Técnico
"""

def get_pattern_info():
    """Obtener información detallada de todos los patrones"""
    return {
        "triangle": {
            "name": "Triángulo",
            "description": "Patrón de consolidación donde el precio se mueve entre líneas de tendencia convergentes.",
            "criteria": [
                "Mínimo 4 puntos de contacto (2 máximos y 2 mínimos)",
                "Líneas de tendencia convergentes",
                "Volumen decreciente durante la formación",
                "Ruptura con incremento de volumen"
            ],
            "theory": "Basado en la teoría de Dow y análisis técnico clásico. Representa un período de indecisión del mercado antes de una ruptura direccional.",
            "reliability": "Alta (70-80% de efectividad)",
            "timeframe": "Funciona mejor en marcos temporales de 1h o superiores"
        },
        "rectangle": {
            "name": "Rectángulo",
            "description": "Patrón de consolidación horizontal donde el precio oscila entre niveles de soporte y resistencia paralelos.",
            "criteria": [
                "Mínimo 4 puntos de contacto en niveles horizontales",
                "Soporte y resistencia claramente definidos",
                "Rango de precio relativamente estable",
                "Volumen variable durante la formación"
            ],
            "theory": "Representa equilibrio entre compradores y vendedores. Basado en conceptos de soporte/resistencia de la teoría técnica clásica.",
            "reliability": "Media-Alta (60-75% de efectividad)",
            "timeframe": "Efectivo en todos los marcos temporales"
        },
        "channel": {
            "name": "Canal",
            "description": "Patrón donde el precio se mueve entre dos líneas de tendencia paralelas (canal alcista, bajista o lateral).",
            "criteria": [
                "Dos líneas de tendencia paralelas",
                "Mínimo 3 puntos de contacto por línea",
                "Precio respeta los límites del canal",
                "Tendencia direccional clara"
            ],
            "theory": "Basado en la teoría de tendencias de Charles Dow. Los canales representan movimientos ordenados del mercado.",
            "reliability": "Alta (75-85% de efectividad)",
            "timeframe": "Más confiable en marcos temporales largos (4h+)"
        },
        "head_and_shoulders": {
            "name": "Cabeza y Hombros",
            "description": "Patrón de reversión que indica el final de una tendencia alcista, formado por tres picos con el central más alto.",
            "criteria": [
                "Tres picos: hombro izquierdo, cabeza, hombro derecho",
                "La cabeza debe ser el pico más alto",
                "Línea de cuello conecta los mínimos",
                "Volumen decreciente en la formación"
            ],
            "theory": "Patrón clásico de reversión identificado por Richard Schabacker y popularizado por Edwards & Magee.",
            "reliability": "Muy Alta (80-90% de efectividad)",
            "timeframe": "Más efectivo en marcos temporales diarios o semanales"
        },
        "double_top": {
            "name": "Doble Techo",
            "description": "Patrón de reversión bajista formado por dos picos de altura similar separados por un valle.",
            "criteria": [
                "Dos picos de altura similar (±3%)",
                "Valle intermedio claramente definido",
                "Ruptura del soporte del valle",
                "Volumen confirmatorio en la ruptura"
            ],
            "theory": "Indica agotamiento de la presión compradora. Concepto desarrollado en el análisis técnico clásico.",
            "reliability": "Alta (70-80% de efectividad)",
            "timeframe": "Funciona en todos los marcos temporales"
        },
        "double_bottom": {
            "name": "Doble Suelo",
            "description": "Patrón de reversión alcista formado por dos mínimos de altura similar separados por un pico.",
            "criteria": [
                "Dos mínimos de altura similar (±3%)",
                "Pico intermedio claramente definido",
                "Ruptura de la resistencia del pico",
                "Volumen confirmatorio en la ruptura"
            ],
            "theory": "Indica agotamiento de la presión vendedora. Patrón complementario al doble techo.",
            "reliability": "Alta (70-80% de efectividad)",
            "timeframe": "Funciona en todos los marcos temporales"
        }
    }

def get_pattern_colors(confidence):
    """Obtener colores para visualización de patrones según confianza"""
    if confidence > 0.7:
        return {
            'color': 'rgba(0, 0, 139, 0.3)',
            'border_color': '#00008B',
            'text_color': '#FFFFFF',
            'bg_color': 'rgba(0, 0, 139, 0.9)',
            'dash_pattern': 'solid',
            'symbol': '🔵'
        }
    elif confidence > 0.5:
        return {
            'color': 'rgba(255, 140, 0, 0.3)',
            'border_color': '#FF8C00',
            'text_color': '#000000',
            'bg_color': 'rgba(255, 255, 255, 0.95)',
            'dash_pattern': 'dash',
            'symbol': '🔶'
        }
    else:
        return {
            'color': 'rgba(220, 20, 60, 0.3)',
            'border_color': '#DC143C',
            'text_color': '#FFFFFF',
            'bg_color': 'rgba(220, 20, 60, 0.9)',
            'dash_pattern': 'dot',
            'symbol': '🔴'
        }