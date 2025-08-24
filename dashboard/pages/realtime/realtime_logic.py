#!/usr/bin/env python3
"""
Lógica de negocio para el módulo de monitoreo en tiempo real.

Este módulo contiene toda la lógica para obtener datos, generar alertas,
calcular métricas y procesar información en tiempo real.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .common import analyze_symbol
from .realtime_config import RealtimeConfig


@dataclass
class PriceMetrics:
    """Métricas de precio para un símbolo"""
    symbol: str
    current_price: float
    previous_price: float
    change: float
    change_pct: float
    volume: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0


@dataclass
class Alert:
    """Estructura para alertas del sistema"""
    symbol: str
    type: str  # 'bullish', 'bearish', 'warning', 'info'
    message: str
    emoji: str
    confidence: float = 0.0
    timestamp: Optional[str] = None


class RealtimeDataProcessor:
    """Procesador de datos en tiempo real"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.config = RealtimeConfig()
    
    def get_price_metrics(self, symbol: str) -> Optional[PriceMetrics]:
        """Obtiene métricas de precio para un símbolo"""
        try:
            # Obtener datos recientes
            df = self.data_manager.get_data(symbol, '1m', limit=2)
            
            if df.empty:
                return None
            
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            # Obtener datos adicionales si están disponibles
            volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0.0
            
            # Obtener datos de 24h para high/low
            df_24h = self.data_manager.get_data(symbol, '1h', limit=24)
            high_24h = df_24h['high'].max() if not df_24h.empty else current_price
            low_24h = df_24h['low'].min() if not df_24h.empty else current_price
            
            return PriceMetrics(
                symbol=symbol,
                current_price=current_price,
                previous_price=prev_price,
                change=change,
                change_pct=change_pct,
                volume=volume,
                high_24h=high_24h,
                low_24h=low_24h
            )
            
        except Exception as e:
            print(f"Error obteniendo métricas para {symbol}: {e}")
            return None
    
    def get_mini_chart_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Obtiene datos para mini gráfico"""
        try:
            points = self.config.CHART_CONFIG['mini_chart_points']
            interval = self.config.DATA_CONFIG['mini_chart_interval']
            return self.data_manager.get_data(symbol, interval, limit=points)
        except Exception as e:
            print(f"Error obteniendo datos de gráfico para {symbol}: {e}")
            return None
    
    def generate_alerts(self, symbol: str) -> List[Alert]:
        """Genera alertas para un símbolo"""
        alerts = []
        
        try:
            # Obtener datos para análisis
            interval = self.config.DATA_CONFIG['alert_analysis_interval']
            periods = self.config.DATA_CONFIG['quick_analysis_periods'][interval]
            df = self.data_manager.get_data(symbol, interval, limit=periods)
            
            if df.empty:
                return alerts
            
            # Realizar análisis
            analysis = analyze_symbol(df)
            
            # Generar alertas de RSI
            alerts.extend(self._generate_rsi_alerts(symbol, analysis))
            
            # Generar alertas de tendencia
            alerts.extend(self._generate_trend_alerts(symbol, analysis))
            
            # Generar alertas de patrones
            alerts.extend(self._generate_pattern_alerts(symbol, analysis))
            
            # Generar alertas de precio
            alerts.extend(self._generate_price_alerts(symbol, df))
            
        except Exception as e:
            alerts.append(Alert(
                symbol=symbol,
                type='warning',
                message=f"Error generando alertas: {str(e)}",
                emoji=self.config.ALERT_EMOJIS['error']
            ))
        
        return alerts
    
    def _generate_rsi_alerts(self, symbol: str, analysis) -> List[Alert]:
        """Genera alertas basadas en RSI"""
        alerts = []
        
        if 'rsi' not in analysis.indicators:
            return alerts
        
        try:
            rsi = analysis.indicators['rsi']
            
            # Handle different RSI data types
            if hasattr(rsi, 'iloc'):
                rsi_value = rsi.iloc[-1]
            elif hasattr(rsi, '__len__') and len(rsi) > 0:
                rsi_value = rsi[-1]
            else:
                rsi_value = rsi
            
            # Verificar umbrales
            if rsi_value > self.config.ALERT_THRESHOLDS['rsi_overbought']:
                alerts.append(Alert(
                    symbol=symbol,
                    type='warning',
                    message=f"{symbol}: RSI sobrecomprado ({rsi_value:.1f})",
                    emoji=self.config.ALERT_EMOJIS['overbought'],
                    confidence=min((rsi_value - 70) / 20, 1.0)
                ))
            elif rsi_value < self.config.ALERT_THRESHOLDS['rsi_oversold']:
                alerts.append(Alert(
                    symbol=symbol,
                    type='bullish',
                    message=f"{symbol}: RSI sobrevendido ({rsi_value:.1f})",
                    emoji=self.config.ALERT_EMOJIS['oversold'],
                    confidence=min((30 - rsi_value) / 20, 1.0)
                ))
                
        except Exception as e:
            print(f"Error generando alertas RSI para {symbol}: {e}")
        
        return alerts
    
    def _generate_trend_alerts(self, symbol: str, analysis) -> List[Alert]:
        """Genera alertas basadas en tendencia"""
        alerts = []
        
        try:
            trend = analysis.trend_analysis.get('current_trend')
            strength = analysis.trend_analysis.get('trend_strength', 0)
            
            threshold = self.config.ALERT_THRESHOLDS['trend_strength_high']
            
            if trend == 'Bullish' and strength > threshold:
                alerts.append(Alert(
                    symbol=symbol,
                    type='bullish',
                    message=f"{symbol}: Tendencia alcista fuerte (fuerza: {strength:.2f})",
                    emoji=self.config.ALERT_EMOJIS['bullish_trend'],
                    confidence=strength
                ))
            elif trend == 'Bearish' and strength > threshold:
                alerts.append(Alert(
                    symbol=symbol,
                    type='bearish',
                    message=f"{symbol}: Tendencia bajista fuerte (fuerza: {strength:.2f})",
                    emoji=self.config.ALERT_EMOJIS['bearish_trend'],
                    confidence=strength
                ))
                
        except Exception as e:
            print(f"Error generando alertas de tendencia para {symbol}: {e}")
        
        return alerts
    
    def _generate_pattern_alerts(self, symbol: str, analysis) -> List[Alert]:
        """Genera alertas basadas en patrones"""
        alerts = []
        
        try:
            if not analysis.patterns:
                return alerts
            
            threshold = self.config.ALERT_THRESHOLDS['pattern_confidence_high']
            high_conf_patterns = [p for p in analysis.patterns if p.confidence > threshold]
            
            if high_conf_patterns:
                pattern_names = [p.name for p in high_conf_patterns[:3]]  # Máximo 3
                alerts.append(Alert(
                    symbol=symbol,
                    type='info',
                    message=f"{symbol}: {len(high_conf_patterns)} patrón(es) detectado(s): {', '.join(pattern_names)}",
                    emoji=self.config.ALERT_EMOJIS['pattern_detected'],
                    confidence=max(p.confidence for p in high_conf_patterns)
                ))
                
        except Exception as e:
            print(f"Error generando alertas de patrones para {symbol}: {e}")
        
        return alerts
    
    def _generate_price_alerts(self, symbol: str, df: pd.DataFrame) -> List[Alert]:
        """Genera alertas basadas en cambios de precio"""
        alerts = []
        
        try:
            if len(df) < 2:
                return alerts
            
            # Calcular cambio de precio en período reciente
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-6] if len(df) >= 6 else df['close'].iloc[0]
            
            change_pct = ((current_price - prev_price) / prev_price) * 100
            threshold = self.config.ALERT_THRESHOLDS['price_change_alert']
            
            if abs(change_pct) > threshold:
                alert_type = 'bullish' if change_pct > 0 else 'bearish'
                direction = 'subida' if change_pct > 0 else 'caída'
                
                alerts.append(Alert(
                    symbol=symbol,
                    type=alert_type,
                    message=f"{symbol}: {direction} significativa del {abs(change_pct):.1f}%",
                    emoji=self.config.ALERT_EMOJIS['price_alert'],
                    confidence=min(abs(change_pct) / 10, 1.0)
                ))
                
        except Exception as e:
            print(f"Error generando alertas de precio para {symbol}: {e}")
        
        return alerts
    
    def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Obtiene resumen del mercado para los símbolos seleccionados"""
        summary = {
            'total_symbols': len(symbols),
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_change': 0.0,
            'top_gainers': [],
            'top_losers': []
        }
        
        try:
            metrics_list = []
            
            for symbol in symbols:
                metrics = self.get_price_metrics(symbol)
                if metrics:
                    metrics_list.append(metrics)
                    
                    # Clasificar tendencia
                    if metrics.change_pct > 1.0:
                        summary['bullish_count'] += 1
                    elif metrics.change_pct < -1.0:
                        summary['bearish_count'] += 1
                    else:
                        summary['neutral_count'] += 1
            
            if metrics_list:
                # Calcular promedio de cambio
                summary['avg_change'] = sum(m.change_pct for m in metrics_list) / len(metrics_list)
                
                # Top gainers y losers
                sorted_by_change = sorted(metrics_list, key=lambda x: x.change_pct, reverse=True)
                summary['top_gainers'] = sorted_by_change[:3]
                summary['top_losers'] = sorted_by_change[-3:]
                
        except Exception as e:
            print(f"Error generando resumen del mercado: {e}")
        
        return summary