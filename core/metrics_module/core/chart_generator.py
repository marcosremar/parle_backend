"""
Chart Generator - Generate charts for metrics visualization
Uses Chart.js for interactive HTML charts
"""

import json
from typing import Dict, List, Any
import statistics


class ChartGenerator:
    """
    Generate Chart.js charts for metrics visualization
    """

    @staticmethod
    def generate_latency_chart(turns: List[Dict]) -> str:
        """Generate latency over turns chart"""

        turn_numbers = [t['turn'] for t in turns]
        latencies = [t['latency_ms'] for t in turns]

        chart_data = {
            'labels': turn_numbers,
            'datasets': [{
                'label': 'Latência (ms)',
                'data': latencies,
                'borderColor': 'rgb(75, 192, 192)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'tension': 0.1
            }]
        }

        return f"""
        <canvas id="latencyChart"></canvas>
        <script>
        new Chart(document.getElementById('latencyChart'), {{
            type: 'line',
            data: {json.dumps(chart_data)},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Latência por Turno de Conversa'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Latência (ms)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Turno'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """

    @staticmethod
    def generate_throughput_chart(turns: List[Dict]) -> str:
        """Generate tokens/second chart"""

        turn_numbers = [t['turn'] for t in turns]
        throughput = [t.get('tokens_per_second', 0) for t in turns]

        chart_data = {
            'labels': turn_numbers,
            'datasets': [{
                'label': 'Tokens/segundo',
                'data': throughput,
                'borderColor': 'rgb(255, 99, 132)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'tension': 0.1
            }]
        }

        return f"""
        <canvas id="throughputChart"></canvas>
        <script>
        new Chart(document.getElementById('throughputChart'), {{
            type: 'line',
            data: {json.dumps(chart_data)},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Throughput (Tokens/segundo)'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Tokens/segundo'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Turno'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """

    @staticmethod
    def generate_stage_breakdown_chart(turns: List[Dict]) -> str:
        """Generate stacked bar chart for stage breakdown"""

        turn_numbers = [t['turn'] for t in turns]

        # Extract stage timings
        tts_gen = [t.get('stages', {}).get('tts_generation', 0) for t in turns]
        stt_trans = [t.get('stages', {}).get('stt_transcription', 0) for t in turns]
        llm_proc = [t.get('stages', {}).get('llm_processing', 0) for t in turns]
        tts_synth = [t.get('stages', {}).get('tts_synthesis', 0) for t in turns]

        chart_data = {
            'labels': turn_numbers,
            'datasets': [
                {
                    'label': 'TTS Inicial',
                    'data': tts_gen,
                    'backgroundColor': 'rgba(255, 206, 86, 0.8)'
                },
                {
                    'label': 'STT Whisper',
                    'data': stt_trans,
                    'backgroundColor': 'rgba(54, 162, 235, 0.8)'
                },
                {
                    'label': 'LLM Ultravox',
                    'data': llm_proc,
                    'backgroundColor': 'rgba(255, 99, 132, 0.8)'
                },
                {
                    'label': 'TTS Resposta',
                    'data': tts_synth,
                    'backgroundColor': 'rgba(75, 192, 192, 0.8)'
                }
            ]
        }

        return f"""
        <canvas id="stageChart"></canvas>
        <script>
        new Chart(document.getElementById('stageChart'), {{
            type: 'bar',
            data: {json.dumps(chart_data)},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Tempo por Estágio de Processamento'
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: true,
                        title: {{
                            display: true,
                            text: 'Turno'
                        }}
                    }},
                    y: {{
                        stacked: true,
                        title: {{
                            display: true,
                            text: 'Tempo (ms)'
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """

    @staticmethod
    def generate_scenario_comparison_chart(scenarios: List[Dict]) -> str:
        """Generate comparison chart for multiple scenarios"""

        scenario_names = [s['name'] for s in scenarios]
        avg_latencies = [s['avg_latency'] for s in scenarios]
        avg_throughputs = [s['avg_throughput'] for s in scenarios]

        chart_data = {
            'labels': scenario_names,
            'datasets': [
                {
                    'label': 'Latência Média (ms)',
                    'data': avg_latencies,
                    'backgroundColor': 'rgba(255, 99, 132, 0.8)',
                    'yAxisID': 'y'
                },
                {
                    'label': 'Throughput Médio (tok/s)',
                    'data': avg_throughputs,
                    'backgroundColor': 'rgba(54, 162, 235, 0.8)',
                    'yAxisID': 'y1'
                }
            ]
        }

        return f"""
        <canvas id="comparisonChart"></canvas>
        <script>
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'bar',
            data: {json.dumps(chart_data)},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Comparação entre Cenários'
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Latência (ms)'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Throughput (tok/s)'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """

    @staticmethod
    def generate_quality_radar_chart(quality_scores: Dict[str, float]) -> str:
        """Generate radar chart for quality metrics"""

        labels = []
        values = []

        label_map = {
            'coherence_score': 'Coerência',
            'directness_score': 'Objetividade',
            'educational_value': 'Valor Educacional',
            'engagement_score': 'Engajamento',
            'naturalness_score': 'Naturalidade',
            'language_quality': 'Qualidade Linguística'
        }

        for key, label in label_map.items():
            if key in quality_scores:
                labels.append(label)
                values.append(quality_scores[key])

        chart_data = {
            'labels': labels,
            'datasets': [{
                'label': 'Scores de Qualidade',
                'data': values,
                'fill': True,
                'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                'borderColor': 'rgb(54, 162, 235)',
                'pointBackgroundColor': 'rgb(54, 162, 235)',
                'pointBorderColor': '#fff',
                'pointHoverBackgroundColor': '#fff',
                'pointHoverBorderColor': 'rgb(54, 162, 235)'
            }]
        }

        return f"""
        <canvas id="qualityRadar"></canvas>
        <script>
        new Chart(document.getElementById('qualityRadar'), {{
            type: 'radar',
            data: {json.dumps(chart_data)},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Métricas de Qualidade'
                    }}
                }},
                elements: {{
                    line: {{
                        borderWidth: 3
                    }}
                }},
                scales: {{
                    r: {{
                        angleLines: {{
                            display: true
                        }},
                        suggestedMin: 0,
                        suggestedMax: 1
                    }}
                }}
            }}
        }});
        </script>
        """

    @staticmethod
    def get_chart_js_cdn() -> str:
        """Get Chart.js CDN link"""
        return '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'

    @staticmethod
    def generate_full_report_with_charts(metrics_data: Dict[str, Any]) -> str:
        """Generate complete HTML report with all charts"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Métricas - Ultravox Pipeline</title>
            {ChartGenerator.get_chart_js_cdn()}
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: auto;
                    background: white;
                    padding: 30px;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #2196F3;
                    padding-bottom: 10px;
                }}
                .chart-container {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #fafafa;
                    border-radius: 8px;
                }}
                .grid-2 {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin: 30px 0;
                }}
                @media (max-width: 768px) {{
                    .grid-2 {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Relatório de Métricas - Análise Detalhada</h1>
        """

        # Add version information if available
        if 'versions' in metrics_data:
            from .version_tracker import VersionTracker
            html += VersionTracker.format_versions_html(metrics_data['versions'])

        # Add charts for each scenario
        for scenario_id, scenario_data in metrics_data.items():
            if scenario_id == 'versions':
                continue

            html += f"""
                <h2>{scenario_data.get('name', scenario_id)}</h2>

                <div class="grid-2">
                    <div class="chart-container">
                        {ChartGenerator.generate_latency_chart(scenario_data.get('turns', []))}
                    </div>
                    <div class="chart-container">
                        {ChartGenerator.generate_throughput_chart(scenario_data.get('turns', []))}
                    </div>
                </div>

                <div class="chart-container">
                    {ChartGenerator.generate_stage_breakdown_chart(scenario_data.get('turns', []))}
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html