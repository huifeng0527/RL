"""Report generator for creating PDF evaluation reports."""
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """Generate PDF evaluation reports with radar charts."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_radar_chart(self, scores: Dict[str, float], output_path: str):
        """Generate radar chart for 5 evaluation dimensions."""
        categories = ['Rapid\nReach', 'Tracking', 'Workspace', 'Rhythm', 'Line\nTracing']
        values = [
            scores.get('rapid_reach', scores.get('sprint', 0)),
            scores.get('continuous_tracking', scores.get('tracking', 0)),
            scores.get('workspace_exploration', scores.get('boundary', 0)),
            scores.get('rhythmic_synchronization', 0),
            scores.get('constrained_line_tracing', 0),
        ]
        values += values[:1]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3')
        ax.fill(angles, values, alpha=0.25, color='#2196F3')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.title('M-HECS 评估雷达图', size=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def _normalize_score(self, value, bound_0, bound_100):
        """Min-Max scaling with clinical boundaries."""
        score = 100.0 * (value - bound_0) / (bound_100 - bound_0)
        return max(0.0, min(100.0, score))

    def _calculate_scores(self, results: Dict) -> Dict[str, float]:
        """Calculate normalized scores (0-100) for the 5-task assessment."""
        scores = {}

        # T1: Rapid Reach (only average movement time)
        rapid = results.get('rapid_reach') or results.get('sprint')
        if rapid:
            avg_move_time = np.mean(rapid.get('movement_times') or rapid.get('catch_times') or [6.0])
            scores['rapid_reach'] = self._normalize_score(avg_move_time, 5.0, 0.8)
        else:
            scores['rapid_reach'] = 0

        # T2: Continuous Tracking
        tracking = results.get('continuous_tracking') or results.get('tracking')
        if tracking:
            rmse = tracking.get('rmse_list', [])
            avg_rmse = np.mean(rmse) if rmse else tracking.get('mean_error', 2.0) or 2.0
            loss_rate = tracking.get('target_loss_rate')
            if loss_rate is None:
                loss_rate = np.mean(np.array(rmse) > 1.5) if rmse else 1.0
            jerk_list = tracking.get('jerk_list', [])
            mean_jerk = np.mean(jerk_list) if jerk_list else 3.0
            scores['continuous_tracking'] = (
                self._normalize_score(avg_rmse, 2.0, 0.0) * 0.55 +
                self._normalize_score(loss_rate, 1.0, 0.0) * 0.25 +
                self._normalize_score(mean_jerk, 3.0, 0.3) * 0.20
            )
        else:
            scores['continuous_tracking'] = 0

        # T3: Workspace Exploration (range of motion + stability)
        boundary = results.get('workspace_exploration') or results.get('adaptive_boundary_challenge') or results.get('boundary')
        if boundary:
            range_x = max(0, boundary.get('max_x', 0) - boundary.get('min_x', 0))
            range_y = max(0, boundary.get('max_y', 0) - boundary.get('min_y', 0))
            max_range_x = 15 - 2  # w_env - margins
            max_range_y = 10 - 2  # h_env - margins
            rom = (range_x * range_y) / max(max_range_x * max_range_y, 1.0)  # 0~1
            vel_list = boundary.get('vel_list', [])
            if len(vel_list) >= 2:
                cv = np.std(vel_list) / max(np.mean(vel_list), 1e-6)
                stability = max(0.0, 1.0 - cv / 2.0)
            else:
                stability = 0.5
            scores['workspace_exploration'] = (
                self._normalize_score(rom, 0.0, 1.0) * 0.60 +
                self._normalize_score(stability, 0.0, 1.0) * 0.40
            )
        else:
            scores['workspace_exploration'] = 0

        # T4: Rhythmic Synchronization (only average response time)
        rhythm = results.get('rhythmic_synchronization') or results.get('rhythmic_switching')
        if rhythm:
            valid_times = [t for t in rhythm.get('response_times', []) if t is not None]
            if valid_times:
                avg_response = np.mean(valid_times)
                scores['rhythmic_synchronization'] = self._normalize_score(avg_response, 2.0, 0.5)
            else:
                scores['rhythmic_synchronization'] = 0
        else:
            scores['rhythmic_synchronization'] = 0

        # T5: Constrained Line Tracing (completion time + lateral accuracy)
        line_trace = results.get('constrained_line_tracing')
        if line_trace:
            completion_times = line_trace.get('completion_times', [])
            avg_completion = np.mean(completion_times) if completion_times else 10.0
            mean_errors = [e for e in line_trace.get('mean_lateral_errors', []) if e is not None]
            avg_lateral = np.mean(mean_errors) if mean_errors else 1.0
            scores['constrained_line_tracing'] = (
                self._normalize_score(avg_completion, 10.0, 2.0) * 0.50 +
                self._normalize_score(avg_lateral, 1.0, 0.0) * 0.50
            )
        else:
            scores['constrained_line_tracing'] = 0

        task_keys = [
            'rapid_reach',
            'continuous_tracking',
            'workspace_exploration',
            'rhythmic_synchronization',
            'constrained_line_tracing',
        ]
        scores['total'] = float(np.mean([scores[key] for key in task_keys]))

        # Legacy aliases for backward compat
        scores['sprint'] = scores['rapid_reach']
        scores['tracking'] = scores['continuous_tracking']
        scores['boundary'] = scores['workspace_exploration']
        scores['league'] = 0
        return scores

    def _estimate_clinical_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Estimate clinical scale scores (FMA-UE and ARAT) based on M-HECS scores."""
        s_reach = scores.get('rapid_reach', 0) / 100.0
        s_tracking = scores.get('continuous_tracking', 0) / 100.0
        s_workspace = scores.get('workspace_exploration', 0) / 100.0
        s_rhythm = scores.get('rhythmic_synchronization', 0) / 100.0
        s_line = scores.get('constrained_line_tracing', 0) / 100.0

        total = scores.get('total', 100.0 * np.mean([s_reach, s_tracking, s_workspace, s_rhythm, s_line]))

        est_fma = 66.0 * (0.15 * s_reach + 0.30 * s_tracking + 0.30 * s_workspace + 0.15 * s_rhythm + 0.10 * s_line)
        est_arat = 57.0 * (0.25 * s_reach + 0.15 * s_tracking + 0.20 * s_workspace + 0.15 * s_rhythm + 0.25 * s_line)

        return {
            'fma_ue': round(est_fma, 1),
            'arat': round(est_arat, 1),
            'total': round(total, 1)
        }

    def generate_report(
        self,
        patient_name: str,
        session_date: datetime,
        results: Dict,
        output_filename: str = None
    ) -> str:
        """
        Generate a complete PDF evaluation report.

        Args:
            patient_name: Patient name
            session_date: Evaluation date
            results: Dictionary containing evaluation results for all 5 tasks
            output_filename: Optional output filename

        Returns:
            Path to generated PDF file
        """
        scores = self._calculate_scores(results)
        clinical = self._estimate_clinical_scores(scores)

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"report_{patient_name}_{timestamp}.pdf"

        output_path = os.path.join(self.output_dir, output_filename)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1976D2')
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#424242')
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10
        )

        story = []

        story.append(Paragraph("M-HECS 康复评估报告", title_style))
        story.append(Spacer(1, 0.3*inch))

        patient_info = [
            ['患者姓名:', patient_name],
            ['评估日期:', session_date.strftime('%Y年%m月%d日 %H:%M')],
            ['报告生成:', datetime.now().strftime('%Y年%m月%d日 %H:%M')]
        ]
        info_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5*inch))

        # Radar chart
        radar_path = os.path.join(self.output_dir, 'temp_radar.png')
        self._generate_radar_chart(scores, radar_path)
        story.append(Image(radar_path, width=4*inch, height=4*inch))
        story.append(Spacer(1, 0.3*inch))

        # Clinical estimation
        story.append(Paragraph("临床评分估算", heading_style))
        clinical_text = f"""
        基于 M-HECS 总分 ({clinical['total']}分)，估算临床量表得分：<br/>
        <b>FMA-UE（上肢 Fugl-Meyer 评估）:</b> 约 {clinical['fma_ue']} / 66 分<br/>
        <b>ARAT（动作研究手臂测试）:</b> 约 {clinical['arat']} / 57 分<br/><br/>
        <i>注：以上为估算值，仅供参考，实际临床评估应由专业人员完成。</i>
        """
        story.append(Paragraph(clinical_text, body_style))
        story.append(Spacer(1, 0.3*inch))

        # Task details
        story.append(PageBreak())
        story.append(Paragraph("任务详细结果", heading_style))

        # T1: Rapid Reach
        rapid = results.get('rapid_reach') or results.get('sprint')
        if rapid:
            story.append(Paragraph("1. Rapid Reach（快速到达）", heading_style))
            movement_times = rapid.get('movement_times', [])
            rapid_data = [['指标', '数值']]
            if movement_times:
                rapid_data.append(['平均运动时间', f'{np.mean(movement_times):.2f}s'])
                rapid_data.append(['最短时间', f'{min(movement_times):.2f}s'])
                rapid_data.append(['最长时间', f'{max(movement_times):.2f}s'])

            t = Table(rapid_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E3F2FD')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['rapid_reach']:.1f} / 100", body_style))

        story.append(Spacer(1, 0.2*inch))

        # T2: Continuous Tracking
        ct = results.get('continuous_tracking') or results.get('tracking')
        if ct:
            story.append(Paragraph("2. Continuous Tracking（连续追踪）", heading_style))
            rmse_list = ct.get('rmse_list', [])
            jerk_list = ct.get('jerk_list', [])
            ct_data = [['指标', '数值']]
            if rmse_list:
                ct_data.append(['平均 RMSE', f'{np.mean(rmse_list):.3f}'])
                ct_data.append(['最大 RMSE', f'{max(rmse_list):.3f}'])
            if jerk_list:
                ct_data.append(['平均 Jerk', f'{np.mean(jerk_list):.3f}'])
            if ct.get('target_loss_rate') is not None:
                ct_data.append(['目标丢失率', f'{ct["target_loss_rate"]:.1%}'])

            t = Table(ct_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F5E9')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['continuous_tracking']:.1f} / 100", body_style))

        story.append(Spacer(1, 0.2*inch))

        # T3: Workspace Exploration
        ws = results.get('workspace_exploration') or results.get('adaptive_boundary_challenge') or results.get('boundary')
        if ws:
            story.append(Paragraph("3. Workspace Exploration（可及空间与稳定性）", heading_style))
            ws_data = [['指标', '数值']]
            if ws.get('min_x') is not None:
                range_x = ws['max_x'] - ws['min_x']
                range_y = ws['max_y'] - ws['min_y']
                ws_data.append(['X 活动范围', f'{range_x:.2f}'])
                ws_data.append(['Y 活动范围', f'{range_y:.2f}'])
                ws_data.append(['覆盖面积', f'{range_x * range_y:.2f}'])
            vel_list = ws.get('vel_list', [])
            if vel_list:
                ws_data.append(['平均速度', f'{np.mean(vel_list):.3f}'])
                ws_data.append(['速度标准差', f'{np.std(vel_list):.3f}'])

            t = Table(ws_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3E5F5')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['workspace_exploration']:.1f} / 100", body_style))

        story.append(Spacer(1, 0.2*inch))

        # T4: Rhythmic Synchronization
        rhythm = results.get('rhythmic_synchronization') or results.get('rhythmic_switching')
        if rhythm:
            story.append(Paragraph("4. Rhythmic Synchronization（节律同步）", heading_style))
            valid_times = [t for t in rhythm.get('response_times', []) if t is not None]
            rhythm_data = [['指标', '数值']]
            if valid_times:
                rhythm_data.append(['平均响应时间', f'{np.mean(valid_times):.2f}s'])
                rhythm_data.append(['最快响应', f'{min(valid_times):.2f}s'])
                rhythm_data.append(['最慢响应', f'{max(valid_times):.2f}s'])
            total = max(len(rhythm.get('beat_times', [])), 1)
            rhythm_data.append(['未响应次数', str(rhythm.get('miss_count', 0))])

            t = Table(rhythm_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FCE4EC')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['rhythmic_synchronization']:.1f} / 100", body_style))

        story.append(Spacer(1, 0.2*inch))

        # T5: Constrained Line Tracing
        lt = results.get('constrained_line_tracing')
        if lt:
            story.append(Paragraph("5. Constrained Line Tracing（受限直线描画）", heading_style))
            line_specs = lt.get('line_specs', [])
            successes = lt.get('successes', [])
            completion_times = lt.get('completion_times', [])
            mean_errors = lt.get('mean_lateral_errors', [])
            lt_data = [['指标', '数值']]
            if completion_times:
                lt_data.append(['平均完成时间', f'{np.mean(completion_times):.2f}s'])
            if mean_errors:
                valid_errors = [e for e in mean_errors if e is not None]
                if valid_errors:
                    lt_data.append(['平均横向误差', f'{np.mean(valid_errors):.3f}'])
            for i, spec in enumerate(line_specs):
                name = spec.get('name', f'Line {i+1}')
                s = '✓' if i < len(successes) and successes[i] else '✗'
                err = f'{mean_errors[i]:.3f}' if i < len(mean_errors) and mean_errors[i] is not None else '-'
                ct_val = f'{completion_times[i]:.2f}s' if i < len(completion_times) else '-'
                lt_data.append([f'{name} 完成', s])
                lt_data.append([f'{name} 横向误差', err])
                lt_data.append([f'{name} 完成时间', ct_val])

            t = Table(lt_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E0F7FA')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['constrained_line_tracing']:.1f} / 100", body_style))

        # Build PDF
        doc.build(story)

        if os.path.exists(radar_path):
            os.remove(radar_path)

        return output_path
