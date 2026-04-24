"""Report generator for creating PDF evaluation reports."""
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import PolarAxes
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
from reportlab.graphics.widgets.markingmodules import Banner
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """Generate PDF evaluation reports with radar charts."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_radar_chart(self, scores: Dict[str, float], output_path: str):
        """Generate radar chart for 4 evaluation dimensions."""
        categories = ['Sprint\n(反应)', 'Tracking\n(追踪)', 'League\n(对抗)', 'Boundary\n(边界)']
        values = [
            scores.get('sprint', 0),
            scores.get('tracking', 0),
            scores.get('league', 0),
            scores.get('boundary', 0)
        ]
        values += values[:1]  # Close the polygon

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

    def _calculate_scores(self, results: Dict) -> Dict[str, float]:
        """Calculate normalized scores (0-100) for each task."""
        scores = {}

        # Sprint: based on average catch time (faster = better)
        if results.get('sprint') and results['sprint'].get('catch_times'):
            catch_times = results['sprint']['catch_times']
            avg_time = np.mean(catch_times)
            # Score: 100 if < 1s, 0 if > 6s
            scores['sprint'] = max(0, min(100, (6 - avg_time) / 5 * 100))
        else:
            scores['sprint'] = 0

        # Tracking: based on average RMSE (lower = better)
        if results.get('tracking') and results['tracking'].get('rmse_list'):
            rmse_list = results['tracking']['rmse_list']
            avg_rmse = np.mean(rmse_list)
            # Score: 100 if < 0.5, 0 if > 5
            scores['tracking'] = max(0, min(100, (5 - avg_rmse) / 4.5 * 100))
        else:
            scores['tracking'] = 0

        # League: based on survival time and caught status
        if results.get('league'):
            is_caught = results['league'].get('is_caught', True)
            survival_time = results['league'].get('survival_time', 0)
            if not is_caught:
                scores['league'] = 100  # Survived full 30s
            else:
                scores['league'] = max(0, survival_time / 30 * 100)
        else:
            scores['league'] = 0

        # Boundary: based on range of motion
        if results.get('boundary'):
            b = results['boundary']
            range_x = b.get('max_x', 0) - b.get('min_x', 0)
            range_y = b.get('max_y', 0) - b.get('min_y', 0)
            total_range = range_x + range_y
            # Score: 100 if total range > 20, 0 if < 5
            scores['boundary'] = max(0, min(100, (total_range - 5) / 15 * 100))
        else:
            scores['boundary'] = 0

        return scores

    def _estimate_clinical_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Estimate clinical scale scores (FMA-UE and ARAT) based on M-HECS scores."""
        total = scores.get('sprint', 0) * 0.2 + scores.get('tracking', 0) * 0.3 + \
                scores.get('league', 0) * 0.3 + scores.get('boundary', 0) * 0.2

        # Rough estimation based on total score
        # FMA-UE: 0-66, ARAT: 0-57
        fma_estimate = total / 100 * 50 + 10  # Range: 10-60
        arat_estimate = total / 100 * 45 + 5    # Range: 5-50

        return {
            'fma_ue': round(fma_estimate, 1),
            'arat': round(arat_estimate, 1),
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
            results: Dictionary containing evaluation results for all 4 tasks
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

        # Create PDF document
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

        # Title
        story.append(Paragraph("M-HECS 康复评估报告", title_style))
        story.append(Spacer(1, 0.3*inch))

        # Patient info
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

        # Sprint
        if results.get('sprint'):
            story.append(Paragraph("1. Sprint（反应与爆发力）", heading_style))
            sprint = results['sprint']
            sprint_data = [['指标', '数值']]
            for i, (ct, pv) in enumerate(zip(sprint['catch_times'], sprint['peak_vels'])):
                sprint_data.append([f'第{i+1}次 catch time', f'{ct:.2f}s'])
                sprint_data.append([f'第{i+1}次 peak velocity', f'{pv:.2f}'])
            sprint_data.append(['平均 catch time', f'{np.mean(sprint["catch_times"]):.2f}s'])
            sprint_data.append(['平均 peak velocity', f'{np.mean(sprint["peak_vels"]):.2f}'])

            t = Table(sprint_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E3F2FD')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['sprint']:.1f} / 100", body_style))

        story.append(Spacer(1, 0.2*inch))

        # Tracking
        if results.get('tracking'):
            story.append(Paragraph("2. Tracking（多轨迹追踪）", heading_style))
            tracking = results['tracking']
            tracking_data = [
                ['指标', '数值'],
                ['平均 RMSE', f'{np.mean(tracking["rmse_list"]):.3f}'],
                ['最大 RMSE', f'{max(tracking["rmse_list"]):.3f}'],
                ['最小 RMSE', f'{min(tracking["rmse_list"]):.3f}'],
                ['平均 Jerk', f'{np.mean(tracking["jerk_list"]):.3f}']
            ]
            t = Table(tracking_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F5E9')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['tracking']:.1f} / 100", body_style))

        story.append(Spacer(1, 0.2*inch))

        # League
        if results.get('league'):
            story.append(Paragraph("3. LeagueGame（对抗与安全距离）", heading_style))
            league = results['league']
            league_data = [
                ['指标', '数值'],
                ['是否被抓到', '是' if league['is_caught'] else '否'],
                ['生存时间', f'{league["survival_time"]:.1f}s / 30s'],
                ['最小距离', f'{min(league["dist_list"]):.2f}'],
                ['平均距离', f'{np.mean(league["dist_list"]):.2f}']
            ]
            t = Table(league_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFF3E0')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['league']:.1f} / 100", body_style))

        story.append(Spacer(1, 0.2*inch))

        # Boundary
        if results.get('boundary'):
            story.append(Paragraph("4. Boundary（活动范围与稳定性）", heading_style))
            boundary = results['boundary']
            boundary_data = [
                ['指标', '数值'],
                ['X 范围', f'{boundary["min_x"]:.2f} - {boundary["max_x"]:.2f}'],
                ['Y 范围', f'{boundary["min_y"]:.2f} - {boundary["max_y"]:.2f}'],
                ['总范围', f'{(boundary["max_x"]-boundary["min_x"] + boundary["max_y"]-boundary["min_y"]):.2f}'],
                ['平均速度', f'{np.mean(boundary["vel_list"]):.3f}']
            ]
            t = Table(boundary_data, colWidths=[2.5*inch, 2.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3E5F5')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Paragraph(f"维度得分: {scores['boundary']:.1f} / 100", body_style))

        # Build PDF
        doc.build(story)

        # Cleanup temp radar image
        if os.path.exists(radar_path):
            os.remove(radar_path)

        return output_path
