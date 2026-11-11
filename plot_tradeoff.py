"""tradeoff.py 가 만든 CSV 로 trade-off 그림 3개를 그린다 (CSV -> 그림, 재현 가능).

무엇을: (1) 롱테일 노출률-NDCG@10 trade-off 곡선, (2) 사용자 성향 점수 P_u 분포,
        (3) 같은 평균 강도(s=0.05)에서 rank P_u 3분위 그룹별 효과.
왜: 숫자는 모두 CSV 에서만 읽는다. 그림을 다시 만들 때 모델을 다시 돌릴 필요가 없다.
사용: python plot_tradeoff.py [--in-dir scratch/tradeoff] [--out-dir scratch/figures]
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                       # 화면 없이 파일로만 저장하는 백엔드
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

# ---- 디자인 상수: 방식별 색/마커는 모든 그림에서 동일 -------------------------
INK, INK2, BG, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#dcdcd8'
METHODS = ['fixed', 'adaptive', 'adaptive_rank']
COLOR = {'fixed': '#2a78d6', 'adaptive': '#eb6834', 'adaptive_rank': '#1baf7a'}
MARKER = {'fixed': 'o', 'adaptive': 's', 'adaptive_rank': '^'}
LABEL = {'fixed': '고정 페널티', 'adaptive': '개인화 (기존 P$_u$)', 'adaptive_rank': '개인화 (순위 P$_u$)'}
BASE_RATIO = 0.95        # 기준선: 기본 NDCG 의 95%
GROUP_S = 0.05           # 그룹 그림에서 쓰는 평균 실효 강도
# 그룹 이름 매핑: tradeoff.py 는 rank P_u 를 3분위로 나눠 낮은 쪽부터 low/mid/high 로 이름 붙인다.
# rank P_u 가 낮다 = 비주류 성향(롱테일 선호) 이므로 low=비주류, high=주류.
GROUPS = [('low', '비주류'), ('mid', '중간'), ('high', '주류')]
SOURCE = '검증 구간, 사용자 449명'
FONT_CANDIDATES = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'Malgun Gothic',
                   'Noto Sans CJK KR', 'Noto Sans KR']


def setup_style():
    """한글 폰트와 공통 rcParams 를 설정한다.

    왜: 기본 폰트에는 한글이 없어 □ 로 깨진다. 폰트가 전혀 없으면 경고만 내고 진행한다.
    matplotlib.rcParams: 그림 전역 기본값 dict. 'axes.unicode_minus'=False 는 한글 폰트에서
    음수 부호(-)가 깨지는 것을 막는다. 'axes.facecolor'/'figure.facecolor' 는 배경색.
    font_manager.fontManager.ttflist: 설치된 폰트 목록(각 항목의 .name 이 폰트 이름).
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in FONT_CANDIDATES if n in installed), None)
    if chosen is None:      # 후보에 없으면 이름에 한글 폰트 흔적이 있는 것을 찾는다
        chosen = next((n for n in sorted(installed)
                       if any(k in n for k in ('Gothic', 'Myeongjo', 'CJK KR', 'Nanum', 'Hangul'))), None)
    if chosen is None:
        warnings.warn('한글 폰트를 찾지 못했다. 글자가 깨질 수 있다.')
    else:
        plt.rcParams['font.family'] = chosen
    plt.rcParams.update({
        'axes.unicode_minus': False, 'figure.facecolor': BG, 'axes.facecolor': BG,
        'savefig.facecolor': BG, 'text.color': INK, 'axes.labelcolor': INK2,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none'})


def style_axes(ax, grid_axis='y'):
    """연한 회색 점선 그리드(한 축만)를 켠다. ax.grid(axis, color, linestyle, linewidth): 그리드 설정.
    set_axisbelow(True) 는 그리드를 데이터 뒤로 보낸다."""
    ax.grid(axis=grid_axis, color=GRID, linestyle=':', linewidth=0.9)
    ax.set_axisbelow(True)


def save(fig, out_dir, name):
    """SVG 와 PNG(dpi 160) 로 저장하고 닫는다.
    fig.savefig(경로, dpi, bbox_inches): bbox_inches='tight' 는 여백을 잘라 라벨 잘림을 막는다."""
    for ext in ('svg', 'png'):
        fig.savefig(out_dir / f'{name}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def caption(fig, text):
    """그림 아래쪽에 보조 캡션을 넣는다. fig.text(x, y, s, ...): 그림 좌표(0~1)에 글자 배치."""
    fig.text(0.01, -0.02, text, fontsize=9, color=INK2, ha='left', va='top')


# ---- 그림 1 ------------------------------------------------------------------
def plot_curve(df, out_dir):
    """x=롱테일 노출률(%), y=NDCG@10. 방식별로 target_strength 순으로 점을 이은 곡선."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    # plt.subplots(figsize): (가로, 세로) 인치. fig 는 그림 전체, ax 는 좌표축.
    base = df[(df.method == 'fixed') & (df.target_strength == 0)].iloc[0]   # s=0 은 세 방식 공통
    base_x, base_y = base['tail'] * 100, base.ndcg
    xmax = df['tail'].max() * 100
    ax.axhline(BASE_RATIO * base_y, color='#8a8985', linestyle='--', linewidth=1.2, zorder=1)
    # ax.axhline(y, color, linestyle, linewidth): 가로 기준선 한 줄.
    ax.text(0.35, BASE_RATIO * base_y, '기본 NDCG의 95%', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=10, color=INK2)
    # 선 끝 라벨 y 위치가 겹치면 조금 벌린다
    ends = {}
    for m in METHODS:
        d = df[df.method == m].sort_values('target_strength')
        # ax.plot(x, y, color, marker, linewidth, markersize): 점을 선으로 잇는다.
        ax.plot(d['tail'] * 100, d.ndcg, color=COLOR[m], marker=MARKER[m], linewidth=1.8,
                markersize=6.5, label=LABEL[m], zorder=3)
        ends[m] = (d['tail'].iloc[-1] * 100, d.ndcg.iloc[-1])
    gap = (df.ndcg.max() - df.ndcg.min()) * 0.07
    ypos, last = {}, None
    for m in sorted(ends, key=lambda k: ends[k][1]):
        y = ends[m][1] if last is None else max(ends[m][1], last + gap)
        ypos[m], last = y, y
    for m in METHODS:
        # ax.annotate(text, xy, xytext, textcoords, arrowprops): xy 점에서 xytext 의 글자로 짧은 선을 긋는다.
        # 선 끝이 서로 가까우면 라벨 y 만 벌려(ypos) 겹침을 피한다.
        ax.annotate(LABEL[m], xy=ends[m], xytext=(xmax * 1.04, ypos[m]), textcoords='data',
                    ha='left', va='center', fontsize=10, color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    ax.plot([base_x], [base_y], 'o', color='#8a8985', markersize=12, zorder=4,
            markeredgecolor=BG, markeredgewidth=1.5)
    ax.annotate('기본 하이브리드', xy=(base_x, base_y), xytext=(16, 0), textcoords='offset points',
                fontsize=10, color=INK, va='center')
    ax.set_xlim(base_x - 0.4, xmax * 1.02)
    ax.set_xlabel('롱테일 노출률 (%)')
    ax.set_ylabel('NDCG@10')
    style_axes(ax, 'both')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_title('롱테일 노출률과 NDCG@10의 trade-off', loc='left', fontsize=14, color=INK, pad=24)
    ax.text(0, 1.02, f'방식별 곡선, 점은 목표 강도 s 순서 ({SOURCE})', transform=ax.transAxes,
            fontsize=10, color=INK2)
    fig.subplots_adjust(right=0.80)
    caption(fig, '같은 평균 실효 강도 grid로 비교. 같은 x에서 위에 있는 곡선이 더 좋은 trade-off.')
    save(fig, out_dir, 'tradeoff_curve')


# ---- 그림 2 ------------------------------------------------------------------
def plot_pref(pref, out_dir):
    """raw P_u 와 rank P_u 의 분포를 두 패널(같은 0~1 범위)로 보여 준다."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, col, title in zip(axes, ['raw_pu', 'rank_pu'], ['기존 P$_u$ (raw)', '순위 P$_u$ (rank)']):
        # ax.hist(x, bins, range, color, edgecolor): 히스토그램. edgecolor=배경색이면 막대 사이에 얇은 흰 간격.
        ax.hist(pref[col], bins=20, range=(0, 1), color='#2a78d6', edgecolor=BG, linewidth=1.0)
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.text(0.03, 0.95, f'평균 (1-P$_u$) = {(1 - pref[col]).mean():.2f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, color=INK)
        style_axes(ax, 'y')
    axes[0].set_ylabel('사용자 수')
    fig.tight_layout(rect=(0, 0, 1, 0.88))      # 위쪽 12% 는 제목/부제 자리
    fig.text(0.01, 0.97, '사용자 성향 점수 P$_u$ 분포', fontsize=14, color=INK, va='top')
    fig.text(0.01, 0.91, f'왼쪽 기존 P$_u$, 오른쪽 순위 P$_u$ ({SOURCE})', fontsize=10, color=INK2, va='top')
    save(fig, out_dir, 'preference_dist')


# ---- 그림 3 ------------------------------------------------------------------
def plot_groups(g, out_dir, s=GROUP_S):
    """rank P_u 3분위 그룹 x 방식 3개의 묶음 막대. 패널: 실효 페널티 / 롱테일 노출률 / NDCG@10."""
    g = g[(g.target_strength - s).abs() < 1e-9]
    panels = [('mean_penalty', '(a) 평균 실효 페널티', 1, '{:.3f}'),
              ('tail', '(b) 롱테일 노출률 (%)', 100, '{:.1f}'),
              ('ndcg', '(c) NDCG@10', 1, '{:.4f}')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    width = 0.26
    for ax, (col, title, scale, fmt) in zip(axes, panels):
        ymax = 0
        for i, m in enumerate(METHODS):
            vals = [g[(g.method == m) & (g.group == k)][col].iloc[0] * scale for k, _ in GROUPS]
            xs = [j + (i - 1) * width for j in range(len(GROUPS))]
            # ax.bar(x, height, width, color, label): 세로 막대. xs 는 막대 중심.
            ax.bar(xs, vals, width=width * 0.92, color=COLOR[m], label=LABEL[m], edgecolor=BG)
            for x, v in zip(xs, vals):
                ax.text(x, v, fmt.format(v), ha='center', va='bottom', fontsize=6.5, color=INK)
            ymax = max(ymax, max(vals))
        ax.set_ylim(0, ymax * 1.15)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([n for _, n in GROUPS])
        ax.set_xlabel('rank P$_u$ 3분위 그룹')
        ax.set_title(title, loc='left', fontsize=11, color=INK)
        style_axes(ax, 'y')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', ncol=3, fontsize=10, bbox_to_anchor=(0.99, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.text(0.01, 0.97, f'같은 평균 강도(s={s})에서 그룹별 효과', fontsize=14, color=INK, va='top')
    fig.text(0.01, 0.91, f'rank P$_u$ 3분위 그룹별 ({SOURCE})', fontsize=10, color=INK2, va='top')
    save(fig, out_dir, 'group_effect')


def main():
    """인자를 읽고 CSV 를 불러 그림 3개를 저장한다.
    argparse.ArgumentParser.add_argument(이름, default, help): 명령행 옵션 정의."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--in-dir', default='scratch/tradeoff', help='tradeoff.py 출력 폴더')
    ap.add_argument('--out-dir', default='scratch/figures', help='그림 저장 폴더')
    a = ap.parse_args()
    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_curve(pd.read_csv(in_dir / 'tradeoff.csv'), out_dir)
    plot_pref(pd.read_csv(in_dir / 'preference.csv'), out_dir)
    plot_groups(pd.read_csv(in_dir / 'tradeoff_groups.csv'), out_dir)
    print(f'저장 완료: {out_dir}')


if __name__ == '__main__':
    main()
