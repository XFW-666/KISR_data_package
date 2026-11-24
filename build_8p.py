"""Build the 8-page IEEE Sensors Journal manuscript directly from the JSEN template.

Template facts this implements (read out of out/_template_check/jsen_template.docx):
  * page 8.5 x 11 in, margins top/bottom 0.70 in, left/right 0.65 in,
    text width 7.2 in, two columns with 288 twips (0.20 in) of spacing,
    so one column is 3.5 in / 88 mm;
  * section 1 is single column (title / authors / abstract / index terms),
    section 2 is the continuous two-column body;
  * "Heading 1"/"Heading 2" styles carry automatic numbering (I., A.) and the
    "References" style carries automatic "[n]" numbering, so the text must not
    repeat those numbers;
  * header1.xml ships the IEEE Sensors Journal logo plus journal line for page 1,
    footer1.xml the IEEE copyright line, header2/header3 the even/odd running heads;
  * figures and tables are one column wide (template: "3.5 inches / 88 millimeters").
"""
import json
import os
import re
import shutil
import zipfile

import numpy as np
from PIL import Image
import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(ROOT, 'out', '_template_check', 'jsen_template.docx')
ORIG = os.path.join(ROOT, 'KISR_IEEE_Sensors_Journal_2col.docx')
WORK = os.path.join(ROOT, 'out', '_8p')
COL = os.path.join(WORK, 'col')
NOV = os.path.join(ROOT, 'out', '_novelty')
os.makedirs(WORK, exist_ok=True)

COL_W = Inches(3.4)       # one template column is 3.5 in; keep a small safety margin
COL_TW = 4896             # 3.4 in expressed in twips, used for table widths

# Figure order is declared once; body text refers to figures by @token@ so a
# caption can never drift from its in-text number.
FIG_ORDER = ['optics', 'chain', 'signals', 'stream', 'dalia', 'subspace', 'mech',
             'sizing', 'record', 'gate']
FIG_RE = re.compile(r'@(%s)@' % '|'.join(FIG_ORDER))


def fig_num(key):
    return FIG_ORDER.index(key) + 1


RUNNING_HEAD_TITLE = 'Wrist-PPG Heart-Rate Monitoring: Sensor Sizing Evidence'


# ---------------------------------------------------------------- figures -----
def top_crop(src, dst, lo=0.30, hi=0.75, mode='largest_end'):
    im = Image.open(src).convert('L')
    a = np.asarray(im, dtype=np.float32).mean(axis=1)
    white = a > 248
    h = len(white)
    runs, st = [], None
    for y, w in enumerate(white):
        if w and st is None:
            st = y
        if not w and st is not None:
            runs.append((st, y - st))
            st = None
    cand = [r for r in runs if r[1] >= 15 and lo * h <= r[0] <= hi * h]
    if not cand:
        cut = h
    elif mode == 'last_end':
        r = cand[-1]; cut = r[0] + r[1]
    elif mode == 'largest_start':
        cut = max(cand, key=lambda r: r[1])[0]
    else:
        r = max(cand, key=lambda r: r[1]); cut = r[0] + r[1]
    Image.open(src).convert('RGB').crop((0, 0, im.width, cut)).save(dst)
    return cut / h


z = zipfile.ZipFile(ORIG)
media = {n: z.read(n) for n in z.namelist() if n.startswith('word/media/')}


def save_media(name, path):
    with open(path, 'wb') as fh:
        fh.write(media['word/media/' + name])


save_media('image5.png', os.path.join(WORK, 'fig1_full.png'))
save_media('image6.png', os.path.join(WORK, 'fig2_src.png'))
top_crop(os.path.join(WORK, 'fig2_src.png'), os.path.join(WORK, 'fig2.png'), 0.55, 0.72, 'last_end')
save_media('image8.png', os.path.join(WORK, 'fig3_src.png'))
top_crop(os.path.join(WORK, 'fig3_src.png'), os.path.join(WORK, 'fig3.png'), 0.30, 0.65, 'largest_start')
shutil.copy(os.path.join(NOV, 'fig_cross_dataset.png'), os.path.join(WORK, 'fig4.png'))
shutil.copy(os.path.join(NOV, 'fig_transient_gate.png'), os.path.join(WORK, 'fig5.png'))

# Column-native renderings win when they exist; the wide originals are only a
# stopgap so the build stays runnable while the single-column set is produced.
FIG_SRC = {
    'fig1_col.png': os.path.join(WORK, 'fig1_full.png'),
    'fig2_col.png': os.path.join(WORK, 'fig1_full.png'),
    'fig3_col.png': os.path.join(WORK, 'fig2.png'),
    'fig4_col.png': os.path.join(WORK, 'fig3.png'),
    'fig5_col.png': os.path.join(WORK, 'fig4.png'),
    'fig6_col.png': os.path.join(WORK, 'fig5.png'),
}


def fig_path(name):
    p = os.path.join(COL, name)
    return p if os.path.exists(p) else FIG_SRC[name]


used_fallback = []      # filled by the figure loop below

# ------------------------------------------------------------------ data ------
WE = json.load(open(os.path.join(NOV, 'wrist_expanded.json')))
BI = json.load(open(os.path.join(NOV, 'bidmc_replication.json')))
TG = json.load(open(os.path.join(NOV, 'transient_gate.json')))
EQ = json.load(open(os.path.join(NOV, 'equivalence.json')))
EN = json.load(open(os.path.join(NOV, 'energy_model.json')))
DA = json.load(open(os.path.join(NOV, 'dalia_channel_law.json')))
CH_LABEL = {'ch_acc_wide': 'acc (wide-range)', 'ch_acc_ln': 'acc (low-noise)',
            'ch_gyro': 'gyro', 'ch_acc_gyro': 'acc + gyro',
            'ch_acc_gyro_mag': 'acc + gyro + mag',
            'ch_full': 'full suite (2 acc + gyro + mag)'}
CH_ORDER = ['ch_acc_wide', 'ch_acc_ln', 'ch_gyro', 'ch_acc_gyro',
            'ch_acc_gyro_mag', 'ch_full']

# ----------------------------------------------------------------- document ---
doc = docx.Document(TPL)
body = doc.element.body
for child in list(body):
    if child.tag in (qn('w:p'), qn('w:tbl')):
        body.remove(child)

# The converted template gives Title and Authors a w:framePr text frame with the
# same anchor, which Word renders on top of itself (title and author line
# overlap). Drop the frames and keep the typographic formatting of the styles.
for style_name in ('Title', 'Authors'):
    st = doc.styles[style_name]._element
    pPr = st.find(qn('w:pPr'))
    if pPr is not None:
        for fr in pPr.findall(qn('w:framePr')):
            pPr.remove(fr)


def para(text, style='PARA_Indent', bold=False, align=None, size=None):
    p = doc.add_paragraph(style=style)
    r = p.add_run(FIG_RE.sub(lambda m: 'Fig. %d' % fig_num(m.group(1)), text))
    if size:
        r.font.size = Pt(size)
    if bold:
        r.bold = True
    if align is not None:
        p.alignment = align
    return p


# Floats are built once (at the end) but placed at their first citation: a
# marker paragraph records the spot, and the finished table/figure block is
# moved onto it before saving.
MARKS, GROUPS = {}, {}


RANGE_RE = re.compile(r'\[(\d+)\]\s*-\s*\[(\d+)\]')
SINGLE_RE = re.compile(r'\[(\d+)\]')
CITE_RE = re.compile(r'\[(\d+)\](?:\s*-\s*\[(\d+)\])?')


def citation_order(t):
    out = []
    for m in CITE_RE.finditer(t):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        for n in range(a, b + 1):
            if n not in out:
                out.append(n)
    return out


def renumber_refs(doc):
    """IEEE requires reference numbers in order of first citation. The
    References style auto-numbers by paragraph order, so reorder the reference
    paragraphs to first-citation order and remap every in-text token."""
    paras = doc.paragraphs
    order = []
    for p in paras:
        if p.style.name in ('References', 'Reference Head'):
            continue
        for n in citation_order(p.text):
            if n not in order:
                order.append(n)
    refparas = [p for p in paras if p.style.name == 'References']
    if len(refparas) != len(order):
        raise SystemExit('ref QC: %d refs but %d cited' % (len(refparas), len(order)))
    new_of = {old: i + 1 for i, old in enumerate(order)}
    for p in paras:
        if p.style.name in ('References', 'Reference Head'):
            continue
        for r in p.runs:
            t = r.text

            def rr(m):
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else a
                if b > a and new_of[b] - new_of[a] == b - a:
                    return '[%d]-[%d]' % (new_of[a], new_of[b])
                return ', '.join('[%d]' % new_of[k] for k in range(a, b + 1))

            t = CITE_RE.sub(rr, t)
            if t != r.text:
                r.text = t
    body = doc.element.body
    idx = body.index(refparas[0]._p)
    for p in refparas:
        body.remove(p._p)
    for old in order:
        body.insert(idx, refparas[old - 1]._p)
        idx += 1
    return new_of


def mark(key):
    p = doc.add_paragraph(style='Normal')
    p.add_run('[[%s]]' % key)
    MARKS[key] = p._p
    return p


# --------------------------------------------- sections and page furniture --
def relid_by_part(reltype, partname):
    for rid, rel in doc.part.rels.items():
        if rel.reltype == reltype and str(rel.target_part.partname) == partname:
            return rid
    raise KeyError(partname)


def clear_refs(sectPr):
    for tag in ('w:headerReference', 'w:footerReference'):
        for el in sectPr.findall(qn(tag)):
            sectPr.remove(el)


def add_ref(sectPr, tag, typ, rid, pos):
    el = OxmlElement(tag)
    el.set(qn('w:type'), typ)
    el.set(qn('r:id'), rid)
    sectPr.insert(pos, el)
    return pos + 1


def set_footnote_pos(sectPr):
    """The template keeps footnotes beneath the column text."""
    for old in sectPr.findall(qn('w:footnotePr')):
        sectPr.remove(old)
    fp = OxmlElement('w:footnotePr')
    p = OxmlElement('w:pos')
    p.set(qn('w:val'), 'beneathText')
    fp.append(p)
    refs = sectPr.findall(qn('w:headerReference')) + sectPr.findall(qn('w:footerReference'))
    at = 0
    for el in refs:
        at = max(at, sectPr.index(el) + 1)
    sectPr.insert(at, fp)


HDR = {n: relid_by_part(RT.HEADER, '/word/%s.xml' % n) for n in ('header1', 'header2', 'header3')}
FTR1 = relid_by_part(RT.FOOTER, '/word/footer1.xml')
FTR2 = relid_by_part(RT.FOOTER, '/word/footer2.xml')

doc.settings.odd_and_even_pages_header_footer = True

sec0 = doc.sections[0]
cols0 = sec0._sectPr.find(qn('w:cols'))
if cols0 is None:
    cols0 = OxmlElement('w:cols')
    sec0._sectPr.append(cols0)
cols0.set(qn('w:num'), '1')
clear_refs(sec0._sectPr)
pos = 0
pos = add_ref(sec0._sectPr, 'w:headerReference', 'first', HDR['header1'], pos)
pos = add_ref(sec0._sectPr, 'w:headerReference', 'even', HDR['header2'], pos)
pos = add_ref(sec0._sectPr, 'w:headerReference', 'default', HDR['header3'], pos)
pos = add_ref(sec0._sectPr, 'w:footerReference', 'first', FTR1, pos)
pos = add_ref(sec0._sectPr, 'w:footerReference', 'even', FTR2, pos)
add_ref(sec0._sectPr, 'w:footerReference', 'default', FTR2, pos)
set_footnote_pos(sec0._sectPr)
# page 1 of a section with titlePg uses the type="first" header/footer, which is
# where the template ships the IEEE Sensors Journal logo and the copyright line.
sec0.different_first_page_header_footer = True

para('Sensor-Channel and Front-End Sizing Evidence for Wrist-PPG Heart-Rate '
     'Monitoring: A Cross-Dataset Controlled Study', 'Title')
ap = doc.add_paragraph(style='Authors')
for i, (nm, aff) in enumerate([('Fangwei Xue', '1'), ('Shengran Zhao', '1'),
                               ('Shiguan Xu', '2'), ('Minghui Gao', '3'),
                               ('Peilin Han', '1')]):
    if i:
        ap.add_run(', and ' if i == 4 else ', ')
    ap.add_run(nm)
    sup = ap.add_run(aff)
    sup.font.superscript = True
star = ap.add_run('*')
star.font.superscript = True
para('Abstract—Wrist photoplethysmography (PPG) loses heart rate under motion, and wearable '
     'designs respond by stacking inertial sensors and over-specifying front ends without '
     'evidence. On the public Wrist exercise database (19 records, 8 subjects, 1664 eight-second '
     'windows, chest-ECG reference) and the BIDMC database (53 subjects, 6306 windows), '
     'replicated on PPG-DaLiA (15 subjects, 2825 windows), with a '
     'fixed subspace cleaner and comb estimator, we '
     'quantify four design facts: a single accelerometer is sufficient (34.14 bpm) and every '
     'inertial union is slightly worse (34.52-34.61 bpm), stable in 8/8 leave-one-subject-out '
     'folds and TOST-equivalent within +/-2 bpm; the optical rate can drop to 16 Hz and '
     'quantization to 8 bits with negligible '
     'penalty on both datasets (BIDMC-equivalent within +/-0.5 bpm); an estimator continuity '
     'prior helps neither under exercise nor at '
     'rest (point-estimate MAE +4.91 and +2.83 bpm, subject-level p >= 0.49); and a '
     'transient-aware gate yields no significant gain, '
     'while a harmonic interference index does not predict residual error. A datasheet-based '
     'budget model shows the recommended configuration draws about 80% less average sensor '
     'current. These are sensor '
     'design guidelines, not a claim of clinical accuracy: exercise errors remain large and the '
     'exercise cohort is small.', 'Abstract')
para('Index Terms—heart rate, photoplethysmography, inertial measurement unit, sensor selection, '
     'front-end sizing, cross-dataset validation, low-power design.', 'IndexTerms')

sec1 = doc.add_section(WD_SECTION_START.CONTINUOUS)
cols1 = sec1._sectPr.find(qn('w:cols'))
if cols1 is None:
    cols1 = OxmlElement('w:cols')
    sec1._sectPr.append(cols1)
cols1.set(qn('w:num'), '2')
cols1.set(qn('w:space'), '288')
clear_refs(sec1._sectPr)
pos = 0
pos = add_ref(sec1._sectPr, 'w:headerReference', 'even', HDR['header2'], pos)
pos = add_ref(sec1._sectPr, 'w:headerReference', 'default', HDR['header3'], pos)
pos = add_ref(sec1._sectPr, 'w:footerReference', 'even', FTR2, pos)
add_ref(sec1._sectPr, 'w:footerReference', 'default', FTR2, pos)
set_footnote_pos(sec1._sectPr)
# add_section() copies section 0's properties, titlePg included; only page 1 of
# the document may use the "first" header (the logo) and the copyright footer.
sec1.different_first_page_header_footer = False


def edit_header(name, repl, page_number_text):
    """Template headers carry placeholder text and a literal page number; swap
    in the real running head and a live PAGE field."""
    part = doc.part.rels[HDR[name]].target_part
    for p in part.element.findall(qn('w:p')):
        for r in list(p.findall(qn('w:r'))):
            t = r.find(qn('w:t'))
            if t is None or t.text is None:
                continue
            if t.text.strip() == page_number_text:
                t.text = '1'
                t.set(qn('xml:space'), 'preserve')
                at = list(p).index(r)
                p.remove(r)
                fld = OxmlElement('w:fldSimple')
                fld.set(qn('w:instr'), ' PAGE \\* MERGEFORMAT ')
                fld.append(r)
                p.insert(at, fld)
            elif t.text in repl:
                t.text = repl[t.text]
                t.set(qn('xml:space'), 'preserve')


edit_header('header1', {}, '1')
edit_header('header2', {}, '2')
edit_header('header3', {'First Author ': 'F. Xue ',
                        '.: Title': '.: ' + RUNNING_HEAD_TITLE}, '3')

# ------------------------------------------------------------------- body -----
para('Introduction', 'Heading 1')
para('Optical pulse sensing is the default modality of consumer wearables [1], [2], yet under '
     'movement the perfusion signal is buried by contact-pressure modulation, venous pooling and '
     'ambient leakage [3] (@optics@). The engineering response has been to add inertial channels and feed '
     'them to adaptive cancellers [6], spectral subtraction [7], mode decomposition [8] or source '
     'separation [9], with gyroscope-informed variants capturing rotation-driven contamination '
     '[4]; a parallel instinct is to stack accelerometers, gyroscopes and magnetometers "for '
     'fusion", even though single-accelerometer decomposition has been explored for heart rate '
     '[5]. Both instincts spend sensor budget, energy and board area without evidence that the '
     'extra channels help the optical estimate. Commercial wrist devices likewise lose agreement '
     'with ECG exactly when training-zone heart rate matters [12]-[14].', 'PARA')
para('This paper asks a narrow, sensor-centric question: how many inertial channels, and what '
     'optical front-end resolution, does a wrist node actually need? We answer with a controlled, '
     'cross-dataset measurement study rather than a new cleaner. A deterministic '
     'kinematic-informed subspace reconstruction (KISR) serves as a fixed probe: it builds a '
     'motion subspace from the inertial channels in the short-time Fourier domain and gates '
     'subtraction per bin by PPG-motion coherence, so adding or removing a channel is isolated '
     'from estimator choice (@chain@). Every channel set feeds the same harmonic-comb estimator '
     '[10], [11] under an independent-window protocol, and every window is scored against ECG R '
     'peaks [22]. Contributions: (i) a leave-one-subject-out stable channel-set ablation showing '
     'one accelerometer is sufficient, replicated on an independent exercise cohort '
     '(PPG-DaLiA); (ii) front-end sizing evidence, replicated on a second '
     'database, showing the optical rate can drop to 16 Hz and quantization to 8 bits; (iii) a '
     'cross-activity-state quantification showing that a hard-wired estimator continuity prior '
     'brings no benefit under exercise or at rest; (iv) two honest negatives (transient-aware '
     'gating, harmonic '
     'interference index) that locate residual error in non-harmonic transients; (v) an '
     'equivalence-tested sizing protocol (TOST with pre-specified margins), a '
     'success-conditioned robustness check, and a datasheet-based energy budget. '
     'Cleaning-family '
     'superiority comparisons and hardware implementation are outside the scope of this sizing '
     'study; parity among cleaners under exercise is reported elsewhere [12], [23], [24].',
     'PARA_Indent')

para('Data and Protocol', 'Heading 1')
para('Datasets and Windows', 'Heading 2')
para('Wrist [18], [19]: chest ECG, wrist PPG (256 Hz), wide-range and low-noise tri-axial '
     'accelerometers, gyroscope and magnetometer during cycling (two resistances), walking and '
     'running; all 19 complete records from 8 subjects are used (1664 eight-second windows, '
     'Table I, @signals@ and @stream@). BIDMC [25]: finger PPG (125 Hz) with lead-II ECG from 53 '
     'subjects at rest or in the ICU (6306 windows); no IMU is available, so BIDMC tests the '
     'front-end and estimator laws with a band-pass front end. PPG-DaLiA [30] supplies the '
     'independent exercise replication: wrist PPG (64 Hz) with wrist and chest accelerometers '
     'and chest ECG from 15 subjects during walking, cycling and stairs (2825 windows); its '
     'public release carries no gyroscope or magnetometer. '
     'Windows whose ECG reference falls '
     'outside 40-180 bpm are excluded by rule. Signal-quality gating follows [20], [21].', 'PARA')
mark('tab:I')
para('Cleaner Probe and Estimator', 'Heading 2')
para('@chain@ fixes the budget being sized. KISR transforms each inertial channel with a short-time '
     'Fourier transform (Hann, 1-s frame, '
     '0.25-s hop); per frame, peaks in 0.5-12 Hz and their first three harmonics mark bins, and '
     'the union over channels forms the motion subspace (@subspace@, @mech@). Per-bin gain is gated '
     'by magnitude-'
     'squared coherence with the accelerometer magnitude, scaled by frame motion intensity, and '
     'floored inside the cardiac band for harmonic-consistent peaks; inverse STFT yields the '
     'cleaned waveform. The estimator is a harmonic comb with low-frequency evidence gate and '
     'parabolic refinement, identical for every configuration, and runs without continuity prior '
     'unless explicitly switched on.', 'PARA')
mark('fig:optics')
mark('fig:chain')
para('Statistics', 'Heading 2')
para('Inference is subject-level: Wilcoxon signed-rank on per-subject MAE differences with '
     'bootstrap 95% CIs clustered by subject (4000 resamples). Window-level tests are secondary '
     'because windows within a record are correlated. The channel law is additionally checked by '
     'leave-one-subject-out (LOSO) refitting of the channel ordering.', 'PARA')

para('Results', 'Heading 1')
para('One Accelerometer Is Enough', 'Heading 2')
t2 = []
for key in CH_ORDER:
    v = WE['channel_ablation'][key]
    vs = v.get('vs_full') or {}
    d, pv = vs.get('delta_mae'), vs.get('p_subject_wilcoxon')
    t2.append([CH_LABEL[key], '%.2f' % v['mae'], '--' if d is None else '%+.2f' % d,
               '--' if pv is None else '%.2f' % pv])
para('Table II reports MAE for six inertial channel sets on the full Wrist cohort. The '
     'single-accelerometer sets are best (wide-range %.2f, low-noise %.2f bpm); gyro alone is '
     'worse (%.2f), and every union is slightly worse than the best single accelerometer '
     '(%.2f-%.2f). Each added channel contributes its own spectral peaks, widening the union '
     'mask into bins that carry genuine pulse energy. Differences are <=0.47 bpm and '
     'subject-level paired tests are non-significant (p >= 0.74); the ordering is nevertheless '
     'stable: in 8/8 leave-one-subject-out folds the best single accelerometer beats the full '
     'suite (wide-range best in 7/8). We report this as a point-estimate design guideline with '
     'cross-subject stability, not a significant superiority claim.'
     % (WE['channel_ablation']['ch_acc_wide']['mae'], WE['channel_ablation']['ch_acc_ln']['mae'],
        WE['channel_ablation']['ch_gyro']['mae'],
        WE['channel_ablation']['ch_acc_gyro_mag']['mae'],
        WE['channel_ablation']['ch_gyro']['mae']), 'PARA')
eqch = EQ['channel_equivalence_wrist']
para('Equivalence testing sharpens this: at a pre-specified, design-irrelevant margin of '
     '%.1f bpm, all six channel sets are statistically equivalent to the full suite '
     '(two one-sided t-tests, p <= %.3f; 90%% CIs of the paired per-subject differences inside '
     '+/-%.1f bpm). "One accelerometer is enough" is therefore an equivalence statement, not '
     'merely a non-significant difference.'
     % (EQ['protocol']['margin_wrist_bpm'], max(v['p_tost'] for v in eqch.values()),
        EQ['protocol']['margin_wrist_bpm']), 'PARA_Indent')
da = DA['channel_ablation']
para('The channel law replicates on an independent exercise cohort (@dalia@). On PPG-DaLiA '
     '(15 subjects, 2825 windows) the wrist accelerometer alone gives %.2f bpm versus %.2f for '
     'the wrist+chest union (Wilcoxon p = %.2f; TOST-equivalent within +/-%.1f bpm, 90%% CI '
     '[%.2f, %.2f]), and is better in %d/15 subjects; the chest accelerometer alone is slightly '
     'but significantly worse (%.2f bpm, p = %.2f) yet still inside the equivalence margin. '
     'DaLiA\'s public release lacks gyro/mag, so this replication tests accelerometer unions.'
     % (da['ch_acc']['mae'], da['ch_acc_both']['mae'],
        da['ch_acc']['vs_full']['p_subject_wilcoxon'],
        da['ch_acc']['tost']['margin_bpm'],
        da['ch_acc']['tost']['ci90'][0], da['ch_acc']['tost']['ci90'][1],
        DA['acc_vs_full_subjects']['acc_better'], da['ch_acc_chest']['mae'],
        da['ch_acc_chest']['vs_full']['p_subject_wilcoxon']),
     'PARA_Indent')
mark('tab:II')
mark('fig:signals')
mark('fig:stream')
mark('fig:dalia')

para('Front-End Sizing: Rate to 16 Hz, 8 Bits, on Two Datasets', 'Heading 2')
w_sz, b_sz = WE['sizing'], BI['sizing']
t3 = [['Configuration', 'Wrist (8 subj.)', 'BIDMC (53 subj.)']]
for label, wk, bk in [('native (256 / 125 Hz, 16-bit)', 'native_256Hz_16bit', 'native_125Hz_16bit'),
                      ('64 Hz', 'sz_fs64', 'fs64'), ('32 Hz', 'sz_fs32', 'fs32'),
                      ('16 Hz', 'sz_fs16', 'fs16'), ('8-bit', 'sz_b8', 'b8'),
                      ('10-bit', 'sz_b10', 'b10'), ('12-bit', 'sz_b12', 'b12')]:
    t3.append([label, '%.2f' % w_sz[wk]['mae'], '%.2f' % b_sz[bk]['mae']])
para('Table III and @sizing@ give the sizing evidence. On Wrist, decimating from the native 256 Hz '
     'to 64/32/16 Hz yields %.2f/%.2f/%.2f bpm versus %.2f native, and 8/10/12-bit quantization '
     'yields %.2f/%.2f/%.2f bpm: no rate cliff down to 16 Hz and a worst-case quantization '
     'penalty of +0.04 bpm. On BIDMC the same law replicates: 16 Hz gives %.2f versus %.2f bpm '
     'native (+%.2f, p = %.2f) and 8-bit gives %.2f bpm; with 53 subjects BIDMC has enough power '
     'for the very small 64 Hz and 32 Hz deltas (+%.3f and +%.3f bpm) to reach nominal '
     'significance even though they are design-irrelevant. The design rule is therefore: do not '
     'over-specify the converter or the clock; 16 Hz with 8-10 bits suffices for comb-based '
     'estimation, and the saved energy should go to optical quality [15]-[17].'
     % (w_sz['sz_fs64']['mae'], w_sz['sz_fs32']['mae'], w_sz['sz_fs16']['mae'],
        w_sz['native_256Hz_16bit']['mae'], w_sz['sz_b8']['mae'], w_sz['sz_b10']['mae'],
        w_sz['sz_b12']['mae'], b_sz['fs16']['mae'], b_sz['native_125Hz_16bit']['mae'],
        b_sz['fs16']['vs_native']['delta_mae'], b_sz['fs16']['vs_native']['p_subject_wilcoxon'],
        b_sz['b8']['mae'], b_sz['fs64']['vs_native']['delta_mae'],
        b_sz['fs32']['vs_native']['delta_mae']), 'PARA')
eqw, eqb = EQ['sizing_equivalence_wrist'], EQ['sizing_equivalence_bidmc']
rw = EQ['success_conditioned_robustness']['wrist']
para('At a %.1f bpm margin every BIDMC configuration is equivalent to native (TOST '
     'p <= %.4f); on Wrist the 64/32 Hz and 8/10/12-bit configurations are equivalent at '
     '%.1f bpm, while 16 Hz is not yet established there (p = %.3f, n = 8). Conditioning on '
     'the %d Wrist windows where the baseline succeeds (|error| <= %.0f bpm, MAE %.2f bpm) '
     'leaves the success fraction unchanged (%.1f%% vs %.1f%%) and holds 16 Hz and 8-bit to '
     '%.2f and %.2f bpm: the sizing law is not an artifact of the failing baseline, and even '
     'where the estimator works the penalty stays below 5 bpm.'
     % (EQ['protocol']['margin_bidmc_bpm'], max(v['p_tost'] for v in eqb.values()),
        EQ['protocol']['margin_wrist_bpm'], eqw['sz_fs16']['p_tost'],
        rw['n_windows_success'], EQ['protocol']['success_threshold_bpm'],
        rw['configs']['ch_acc_gyro']['mae_on_success_subset'],
        100 * rw['configs']['ch_acc_gyro']['success_fraction'],
        100 * rw['configs']['sz_fs16']['success_fraction'],
        rw['configs']['sz_fs16']['mae_on_success_subset'],
        rw['configs']['sz_b8']['mae_on_success_subset']), 'PARA_Indent')
mark('tab:III')
mark('fig:subspace')
mark('fig:mech')
mark('fig:sizing')

para('Continuity Priors Help Neither Under Exercise nor at Rest', 'Heading 2')
w_ct, b_ct = WE['continuity'], BI['continuity']
para('Switching the estimator continuity prior on moves Wrist MAE from %.2f to %.2f bpm '
     '(%+.2f bpm, 95%% CI %+.2f to %+.2f, subject-level p = %.2f, better in %d of %d subjects): '
     'the mechanism we see in the traces is that the prior locks onto a wrong spectral peak '
     'after a single mis-selection during rapid heart-rate transitions and then propagates the '
     'error. On resting BIDMC the point estimate is again worse (%.2f vs %.2f bpm, %+.2f, p = '
     '%.2f, better in %d of %d), so the hypothesized resting benefit does not materialize and '
     'neither state supports an activity-gated prior with these data. Because both contrasts are '
     'statistically unresolved, the defensible statement is asymmetric: a hard-wired continuity '
     'prior buys nothing measurable here while carrying a large exercise penalty, so it should '
     'be disabled or its gate learned rather than assumed free.'
     % (w_ct['prior_off']['mae'], w_ct['prior_on']['mae'], w_ct['on_minus_off']['delta_mae'],
        w_ct['on_minus_off']['ci95_subject_cluster'][0], w_ct['on_minus_off']['ci95_subject_cluster'][1],
        w_ct['on_minus_off']['p_subject_wilcoxon'], w_ct['subjects_prior_better'],
        w_ct['n_subjects_paired'], b_ct['prior_on']['mae'], b_ct['prior_off']['mae'],
        b_ct['on_minus_off']['delta_mae'], b_ct['on_minus_off']['p_subject_wilcoxon'],
        b_ct['subjects_prior_better'], b_ct['n_subjects_paired']), 'PARA')

para('Error Lives in Activity and Non-Harmonic Transients', 'Heading 2')
para('Per-record MAE clusters by activity rather than subject (@record@): walking is easy for '
     'every configuration while high-resistance cycling and running are hard for all. A harmonic '
     'interference index (fraction of cardiac-band bins covered by the motion subspace) does not '
     'predict residual error (Spearman rho = -0.005, p = 0.841), locating failure in non-'
     'harmonic transients. Motivated by this, we added a transient-aware gate (IMU jerk plus PPG '
     'kurtosis/spectral-flatness flagging, %.1f%% of frames): on the %d hard-activity records '
     '(%d windows, %d subjects) the conservative band-pass variant changes MAE by %+.2f bpm '
     '(p = %.3f) and the soft-gain variant by %+.2f bpm (p = %.3f) versus KISR (%.2f bpm) - no '
     'significant gain (@gate@). Simple transient gating is therefore insufficient; '
     'transient-aware models remain open.'
     % (100 * TG['flagged_frame_fraction']['gate_soft']['mean'], TG['cohort']['n_records'],
        TG['cohort']['n_windows'], TG['cohort']['n_subjects'],
        TG['overall']['gate_bp']['vs_kisr']['delta_mae'],
        TG['overall']['gate_bp']['vs_kisr']['p_subject_wilcoxon'],
        TG['overall']['gate_soft']['vs_kisr']['delta_mae'],
        TG['overall']['gate_soft']['vs_kisr']['p_subject_wilcoxon'],
        TG['overall']['kisr']['mae']), 'PARA')
mark('fig:record')
mark('fig:gate')
SP_PATH = os.path.join(NOV, 'strong_estimator_probes.json')
if os.path.exists(SP_PATH):
    SP = json.load(open(SP_PATH))
    para('Estimator Probes: the Residual Is Not a Tracking Artifact', 'Heading 2')
    para('Spectral inspection of hard windows shows the stride fundamental and its harmonics '
         '(60/120/180 bpm) often exceed the cardiac peak inside 0.7-3.3 Hz, and coherence-gated '
         'projection leaves them when PPG-IMU coherence is low. Three stronger estimators '
         'probed on all %d windows - 4-s zero-padded spectra with within-window Viterbi, '
         'multi-reference RLS cancellation, and a harmonic-evidence gate on every candidate - '
         'none improves on the submitted probe (Table IV). The residual is therefore a property '
         'of the motion-contaminated channel, not of resolution, tracking or cancellation; '
         'this bounds cleaner-family claims and motivates transient-aware models.'
         % SP['n_windows'], 'PARA')
    mark('tab:IV')

para('Discussion and Limitations', 'Heading 1')
para('Three design laws follow: fewer, better-chosen inertial channels; a modest optical front '
     'end (16 Hz, 8-10 bits) replicated across two databases; and no measurable benefit from a '
     'hard-wired continuity '
     'prior. Together they redirect budget from inertial stacking, converter bits and clock rate '
     'toward optical quality and transient-aware artifact models.', 'PARA')
en = EN['rows']
para('The budget argument can be quantified without a prototype. A datasheet-based analytical '
     'model (BMI270 accelerometer/gyroscope, BMM150 magnetometer, MAX86176 optical AFE, '
     'nRF52840 SoC and BLE; linear ODR scaling; Table V) puts the full suite at 256 Hz / '
     '16-bit at %.2f mA average supply current, against %.2f mA for one accelerometer at '
     '16 Hz / 8-bit - a %.0f%% reduction - with the optical channel itself untouched '
     '[26]-[29]. The saved current is precisely the budget that should buy optical quality '
     '(LED drive, aperture, gasketing) instead of inertial redundancy. These are vendor-'
     'typical estimates, not measured power.'
     % (en[0]['i_total_ma'], en[-1]['i_total_ma'], en[-1]['saving_vs_full_pct']),
     'PARA_Indent')
mark('tab:V')
para('Limitations are candid. Exercise absolute errors (34-43 bpm) are far from clinical use, '
     'and three stronger estimator probes did not reduce them; '
     'this is a diagnostic and sizing study, not a validated monitor, and no ISO 81060-2 or '
     'ANSI/AAMI EC13 agreement is claimed. The exercise evidence spans two databases (Wrist, '
     '8 subjects; PPG-DaLiA, 15), but the public DaLiA release lacks gyroscope and '
     'magnetometer, so the gyro/mag part of the channel law rests on Wrist alone; BIDMC adds '
     '53 subjects but is resting finger PPG without '
     'IMU, so the channel law cannot be tested there and modality/setting differences limit '
     'direct transfer. No hardware prototype, on-wrist validation or measured power data is part '
     'of this study; front-end recommendations derive from decimation/quantization experiments '
     'on public recordings, not from a fabricated node. The 16 Hz equivalence on Wrist is '
     'underpowered (8 subjects), and the energy budget is a datasheet estimate.', 'PARA_Indent')

para('Conclusion', 'Heading 1')
para('Across two public databases, one clean accelerometer matches or beats every inertial union '
     '(LOSO-stable), the optical front end may drop to 16 Hz and 8 bits, a hard-wired continuity '
     'prior buys nothing under exercise or at rest, and neither harmonic interference nor simple '
     'transient gating explains or removes the residual. These measured, partly negative '
     'statements give wrist-node designers a sensor budget law rather than another cleaning '
     'leaderboard.', 'PARA')
para('Declarations', 'Heading 1')
para('Author Contributions', 'Heading 2')
para('Conceptualization, F. Xue and P. Han; methodology, F. Xue; software, F. Xue and S. Zhao; '
     'validation, S. Zhao and M. Gao; formal analysis, F. Xue; investigation, F. Xue, S. Zhao, '
     'S. Xu, and M. Gao; data curation, S. Xu; '
     'visualization, S. Xu; writing - original draft, F. Xue; writing - review and editing, all '
     'authors; supervision, P. Han; project administration, P. Han. All authors have read and '
     'agreed to the final version of the manuscript.', 'PARA')
para('Conflict of Interest', 'Heading 2')
para('The authors declare that the research was conducted in the absence of any commercial or '
     'financial relationships that could be construed as a potential conflict of interest.',
     'PARA')
para('Data and Code Availability', 'Heading 2')
para('The Wrist [19] and BIDMC [25] databases analyzed in this study are public on PhysioNet. '
     'The analysis code and experiment outputs are archived at '
     'https://doi.org/10.5281/zenodo.23066297 and mirrored on GitHub at '
     'https://github.com/XFW-666/kisr-wrist-ppg. Supplementary material comprises subspace '
     'intermediate maps, component sweeps, cleaning-family comparison, transient-gate variants '
     'and per-subject tables.', 'PARA')
para('Funding', 'Heading 2')
para('This work received no external funding.', 'PARA')


# ---------------------------------------------------------------- tables ------
def _set(el, **kw):
    for k, v in kw.items():
        el.set(qn('w:' + k), str(v))


def table_caption(text):
    p = para(text, 'Table Title')
    p.paragraph_format.keepNext = True
    p.paragraph_format.spaceBefore = Pt(4)
    p.paragraph_format.spaceAfter = Pt(2)
    return p


def ieee_table(rows, fracs, align, size=8):
    """One-column IEEE table: three rules, no vertical lines, centred."""
    ncols = len(fracs)
    tb = doc.add_table(rows=len(rows), cols=ncols)
    tb.autofit = False
    tblPr = tb._tbl.tblPr
    for tag in ('w:tblStyle', 'w:tblW', 'w:tblLayout', 'w:tblBorders', 'w:jc', 'w:tblCellMar',
                'w:tblInd'):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    tblW = OxmlElement('w:tblW')
    _set(tblW, w=COL_TW, type='dxa')
    tblPr.append(tblW)
    jc = OxmlElement('w:jc')
    _set(jc, val='center')
    tblPr.append(jc)
    borders = OxmlElement('w:tblBorders')
    for side, val in (('top', 'single'), ('bottom', 'single'), ('left', 'none'),
                      ('right', 'none'), ('insideH', 'none'), ('insideV', 'none')):
        b = OxmlElement('w:' + side)
        _set(b, val=val, sz=8 if val == 'single' else 0, space=0, color='auto')
        borders.append(b)
    tblPr.append(borders)
    lay = OxmlElement('w:tblLayout')
    _set(lay, type='fixed')
    tblPr.append(lay)
    mar = OxmlElement('w:tblCellMar')
    for side, w in (('top', 0), ('left', 36), ('bottom', 0), ('right', 36)):
        m = OxmlElement('w:' + side)
        _set(m, w=w, type='dxa')
        mar.append(m)
    tblPr.append(mar)
    grid = tb._tbl.find(qn('w:tblGrid'))
    for gc, frac in zip(grid.findall(qn('w:gridCol')), fracs):
        _set(gc, w=int(COL_TW * frac))
    aligns = align if isinstance(align, list) else [align] * ncols
    for i, row in enumerate(rows):
        trPr = tb.rows[i]._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        if i == 0:
            trPr.append(OxmlElement('w:tblHeader'))
        for j, val in enumerate(row):
            cell = tb.rows[i].cells[j]
            p = cell.paragraphs[0]
            p.style = doc.styles['Normal']
            p.alignment = aligns[j]
            pf = p.paragraph_format
            pf.spaceBefore = Pt(1)
            pf.spaceAfter = Pt(1)
            pf.lineSpacing = 1.0
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            r.font.name = 'Times New Roman'
            if i == 0:
                r.bold = True
                tcB = OxmlElement('w:tcBorders')
                bot = OxmlElement('w:bottom')
                _set(bot, val='single', sz=6, space=0, color='auto')
                tcB.append(bot)
                cell._tc.get_or_add_tcPr().append(tcB)
    tail = doc.add_paragraph(style='Normal')
    tail.paragraph_format.spaceAfter = Pt(2)
    return tb, tail


recs = WE['cohort']['records']
n_bike = sum(1 for r in recs if 'bike' in r)
n_bike_hi = sum(1 for r in recs if 'high_resistance_bike' in r)
n_run = sum(1 for r in recs if 'run' in r)
n_walk = sum(1 for r in recs if 'walk' in r)
cap = table_caption('TABLE I. EVALUATION SET (WRIST: ALL COMPLETE RECORDS)')
tb, tail = ieee_table([['Item', 'Value'],
            ['Records (complete)', '%d (subjects s1-s6, s8, s9)' % len(recs)],
            ['Subjects', '%d' % WE['cohort']['n_subjects']],
            ['Activities', 'bike %d (%d high-, %d low-resistance), run %d, walk %d'
             % (n_bike, n_bike_hi, n_bike - n_bike_hi, n_run, n_walk)],
            ['Window / step', '8 s / 4 s (%d windows)' % WE['cohort']['n_windows']],
            ['Native rate', '256 Hz (PPG, acc, gyro, mag, ECG)'],
            ['Channels', 'PPG, wide-range acc, low-noise acc, gyro, mag, chest ECG'],
            ['Reference HR', 'ECG R-peaks with median-RR fallback'],
            ['Exclusion rule', 'reference outside 40-180 bpm']],
           [0.30, 0.70], WD_ALIGN_PARAGRAPH.LEFT)
GROUPS['tab:I'] = [cap._p, tb._tbl, tail._p]

cap = table_caption('TABLE II. MAE BY INERTIAL CHANNEL SET (WRIST, %d WINDOWS, %d SUBJECTS)'
              % (WE['cohort']['n_windows'], WE['cohort']['n_subjects']))
tb, tail = ieee_table([['Channel set', 'MAE', '\u0394 full', 'p']] + t2,
           [0.46, 0.16, 0.19, 0.19],
           [WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 3)
GROUPS['tab:II'] = [cap._p, tb._tbl, tail._p]

cap = table_caption('TABLE III. FRONT-END SIZING ON TWO DATABASES (MAE, BPM)')
tb, tail = ieee_table(t3, [0.44, 0.28, 0.28],
           [WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 2)
GROUPS['tab:III'] = [cap._p, tb._tbl, tail._p]

cap = table_caption('TABLE V. ESTIMATED AVERAGE SUPPLY CURRENT (DATASHEET MODEL)')
tb, tail = ieee_table(
    [['Configuration', 'mA', 'Save %']] +
    [[r['config'], '%.2f' % r['i_total_ma'], '%.0f' % r['saving_vs_full_pct']]
     for r in EN['rows']],
    [0.56, 0.20, 0.24],
    [WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 2)
GROUPS['tab:V'] = [cap._p, tb._tbl, tail._p]

if os.path.exists(SP_PATH):
    cap = table_caption('TABLE IV. ESTIMATOR PROBES ON ALL WRIST WINDOWS (MAE, BPM)')
    tb, tail = ieee_table(
        [['Estimator probe', 'MAE'],
         ['KISR + comb (submitted)', '%.2f' % SP['mae']['kisr_comb_submitted']],
         ['4-s zero-pad + Viterbi', '%.2f' % SP['mae']['hires_viterbi']],
         ['multi-reference RLS ANC', '%.2f' % SP['mae']['rls_anc']],
         ['strict harmonic-evidence gate', '%.2f' % SP['mae']['strict_harmonic_gate']]],
        [0.72, 0.28], [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])
    GROUPS['tab:IV'] = [cap._p, tb._tbl, tail._p]

# --------------------------------------------------------------- figures ------
FIG_FILES = {
    'chain': ('figA_sensor_chain.png', os.path.join(ROOT, 'figures', 'g1_methods.png')),
    'optics': ('figC_optical_front_end.png', os.path.join(ROOT, 'figures', 'fig3_mechanism.png')),
    'signals': ('fig1_col.png', os.path.join(WORK, 'fig1_full.png')),
    'subspace': ('figB_subspace.png', os.path.join(ROOT, 'figures', 'g2_subspace.png')),
    'mech': ('fig2_col.png', os.path.join(WORK, 'fig1_full.png')),
    'stream': ('fig3_col.png', os.path.join(WORK, 'fig2.png')),
    'dalia': ('figD_dalia.png', None),
    'record': ('fig4_col.png', os.path.join(WORK, 'fig3.png')),
    'sizing': ('fig5_col.png', os.path.join(WORK, 'fig4.png')),
    'gate': ('fig6_col.png', os.path.join(WORK, 'fig5.png')),
}

FIG_CAPTION = {
    'chain': 'Sensing arrangement probed by this study: wrist node with the reflective optical '
             'PPG channel, two tri-axial accelerometers, gyroscope and magnetometer on one sample '
             'clock, and the KISR cleaning plus harmonic-comb estimation chain. It defines the '
             'sensor budget the paper sizes; it is not a claimed validated prototype.',
    'optics': 'Optical front end and the artifact pathways it is exposed to: reflective PPG '
              'geometry (emitter, detector, skin and microvasculature), and the four mechanisms '
              'that corrupt the pulse under motion - contact-pressure modulation of the DC '
              'operating point, venous pooling, ambient leakage and the motion spectrum that '
              'overlaps the cardiac band.',
    'signals': 'Real signals and spectra for cycling (high resistance), walking and running: '
               'chest-ECG reference, wrist PPG, tri-axial accelerometer, and PPG spectra at three '
               'motion intensities (PhysioNet Wrist recordings; no synthetic traces).',
    'subspace': 'Motion subspace built from the inertial channels on a real high-intensity '
                'cycling window: per-channel STFT support, the union mask of peaks and harmonics, '
                'and the projection that removes the motion subspace from the PPG.',
    'mech': 'Artifact mechanism on one hard cycling window: mean STFT power before and after '
            'cleaning with motion harmonics marked, the applied per-bin gain, and PPG-IMU '
            'magnitude-squared coherence per activity.',
    'stream': 'Simultaneous multisensor stream on a real 12 s walking segment: chest ECG '
              'reference, wrist PPG, wide-range and low-noise accelerometer, gyroscope and '
              'magnetometer.',
    'dalia': 'Independent exercise replication on PPG-DaLiA (15 subjects, 2825 eight-second '
             'windows; walking, cycling, stairs): per-activity MAE by channel set (a) and the '
             'paired difference versus the wrist+chest accelerometer union with the '
             '+/-2 bpm TOST band (b). The public release carries no gyroscope or magnetometer, '
             'so the inertial union here is two accelerometers.',
    'record': 'Per-record MAE over all 19 Wrist records: clustering by activity rather than '
              'subject; walking is easy for every channel set while high-resistance cycling and '
              'running are hard for all.',
    'sizing': 'Cross-dataset front-end sizing: MAE versus sampling rate and ADC resolution on '
              'Wrist (exercise, wrist PPG) and BIDMC (rest, finger PPG); no rate cliff to 16 Hz '
              'and negligible 8-bit penalty on both.',
    'gate': 'Transient-aware gating on hard activities: KISR versus soft-gain and conservative '
            'band-pass variants on the eight high-resistance cycling and running records; no '
            'significant gain (subject-clustered 95% CIs).',
}

for i, key in enumerate(FIG_ORDER, start=1):
    fname, fallback = FIG_FILES[key]
    col_path = os.path.join(COL, fname)
    src = col_path if os.path.exists(col_path) else (fallback or col_path)
    if fallback and src == fallback:
        used_fallback.append(key)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keepNext = True
    p.paragraph_format.spaceBefore = Pt(4)
    p.paragraph_format.spaceAfter = Pt(2)
    p.add_run().add_picture(src, width=COL_W)
    cap_p = para('Fig. %d.  %s' % (i, FIG_CAPTION[key]), 'Figure Caption')
    GROUPS['fig:' + key] = [p._p, cap_p._p]

# place every float at its first citation instead of clustering at the end
for key, els in GROUPS.items():
    m = MARKS[key]
    for el in els:
        m.addprevious(el)
    m.getparent().remove(m)

# ------------------------------------------------------------- references -----
para('References', 'Reference Head')
refs = [
 'J. Allen, "Photoplethysmography and its application in clinical physiological measurement," Physiol. Meas., vol. 28, no. 3, pp. R1-R39, 2007.',
 'P. Charlton and P. A. Kyriacou, "Wearable photoplethysmography: current status and future challenges," in Proc. Comput. Cardiol. Conf., 2023.',
 'M. Pflugradt and R. Orglmeister, "Improved signal quality indication for photoplethysmographic signals incorporating motion artifact detection," in Proc. IEEE EMBC, 2014, pp. 1872-1875.',
 'H. Lee, H. Chung, and J. Lee, "Motion artifact cancellation in wearable photoplethysmography using gyroscope," IEEE Sens. J., vol. 19, no. 3, pp. 1166-1175, 2019.',
 'C. Zhao, W. Zeng, D. Hu, and H. Liu, "Robust heart rate monitoring by a single wrist-worn accelerometer based on signal decomposition," IEEE Sens. J., vol. 21, no. 14, pp. 15962-15971, 2021.',
 'S. Kunchon, T. Desudchit, and C. Chinrungrueng, "Comparative and evaluative the adaptive filter in motion artifact cancellation of pulse oximetry," in Proc. IEEE NEBC, 2009, pp. 1-2.',
 'Y.-H. Yang and K.-T. Tang, "A pulse oximetry system with motion artifact reduction based on Fourier analysis," in Proc. IEEE ISBB, 2014, pp. 1-4.',
 'W. He et al., "Variational mode decomposition-based heart rate estimation using wrist-type photoplethysmography during physical exercise," in Proc. ICPR, 2018, pp. 3766-3771.',
 'B. Mijovic, M. De Vos, I. Gligorijevic, J. Taelman, and S. Van Huffel, "Source separation from single-channel recordings by combining empirical-mode decomposition and independent component analysis," IEEE Trans. Biomed. Eng., vol. 57, no. 9, pp. 2188-2196, 2010.',
 'B. Sun, H. Feng, and Z. Zhang, "A new approach for heart rate monitoring using photoplethysmography signals contaminated by motion artifacts," in Proc. IEEE ICASSP, 2016, pp. 1-5.',
 'H. Dubey, R. Kumaresan, and K. Mankodiya, "Harmonic sum-based method for heart rate estimation using PPG signals affected with motion artifacts," J. Ambient Intell. Humaniz. Comput., vol. 9, no. 1, pp. 137-150, 2016.',
 'D. Jarchi and A. J. Casson, "Towards photoplethysmography-based estimation of instantaneous heart rate during physical activity," IEEE Trans. Biomed. Eng., vol. 64, no. 9, pp. 2042-2053, 2017.',
 'J. F. Horton, P. Stergiou, T. S. Fung, and L. Katz, "Comparison of Polar M600 optical heart rate and ECG heart rate during exercise," Med. Sci. Sports Exerc., vol. 49, no. 12, pp. 2600-2607, 2017.',
 'Y. Jeong, S. Yoon, J. Lee, W. M. Jeong, and M. S. Ha, "Comparison of heart rate measurement accuracy among commercially available photoplethysmography-based wearable devices across exercise intensities," Phys. Act. Nutr., vol. 30, no. 2, pp. 142-150, 2026.',
 'E. D. Giovanni, S. Murali, F. Rincon, and D. Atienza, "Ultra-low power estimation of heart rate under physical activity using a wearable photoplethysmographic system," in Proc. Euromicro DSD, 2016, pp. 553-560.',
 'C. Zheng, C. Yong, Q. Wei, and F. Qiao, "Flexible wearable heart rate monitoring system and low-power design: a review," Sensors, vol. 25, no. 16, p. 4913, 2025.',
 'J. Y. Kim and C. H. Chu, "Analysis of energy consumption for wearable ECG devices," in Proc. IEEE SENSORS, 2014, pp. 962-965.',
 'G. Moody, R. Mark, and A. Goldberger, "PhysioNet: a Web-based resource for the study of physiologic signals," IEEE Eng. Med. Biol. Mag., vol. 20, no. 3, pp. 70-75, 2001.',
 'PhysioNet, "Wrist: wrist-worn PPG, accelerometer, gyroscope, magnetometer and ECG during exercise (v1.0.0)," 2019. [Online]. Available: https://physionet.org/content/wrist/1.0.0/',
 'M. Elgendi, "Optimal signal quality index for photoplethysmogram signals," Bioengineering, vol. 3, no. 4, p. 21, 2016.',
 'P. Vaz, J. Henriques, P. de Carvalho, and R. Couceiro, "An automatic method for motion artifacts detection in photoplethysmographic signals referenced with electrocardiography data," in Proc. BMEI, 2014, pp. 704-708.',
 'S. Zhang et al., "Design and implementation of a novel real time P-QRS-T waves detection algorithm," in Proc. IEEE ITNEC, 2020, pp. 1-5.',
 'K. Arunkumar and M. Bhaskar, "Heart rate estimation from wrist-type photoplethysmography signals during physical exercise," Biomed. Signal Process. Control, vol. 57, p. 101790, 2020.',
 'J. Song, D. Li, X. Ma, G. Teng, and J. Wei, "A robust dynamic heart-rate detection algorithm framework during intense physical activities using photoplethysmographic signals," Sensors, vol. 17, no. 11, p. 2450, 2017.',
 'PhysioNet, "BIDMC PPG and respiration dataset (v1.0.0)," 2017. [Online]. Available: https://physionet.org/physiobank/database/bidmc/',
 'Analog Devices, "MAX86176: ultra-low power pulse oximeter and heart-rate monitor AFE," datasheet, 2023. [Online]. Available: https://www.analog.com/en/products/max86176.html',
 'Bosch Sensortec, "BMI270: 6-axis inertial measurement unit," datasheet BST-BMI270-DS000, 2023. [Online]. Available: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf',
 'Bosch Sensortec, "BMM150: 3-axis geomagnetic sensor," datasheet BST-BMM150-DS001, 2020. [Online]. Available: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmm150-ds001.pdf',
 'Nordic Semiconductor, "nRF52840 product specification," v1.7, 2022. [Online]. Available: https://www.nordicsemi.com/Products/nRF52840',
 'A. Reiss, I. Strümke, G. Hendrikx, and P. Lukowicz, "Deep PPG: large-scale heart rate estimation with convolutional neural networks," Sensors, vol. 19, no. 14, p. 3079, 2019. [Online]. Available: https://archive.ics.uci.edu/dataset/495/ppg+dalia',
]
for r in refs:
    para(r, 'References')

REFMAP = renumber_refs(doc)

dst = os.path.join(ROOT, 'KISR_SensorsJ_8p.docx')
try:
    doc.save(dst)
except PermissionError:
    # the previous build is still open in Word; never overwrite a locked file
    dst = os.path.join(ROOT, 'KISR_SensorsJ_8p_v2.docx')
    doc.save(dst)

# -------------------------------------------- author footnote from the original
zo = zipfile.ZipFile(ORIG)
fn_xml = zo.read('word/footnotes.xml')
# drop the removed authors' affiliation lines from the inherited footnote
fn_xml = re.sub(rb'<w:p\b[^>]*>(?:(?!</w:p>).)*?Y\. Chang is with(?:(?!</w:p>).)*?</w:p>',
                b'', fn_xml, flags=re.S)
fn_xml = fn_xml.replace(b'F. Xue, S. Zhao, J. Liu, and P. Han',
                        b'F. Xue, S. Zhao, and P. Han')
# number the affiliation paragraphs to match the author-line superscripts
_chunks = fn_xml.split(b'</w:p>')
for _i, _ch in enumerate(_chunks):
    for _needle, _mk in [(b'F. Xue, S. Zhao, and P. Han', b'1'),
                         (b'S. Xu is with', b'2'), (b'M. Gao is with', b'3')]:
        if _needle in _ch:
            _j = _ch.find(b'</w:pPr>')
            _j = _j + len(b'</w:pPr>') if _j >= 0 else _ch.find(b'>', _ch.find(b'<w:p')) + 1
            _ch = (_ch[:_j] + b'<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr>'
                   b'<w:t>' + _mk + b'</w:t></w:r>' + _ch[_j:])
    _chunks[_i] = _ch
fn_xml = b'</w:p>'.join(_chunks)
fn_xml = fn_xml.replace(b'(Corresponding author: Peilin Han.)',
                        b'*Corresponding author: Peilin Han.')
m = re.search(rb'<w:footnote [^>]*w:id="(\d+)"[^>]*>(?:(?!</w:footnote>).)*?Corresponding author',
              fn_xml, re.S)
fid = m.group(1).decode() if m else '1'
zin = zipfile.ZipFile(dst)
items = {n: zin.read(n) for n in zin.namelist()}
zin.close()
items['word/footnotes.xml'] = fn_xml
rels = items['word/_rels/document.xml.rels'].decode('utf-8')
if 'footnotes.xml' not in rels:
    rels = rels.replace('</Relationships>',
        '<Relationship Id="rIdFn1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/footnotes" Target="footnotes.xml"/></Relationships>')
    items['word/_rels/document.xml.rels'] = rels.encode('utf-8')
ct = items['[Content_Types].xml'].decode('utf-8')
if 'footnotes+xml' not in ct:
    ct = ct.replace('</Types>',
        '<Override PartName="/word/footnotes.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/></Types>')
    items['[Content_Types].xml'] = ct.encode('utf-8')
dx = items['word/document.xml'].decode('utf-8')
idx = dx.find('Fangwei Xue')
endp = dx.find('</w:p>', idx)
if idx < 0 or endp < 0:
    raise SystemExit('footnote anchor not found')
run = ('<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr>'
       '<w:footnoteReference w:customMarkFollows="1" w:id="%s"/>'
       '<w:sym w:font="Symbol" w:char="F020"/></w:r>' % fid)
dx = dx[:endp] + run + dx[endp:]
items['word/document.xml'] = dx.encode('utf-8')


def rels_owner(rels_name):
    """word/_rels/document.xml.rels applies to the part word/document.xml."""
    d, f = os.path.split(rels_name)
    d = d.replace('_rels', '').strip('/').rstrip('\\')
    return (d + '/' if d else '') + f[:-len('.rels')]


def target_of(owner, tgt):
    return os.path.normpath(os.path.join(os.path.dirname(owner), tgt)).replace('\\', '/')


# The template ships its own demo graphics (author photos, an equation object);
# the cleared body no longer uses them, so drop the orphaned parts.
media_parts = [n for n in items if re.match(r'word/(media|embeddings)/', n)]
rels_names = [n for n in items if n.endswith('.rels')]
dropped = []
for part in media_parts:
    used = False
    for rels_name in rels_names:
        owner = rels_owner(rels_name)
        owner_xml = items.get(owner)
        if owner_xml is None:
            continue
        owner_text = owner_xml.decode('utf-8', 'ignore')
        rels_text = items[rels_name].decode('utf-8', 'ignore')
        for rid, tgt in re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels_text):
            if target_of(owner, tgt) == part and rid in owner_text:
                used = True
                break
        if used:
            break
    if not used:
        dropped.append(part)
        del items[part]

for rels_name in [n for n in items if n.endswith('.rels')]:
    owner = rels_owner(rels_name)

    def strip(m):
        tag = m.group(0)
        tgt = re.search(r'Target="([^"]+)"', tag).group(1)
        return '' if target_of(owner, tgt) in dropped else tag

    items[rels_name] = re.sub(r'<Relationship [^>]*/>', strip,
                              items[rels_name].decode('utf-8')).encode('utf-8')

with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
    for n, b in items.items():
        zout.writestr(n, b)

# ------------------------------------------------------------------ report ----
d2 = docx.Document(dst)
words = sum(len(p.text.split()) for p in d2.paragraphs)
hs = [round(sh.height.inches * 25.4) for sh in d2.inline_shapes]
style_use = {}
for p in d2.paragraphs:
    style_use[p.style.name] = style_use.get(p.style.name, 0) + 1
tw_total = sum(float(t._tbl.tblPr.find(qn('w:tblW')).get(qn('w:w'))) for t in d2.tables) / 1440
print('saved', dst)
print('dropped orphan template media:', dropped or 'none')
print('paras', len(d2.paragraphs), 'tables', len(d2.tables), 'images', len(d2.inline_shapes),
      'words', words)
print('table widths total in = %.2f over %d tables' % (tw_total, len(d2.tables)))
print('fig heights mm', hs, 'sum', sum(hs))
print('styles', style_use)
print('column-native figures:', sorted(set(FIG_FILES) - set(used_fallback)))
if used_fallback:
    print('FALLBACK still wide:', used_fallback)
print('est pages ~', round(words / 850 + sum(hs) / 240 + 0.45 + 0.5 + 0.3, 1))
