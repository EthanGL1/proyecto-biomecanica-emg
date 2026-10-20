"""Render a self-contained, local EMG quality and feature explorer."""

from __future__ import annotations

import json
import math
import base64
import gzip
from typing import Any

import numpy as np


TRACE_FIELDS = (
    "series", "run_index", "sample_index", "time_s", "raw_adc", "control_adc",
    "hp20_adc", "hp40_adc", "bp20_400_adc", "fir_bp20_400_adc", "common_valid",
)
FEATURE_FIELDS = (
    "series", "run_index", "window_index", "time_s", "filter_branch",
    "rms_adc", "mdf_hz", "mnf_hz",
)
STFT_FIELDS = (
    "series", "run_index", "window_index", "time_s", "filter_branch",
    "frequency_hz", "psd_adc2_per_hz",
)
FFT_FIELDS = (
    "series", "run_index", "window_index", "time_s", "filter_branch",
    "frequency_hz", "amplitude_adc",
)
PSD_FIELDS = (
    "series", "run_index", "filter_branch", "frequency_hz",
    "psd_adc2_per_hz",
)
SEGMENT_FIELDS = ("series", "run_index", "start_time_s", "end_time_s", "status")
IDENTITY_FIELDS = {"participant_id", "subject_id", "participant", "subject"}


def _otsu_threshold(values: list[float]) -> float | None:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 4 or float(np.max(data) - np.min(data)) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(data)))):
        return None
    bins = min(128, max(8, int(math.sqrt(data.size))))
    counts, edges = np.histogram(data, bins=bins)
    weights = counts.astype(float) / float(data.size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative_weight = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * centers)
    total_mean = float(cumulative_mean[-1])
    denominator = cumulative_weight * (1.0 - cumulative_weight)
    between = np.zeros_like(denominator)
    eligible = denominator > 0
    between[eligible] = (
        (total_mean * cumulative_weight[eligible] - cumulative_mean[eligible]) ** 2
        / denominator[eligible]
    )
    if not eligible.any():
        return None
    index = int(np.argmax(between))
    threshold = float(edges[index + 1])
    if not (float(np.min(data)) < threshold < float(np.max(data))):
        return None
    return threshold


def derive_candidate_activity(
    trace_rows: list[dict[str, Any]],
    *,
    branch: str = "bp20_400",
    frame_seconds: float = 0.05,
    overlap: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create exploratory RMS/Otsu activity candidates without bridging gaps.

    Labels are signal-amplitude candidates only. They do not identify verified
    contractions, mechanical repetitions, rest, artifacts, or fatigue.
    """
    branch_field = f"{branch}_adc"
    if branch not in {"control", "hp20", "hp40", "bp20_400", "fir_bp20_400"}:
        raise ValueError("unsupported candidate-activity filter branch")
    if not math.isfinite(frame_seconds) or frame_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError("candidate frame duration and overlap are invalid")
    projected = _project_rows(trace_rows, TRACE_FIELDS, table_name="trace rows")
    by_series_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in projected:
        by_series_run.setdefault((str(row["series"]), int(row["run_index"])), []).append(row)

    raw_frames: dict[str, list[dict[str, Any]]] = {}
    for (series, run_index), rows in sorted(by_series_run.items()):
        rows.sort(key=lambda row: int(row["sample_index"]))
        times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
        if times.size < 2:
            continue
        intervals = np.diff(times)
        positive = intervals[np.isfinite(intervals) & (intervals > 0)]
        if not positive.size:
            continue
        dt = float(np.median(positive))
        if float((np.percentile(positive, 95) - np.percentile(positive, 5)) / dt) > 0.05:
            continue
        gap_limit = max(0.0015, 1.5 * dt) + 1e-9
        fs_hz = 1.0 / dt
        window_samples = max(8, int(round(fs_hz * frame_seconds)))
        step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
        values = np.asarray([
            np.nan if row[branch_field] is None else float(row[branch_field])
            for row in rows
        ], dtype=float)
        common = np.asarray([bool(row["common_valid"]) for row in rows], dtype=bool)
        contiguous = np.zeros(rows.__len__(), dtype=bool)
        contiguous[0] = True
        contiguous[1:] = (np.diff([int(row["sample_index"]) for row in rows]) == 1) & (intervals <= gap_limit)
        # Every invalid sample starts a new run so filter edges are never crossed.
        groups: list[tuple[int, int]] = []
        group_start: int | None = None
        for index in range(len(rows) + 1):
            is_valid = index < len(rows) and common[index] and np.isfinite(values[index])
            continues = is_valid and index > 0 and contiguous[index] and common[index - 1] and np.isfinite(values[index - 1])
            if is_valid and group_start is None:
                group_start = index
            elif is_valid and group_start is not None and not continues:
                if index - group_start >= window_samples:
                    groups.append((group_start, index))
                group_start = index
            elif not is_valid and group_start is not None:
                if index - group_start >= window_samples:
                    groups.append((group_start, index))
                group_start = None
        frames = raw_frames.setdefault(series, [])
        for start, end in groups:
            for frame_start in range(start, end - window_samples + 1, step_samples):
                frame_end = frame_start + window_samples
                window = values[frame_start:frame_end]
                centered = window - float(np.mean(window))
                rms = float(np.sqrt(np.mean(centered**2)))
                frames.append({
                    "series": series,
                    "run_index": run_index,
                    "start_time_s": float(times[frame_start]),
                    "end_time_s": float(times[frame_end - 1] + dt),
                    "time_s": float((times[frame_start] + times[frame_end - 1]) / 2.0),
                    "rms_adc": rms,
                    "quality": "common_mask_valid",
                })

    output_frames: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for series, frames in sorted(raw_frames.items()):
        threshold = _otsu_threshold([float(row["rms_adc"]) for row in frames])
        for row in frames:
            row["threshold_rms_adc"] = threshold
            row["status"] = (
                "not_evaluable" if threshold is None else
                "candidate_activity" if float(row["rms_adc"]) > threshold else
                "below_threshold"
            )
        output_frames.extend(frames)
        active = [row for row in frames if row["status"] == "candidate_activity"]
        current: list[dict[str, Any]] = []
        def emit_event() -> None:
            if not current:
                return
            start_s = float(current[0]["start_time_s"])
            end_s = float(max(row["end_time_s"] for row in current))
            rms_values = [float(row["rms_adc"]) for row in current]
            events.append({
                "series": series,
                "run_index": int(current[0]["run_index"]),
                "start_time_s": start_s,
                "end_time_s": end_s,
                "duration_s": end_s - start_s,
                "threshold_rms_adc": float(current[0]["threshold_rms_adc"]),
                "mean_rms_adc": float(np.mean(rms_values)),
                "peak_rms_adc": float(np.max(rms_values)),
                "quality": "common_mask_valid",
                "interpretation": "candidate_activity_only",
            })
            current.clear()
        for row in active:
            if current and (
                row["run_index"] != current[-1]["run_index"]
                or float(row["start_time_s"]) > float(current[-1]["end_time_s"]) + 1e-9
            ):
                emit_event()
            current.append(row)
        emit_event()
    output_frames.sort(key=lambda row: (row["series"], row["run_index"], row["start_time_s"]))
    events.sort(key=lambda row: (row["series"], row["run_index"], row["start_time_s"]))
    return output_frames, events

_HTML = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Explorador local de EMG</title>
<style>
:root{color-scheme:light;--ink:#182638;--muted:#5c6d80;--blue:#139ac7;--cyan:#17b8d0;--orange:#e59138;--green:#36a77b;--red:#d87557;--line:#dbe4ee;--panel:#fff;--bg:#f3f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "Segoe UI",Arial,sans-serif}
header{background:#102339;color:white;padding:22px max(22px,calc((100vw - 1220px)/2));box-shadow:0 2px 10px #18263822}h1{font-size:24px;margin:0 0 5px}header p{color:#cad9e6;margin:0;max-width:1050px}
main{max-width:1220px;margin:22px auto;padding:0 18px}.notice{background:#fff7e7;border:1px solid #f0d7a5;border-left:5px solid #dfa344;padding:12px 15px;border-radius:8px;margin-bottom:16px;color:#594523}
.toolbar,.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #1935500a}.toolbar{padding:14px;display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin-bottom:16px}.field{display:grid;gap:4px;color:var(--muted);font-size:12px}select,input,button{font:inherit;color:var(--ink);border:1px solid #c8d4e0;border-radius:6px;padding:8px 10px;background:white}button{background:#176c98;color:white;border-color:#176c98;font-weight:600;cursor:pointer}button:hover{background:#105677}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{padding:13px 14px;min-width:0}.wide{grid-column:1/-1}h2{font-size:16px;margin:0 0 5px}.sub{color:var(--muted);font-size:12px;margin:0 0 8px}.chart{display:block;width:100%;height:250px}.chart.tall{height:300px}.chart.short{height:145px}.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin:4px 0 0 44px}.legend span:before{content:"";display:inline-block;width:11px;height:3px;background:var(--color);vertical-align:middle;margin-right:5px}.stats{color:var(--muted);font-size:13px}.empty{padding:25px;text-align:center;color:var(--muted)}footer{margin:18px 2px 30px;color:var(--muted);font-size:12px}
@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.chart{height:220px}.chart.tall{height:250px}}
</style></head><body>
<header><h1>Explorador de señal EMG</h1><p>Procesamiento local por grabación: comparación de señal almacenada y rama filtrada, indicadores por ventana, espectrograma y máscara de calidad. Valores de amplitud en cuentas ADC; el canal no tiene calibración física confirmada.</p></header>
<main><div id="loadError" class="notice" role="status" style="display:none"></div><div class="notice"><strong>Alcance:</strong> este tablero no incluye un score de fatiga validado ni probabilidades de un modelo. No existe una etiqueta independiente compatible con esta señal para entrenar ese objetivo. Los descensos espectrales son indicadores exploratorios; no prueban pérdida de fuerza, repeticiones ni fatiga funcional.</div>
<section class="toolbar">
<label class="field">Serie<select id="series"></select></label>
<label class="field">Rama<select id="branch"><option value="control">Señal almacenada (control)</option><option value="hp20">Butterworth HP20</option><option value="hp40">Butterworth HP40</option><option value="bp20_400">Butterworth BP20–400</option><option value="fir_bp20_400">FIR Hamming BP20–400</option></select></label>
<label class="field">Inicio (s)<input id="t0" type="number" step="0.1"></label>
<label class="field">Fin (s)<input id="t1" type="number" step="0.1"></label>
<button id="apply">Aplicar intervalo</button><span id="rangeInfo" class="stats"></span>
</section>
<section class="grid">
<article class="card wide"><h2>Señal original y filtrada</h2><p class="sub">Eje X: tiempo (s). Eje Y: amplitud (cuentas ADC, sin conversión validada a mV).</p><canvas id="signal" class="chart tall"></canvas><div class="legend"><span style="--color:var(--muted)">Almacenada</span><span style="--color:var(--cyan)">Procesada</span></div></article>
<article class="card"><h2>RMS por ventana</h2><p class="sub">Amplitud efectiva, ADC. No se interpreta en solitario como fatiga.</p><canvas id="rms" class="chart"></canvas></article>
<article class="card"><h2>MDF y MNF</h2><p class="sub">Frecuencia mediana y media de la PSD, Hz; ventanas sin datos válidos no aparecen.</p><canvas id="freq" class="chart"></canvas><div class="legend"><span style="--color:var(--blue)">MDF</span><span style="--color:var(--orange)">MNF</span></div></article>
<article class="card"><h2>FFT de ventana</h2><p class="sub">Espectro unilateral de una ventana válida de 1 s; selecciona la ventana para inspeccionarla.</p><label class="field">Ventana<select id="fftWindow"></select></label><canvas id="fft" class="chart"></canvas></article>
<article class="card"><h2>PSD promedio por tramo</h2><p class="sub">Promedio de periodogramas por tramo continuo, sin unir discontinuidades. Unidades: ADC²/Hz.</p><canvas id="psd" class="chart"></canvas><div id="psdLegend" class="legend"></div></article>
<article class="card"><h2>STFT</h2><p class="sub">Eje X: tiempo (s). Eje Y: frecuencia (Hz). Color: log PSD (ADC²/Hz). Solo ventanas generadas en tramos procesables.</p><canvas id="stft" class="chart tall"></canvas></article>
<article class="card wide"><h2>Envolvente candidata y Gantt de calidad</h2><p class="sub">Otsu se usa solo para separar ventanas candidatas según RMS. “Por debajo” no confirma reposo; “por encima” no confirma contracción ni repetición. No equivale a reposo ni repetición verificados. Gris: bordes o datos no evaluables; la máscara común resume calidad de procesamiento, no calidad fisiológica. Este flujo aún no identifica artefactos de movimiento ni saturación del ADC.</p><canvas id="gantt" class="chart short"></canvas><div id="quality" class="stats"></div><h3>Eventos candidatos</h3><div class="sub">Intervalos calculados en tramos continuos de la rama BP20–400, con ventanas RMS de 50 ms y 50 % de solapamiento.</div><div id="events" class="stats"></div></article>
</section><footer>Los gráficos se generan a partir de las filas de esta grabación. Los tramos con huecos se mantienen separados; el tablero no interpola discontinuidades. Guarda el HTML y sus resultados en almacenamiento local restringido.</footer></main>
<script>
const DATA_GZIP_B64="__DATA_GZIP_B64__";
async function bootDashboard(){try{if(!(typeof DecompressionStream!=="function"))throw new Error("unsupported browser");const bytes=Uint8Array.from(atob(DATA_GZIP_B64),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));const DATA=JSON.parse(await new Response(stream).text());initializeDashboard(DATA)}catch(e){const node=document.getElementById("loadError");node.style.display="block";node.textContent="No se pudo abrir el visor en este navegador. Prueba con una versión actualizada de Edge, Chrome o Firefox."}}
function initializeDashboard(DATA){
const branchNames={control:"control_adc",hp20:"hp20_adc",hp40:"hp40_adc",bp20_400:"bp20_400_adc",fir_bp20_400:"fir_bp20_400_adc"};
const branchLabels={control:"Señal almacenada",hp20:"Butterworth HP20",hp40:"Butterworth HP40",bp20_400:"Butterworth BP20–400",fir_bp20_400:"FIR Hamming BP20–400"};
const palette={raw:"#63788e",filtered:"#17b8d0",blue:"#139ac7",orange:"#e59138",valid:"#36a77b",invalid:"#cbd5df"};
const seriesSelect=document.getElementById("series"), branchSelect=document.getElementById("branch"), t0=document.getElementById("t0"), t1=document.getElementById("t1");
const seriesList=[...new Set(DATA.traces.map(r=>r.series))].sort();
for(const s of seriesList){const o=document.createElement("option");o.value=s;o.textContent=s;seriesSelect.appendChild(o)}
function rowsFor(rows,s,branch){return rows.filter(r=>r.series===s&&(!branch||r.filter_branch===branch))}
function rangeRows(){const s=seriesSelect.value;return rowsFor(DATA.traces,s)}
function currentRange(){let a=Number(t0.value),b=Number(t1.value);if(!Number.isFinite(a)||!Number.isFinite(b)||b<=a){const r=rangeRows();a=Math.min(...r.map(x=>x.time_s));b=Math.max(...r.map(x=>x.time_s));t0.value=a.toFixed(3);t1.value=b.toFixed(3)}return [a,b]}
function setup(id){const c=document.getElementById(id),rect=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.max(1,Math.floor(rect.width*dpr));c.height=Math.max(1,Math.floor(rect.height*dpr));const x=c.getContext("2d");x.scale(dpr,dpr);return {x,w:rect.width,h:rect.height}}
function decimate(points,maxN=1800){if(points.length<=maxN)return points;const bins=Math.max(1,Math.floor(maxN/2)),size=Math.ceil(points.length/bins),out=[];for(let i=0;i<points.length;i+=size){const part=points.slice(i,i+size),valid=part.filter(p=>Number.isFinite(p.y));if(!valid.length)continue;let lo=valid[0],hi=valid[0];for(const p of valid){if(p.y<lo.y)lo=p;if(p.y>hi.y)hi=p}const pair=[lo,hi].sort((a,b)=>a.x-b.x);out.push(...pair)}return out}
function drawLines(id,sets,xMin,xMax,yLabel){const {x,w,h}=setup(id),L=56,R=14,T=18,B=39,pw=w-L-R,ph=h-T-B;x.clearRect(0,0,w,h);const all=sets.flatMap(d=>d.points.filter(p=>p.x>=xMin&&p.x<=xMax&&Number.isFinite(p.y)));if(!all.length){x.fillStyle="#5c6d80";x.fillText("No hay ventanas válidas en este intervalo",L,T+30);return}let ymin=Math.min(...all.map(p=>p.y)),ymax=Math.max(...all.map(p=>p.y));if(ymax===ymin){const pad=Math.abs(ymax)*.05||1;ymin-=pad;ymax+=pad}else{const pad=(ymax-ymin)*.06;ymin-=pad;ymax+=pad}const X=v=>L+(v-xMin)/(xMax-xMin||1)*pw,Y=v=>T+ph-(v-ymin)/(ymax-ymin||1)*ph;
x.font="11px Segoe UI";x.strokeStyle="#e3eaf1";x.fillStyle="#526477";for(let i=0;i<=4;i++){const yy=T+i*ph/4,val=ymax-i*(ymax-ymin)/4;x.beginPath();x.moveTo(L,yy);x.lineTo(w-R,yy);x.stroke();x.fillText(val.toPrecision(3),4,yy+4)}x.strokeStyle="#8797a8";x.beginPath();x.moveTo(L,T);x.lineTo(L,T+ph);x.lineTo(w-R,T+ph);x.stroke();x.fillStyle="#32465b";x.fillText("Tiempo (s)",L+pw/2-24,h-7);x.save();x.translate(13,T+ph/2+20);x.rotate(-Math.PI/2);x.fillText(yLabel,0,0);x.restore();
for(const d of sets){const points=decimate(d.points.filter(p=>p.x>=xMin&&p.x<=xMax&&Number.isFinite(p.y)));x.strokeStyle=d.color;x.lineWidth=1.5;x.beginPath();let prev=null;for(const p of points){if(prev&&p.run!==prev.run){x.stroke();x.beginPath();prev=null}if(!prev)x.moveTo(X(p.x),Y(p.y));else x.lineTo(X(p.x),Y(p.y));prev=p}x.stroke()}}
function drawSignal(s,a,b){const all=rowsFor(DATA.traces,s).sort((u,v)=>u.run_index-v.run_index||u.sample_index-v.sample_index),branch=branchNames[branchSelect.value];drawLines("signal",[{color:palette.raw,points:all.map(r=>({x:r.time_s,y:r.raw_adc,run:r.run_index}))},{color:palette.filtered,points:all.map(r=>({x:r.time_s,y:r[branch],run:r.run_index}))}],a,b,"Amplitud (ADC)")}
function drawFeatures(s,a,b){const branch=branchSelect.value,rows=rowsFor(DATA.features,s,branch).sort((u,v)=>u.time_s-v.time_s);drawLines("rms",[{color:palette.blue,points:rows.map(r=>({x:r.time_s,y:r.rms_adc,run:r.run_index}))}],a,b,"RMS (ADC)");drawLines("freq",[{color:palette.blue,points:rows.map(r=>({x:r.time_s,y:r.mdf_hz,run:r.run_index}))},{color:palette.orange,points:rows.map(r=>({x:r.time_s,y:r.mnf_hz,run:r.run_index}))}],a,b,"Frecuencia (Hz)")}
function drawSpectrum(id,sets,yLabel){const {x,w,h}=setup(id),L=56,R=14,T=18,B=39,pw=w-L-R,ph=h-T-B;x.clearRect(0,0,w,h);const all=sets.flatMap(s=>s.points.filter(p=>Number.isFinite(p.y)));if(!all.length){x.fillStyle="#5c6d80";x.fillText("Sin espectro en el intervalo",L,T+30);return}const xmax=Math.max(...all.map(p=>p.x)),ymin=Math.min(...all.map(p=>p.y)),ymax=Math.max(...all.map(p=>p.y));const pad=ymax===ymin?(Math.abs(ymax)*.05||1):(ymax-ymin)*.06,lo=ymin-pad,hi=ymax+pad,X=v=>L+v/(xmax||1)*pw,Y=v=>T+ph-(v-lo)/(hi-lo||1)*ph;x.font="11px Segoe UI";x.strokeStyle="#e3eaf1";x.fillStyle="#526477";for(let i=0;i<=4;i++){const yy=T+i*ph/4,val=hi-i*(hi-lo)/4;x.beginPath();x.moveTo(L,yy);x.lineTo(w-R,yy);x.stroke();x.fillText(val.toPrecision(3),4,yy+4)}x.strokeStyle="#8797a8";x.beginPath();x.moveTo(L,T);x.lineTo(L,T+ph);x.lineTo(w-R,T+ph);x.stroke();x.fillStyle="#32465b";x.fillText("Frecuencia (Hz)",L+pw/2-38,h-7);x.save();x.translate(13,T+ph/2+20);x.rotate(-Math.PI/2);x.fillText(yLabel,0,0);x.restore();for(const set of sets){x.strokeStyle=set.color;x.lineWidth=1.5;x.beginPath();const pts=decimate(set.points);for(let i=0;i<pts.length;i++){const p=pts[i];if(i===0)x.moveTo(X(p.x),Y(p.y));else x.lineTo(X(p.x),Y(p.y))}x.stroke()}}
function drawSpectra(s,a,b){const branch=branchSelect.value,all=DATA.fft.filter(r=>r.series===s&&r.filter_branch===branch&&r.time_s>=a&&r.time_s<=b);const selector=document.getElementById("fftWindow");const keys=[...new Set(all.map(r=>`${r.run_index}:${r.window_index}:${r.time_s}`))];const previous=selector.value;selector.replaceChildren();for(const key of keys){const [run,win,time]=key.split(":");const option=document.createElement("option");option.value=key;option.textContent=`tramo ${run} · ventana ${win} · ${Number(time).toFixed(2)} s`;selector.appendChild(option)}if(keys.includes(previous))selector.value=previous;const key=selector.value,chosen=all.filter(r=>`${r.run_index}:${r.window_index}:${r.time_s}`===key);drawSpectrum("fft",[{color:palette.blue,points:chosen.map(r=>({x:r.frequency_hz,y:r.amplitude_adc}))}],"Amplitud (ADC)");const grouped=new Map();for(const row of DATA.psd.filter(r=>r.series===s&&r.filter_branch===branch)){if(!grouped.has(row.run_index))grouped.set(row.run_index,[]);grouped.get(row.run_index).push({x:row.frequency_hz,y:Math.log10(Math.max(Number(row.psd_adc2_per_hz),1e-15))})}const colors=["#139ac7","#e59138","#36a77b","#d87557","#8a68a8","#4b9a8e"],psdSets=[...grouped.entries()].map(([run,points],i)=>({color:colors[i%colors.length],points,run}));drawSpectrum("psd",psdSets,"log10 PSD (ADC²/Hz)");const legend=document.getElementById("psdLegend");legend.replaceChildren();for(const set of psdSets){const span=document.createElement("span");span.textContent=`Tramo ${set.run}`;span.style.setProperty("--color",set.color);legend.appendChild(span)}}
function drawStft(s,a,b){const {x,w,h}=setup("stft"),L=56,R=38,T=18,B=38,pw=w-L-R,ph=h-T-B,branch=branchSelect.value;const rows=rowsFor(DATA.stft,s,branch).filter(r=>r.time_s>=a&&r.time_s<=b&&r.frequency_hz<=400);x.clearRect(0,0,w,h);if(!rows.length){x.fillStyle="#5c6d80";x.fillText("No hay espectrograma para este intervalo",L,T+25);return}const ts=[...new Set(rows.map(r=>r.time_s))].sort((u,v)=>u-v),fs=[...new Set(rows.map(r=>r.frequency_hz))].sort((u,v)=>u-v);const vals=rows.map(r=>Math.log10(Math.max(Number(r.psd_adc2_per_hz),1e-15))),vmin=Math.min(...vals),vmax=Math.max(...vals);let hop=.5;if(ts.length>1){const ds=[];for(let i=1;i<ts.length;i++)if(ts[i]>ts[i-1])ds.push(ts[i]-ts[i-1]);ds.sort((u,v)=>u-v);hop=ds[Math.floor(ds.length/2)]||.5}const cellW=Math.max(1,hop/(b-a||1)*pw+1),df=fs.length>1?fs[1]-fs[0]:1,cellH=Math.max(1,df/400*ph+1);for(const r of rows){const v=Math.log10(Math.max(Number(r.psd_adc2_per_hz),1e-15)),q=(v-vmin)/(vmax-vmin||1),red=Math.round(35+220*q),green=Math.round(70+110*q),blue=Math.round(170-145*q);x.fillStyle=`rgb(${red},${green},${blue})`;const xx=L+(r.time_s-a)/(b-a||1)*pw,yy=T+ph-r.frequency_hz/400*ph;x.fillRect(xx,yy-cellH/2,cellW,cellH)}x.strokeStyle="#8797a8";x.beginPath();x.moveTo(L,T);x.lineTo(L,T+ph);x.lineTo(w-R,T+ph);x.stroke();x.fillStyle="#32465b";x.font="11px Segoe UI";x.fillText("Tiempo (s)",L+pw/2-24,h-7);x.save();x.translate(13,T+ph/2+18);x.rotate(-Math.PI/2);x.fillText("Frecuencia (Hz)",0,0);x.restore();x.fillStyle="#526477";x.fillText("400",L-27,T+5);x.fillText("0",L-14,T+ph+3);x.fillText("alta",w-33,T+12);x.fillText("baja",w-33,T+ph)}
function drawGantt(s,a,b){const {x,w,h}=setup("gantt"),L=128,R=14,T=10,B=24,pw=w-L-R,ph=h-T-B,rows=rowsFor(DATA.traces,s).sort((u,v)=>u.run_index-v.run_index||u.sample_index-v.sample_index),frames=DATA.activity_frames.filter(r=>r.series===s&&r.start_time_s<=b&&r.end_time_s>=a);x.clearRect(0,0,w,h);const lanes=["Actividad candidata: RMS > umbral","RMS ≤ umbral (no reposo)","Máscara común válida"],laneH=ph/3;x.font="10px Segoe UI";for(let l=0;l<lanes.length;l++){const y=T+l*laneH;x.fillStyle="#526477";x.fillText(lanes[l],3,y+laneH/2+3);x.fillStyle=l===2?palette.valid:"#edf1f5";x.fillRect(L,y,pw,laneH-3)}for(const f of frames){const lane=f.status==="candidate_activity"?0:f.status==="below_threshold"?1:-1;if(lane<0)continue;const xx=L+(Math.max(a,f.start_time_s)-a)/(b-a||1)*pw,xe=L+(Math.min(b,f.end_time_s)-a)/(b-a||1)*pw;x.fillStyle=lane===0?palette.valid:palette.orange;x.fillRect(xx,T+lane*laneH,Math.max(1,xe-xx),laneH-3)}let start=null,prev=null,state=null;function flush(){if(start&&prev&&!state){const xx=L+(start.time_s-a)/(b-a||1)*pw,xe=L+(prev.time_s-a)/(b-a||1)*pw;x.fillStyle=palette.invalid;x.fillRect(xx,T+2*laneH,Math.max(1,xe-xx+1),laneH-3)}}for(const r of rows){if(r.time_s<a||r.time_s>b)continue;const val=Boolean(r.common_valid);if(!prev||r.run_index!==prev.run_index||val!==state||r.sample_index!==prev.sample_index+1){flush();start=r;state=val}else if(start===null){start=r}prev=r}flush();x.strokeStyle="#8797a8";x.beginPath();x.moveTo(L,T+ph);x.lineTo(w-R,T+ph);x.stroke();x.font="10px Segoe UI";x.fillStyle="#526477";for(let tick=0;tick<=4;tick++){const xx=L+tick*pw/4;x.beginPath();x.moveTo(xx,T+ph);x.lineTo(xx,T+ph+4);x.stroke();x.fillText((a+(b-a)*tick/4).toFixed(2),xx-14,T+ph+15)}x.fillText("Tiempo (s)",L+pw/2-22,h-1);const valid=rows.filter(r=>r.time_s>=a&&r.time_s<=b&&r.common_valid).length,total=rows.filter(r=>r.time_s>=a&&r.time_s<=b).length;document.getElementById("quality").textContent=`${valid.toLocaleString()} de ${total.toLocaleString()} muestras con máscara común válida; actividad candidata sobre ${frames.length.toLocaleString()} ventanas de envolvente.`;const events=DATA.activity_events.filter(r=>r.series===s&&r.start_time_s<=b&&r.end_time_s>=a);const box=document.getElementById("events");box.replaceChildren();if(!events.length){box.textContent="Sin intervalos candidatos en el recorte seleccionado.";return}const table=document.createElement("table");table.style.cssText="border-collapse:collapse;width:100%;font-size:12px";const head=document.createElement("tr");for(const label of ["Tramo","Inicio (s)","Fin (s)","Duración (s)","RMS medio (ADC)","RMS pico (ADC)","Calidad"]){const th=document.createElement("th");th.textContent=label;th.style.cssText="text-align:left;border-bottom:1px solid #dbe4ee;padding:5px";head.appendChild(th)}table.appendChild(head);for(const ev of events){const tr=document.createElement("tr");for(const value of [ev.run_index,Number(ev.start_time_s).toFixed(3),Number(ev.end_time_s).toFixed(3),Number(ev.duration_s).toFixed(3),Number(ev.mean_rms_adc).toPrecision(4),Number(ev.peak_rms_adc).toPrecision(4),"máscara común válida"]){const td=document.createElement("td");td.textContent=String(value);td.style.cssText="padding:5px;border-bottom:1px solid #edf1f5";tr.appendChild(td)}table.appendChild(tr)}box.appendChild(table)}
function render(){const s=seriesSelect.value,r=rangeRows();if(!r.length)return;let a=Number(t0.value),b=Number(t1.value);const min=Math.min(...r.map(x=>x.time_s)),max=Math.max(...r.map(x=>x.time_s));if(!Number.isFinite(a)||!Number.isFinite(b)||b<=a){a=min;b=max;t0.value=a.toFixed(3);t1.value=b.toFixed(3)}a=Math.max(min,a);b=Math.min(max,b);if(b<=a){a=min;b=max}document.getElementById("rangeInfo").textContent=`${a.toFixed(2)}–${b.toFixed(2)} s · ${branchLabels[branchSelect.value]}`;drawSignal(s,a,b);drawFeatures(s,a,b);drawSpectra(s,a,b);drawStft(s,a,b);drawGantt(s,a,b)}
document.getElementById("apply").addEventListener("click",render);seriesSelect.addEventListener("change",()=>{t0.value="";t1.value="";render()});branchSelect.addEventListener("change",render);document.getElementById("fftWindow").addEventListener("change",render);window.addEventListener("resize",render);render();
}
bootDashboard();
</script></body></html>"""


def _project_rows(
    rows: list[dict[str, Any]], fields: tuple[str, ...], *, table_name: str
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        if IDENTITY_FIELDS.intersection(row):
            raise ValueError(f"dashboard {table_name} contains identity fields")
        projected_row: dict[str, Any] = {}
        for name in fields:
            if name not in row:
                raise ValueError(f"dashboard {table_name} is missing {name}")
            value = row[name]
            if name in {"series", "filter_branch", "status"}:
                if not isinstance(value, str):
                    raise ValueError(f"dashboard {table_name} has a non-text {name}")
                projected_row[name] = value
            elif name == "common_valid":
                projected_row[name] = bool(value)
            else:
                if value is None:
                    projected_row[name] = None
                else:
                    number = float(value)
                    projected_row[name] = number if math.isfinite(number) else None
        projected.append(projected_row)
    return projected


def render_dashboard_html(
    trace_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    stft_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    fft_rows: list[dict[str, Any]] | None = None,
    psd_rows: list[dict[str, Any]] | None = None,
    activity_frames: list[dict[str, Any]] | None = None,
    activity_events: list[dict[str, Any]] | None = None,
) -> str:
    """Return a single-file interactive dashboard with no participant IDs or scores."""
    if not trace_rows:
        raise ValueError("dashboard requires trace rows from one recording or selected cohort group")
    payload = {
        "traces": _project_rows(trace_rows, TRACE_FIELDS, table_name="trace rows"),
        "features": _project_rows(feature_rows, FEATURE_FIELDS, table_name="feature rows"),
        "stft": _project_rows(stft_rows, STFT_FIELDS, table_name="STFT rows"),
        "fft": _project_rows(fft_rows or [], FFT_FIELDS, table_name="FFT rows"),
        "psd": _project_rows(psd_rows or [], PSD_FIELDS, table_name="PSD rows"),
        "segments": _project_rows(segment_rows, SEGMENT_FIELDS, table_name="segment rows"),
    }
    if activity_frames is None or activity_events is None:
        activity_frames, activity_events = derive_candidate_activity(payload["traces"])
    payload["activity_frames"] = activity_frames
    payload["activity_events"] = activity_events
    json_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    compressed = gzip.compress(json_payload.encode("utf-8"), compresslevel=6, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    return _HTML.replace("__DATA_GZIP_B64__", encoded)
